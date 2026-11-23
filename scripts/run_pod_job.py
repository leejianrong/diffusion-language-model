"""Launch a pod plan, watch its log live, fail fast, and always terminate and verify.

Spends real money. Layers: the pod's own dead-man's-switch, this driver's `finally`, and the
budget gate. The driver prints the interesting log lines as they arrive and keeps every raw line in
.pod-logs/ so a failure can be diagnosed after the pod (and its logs) are gone.

Run: uv run python scripts/run_pod_job.py configs/pods/smoke.yaml [--ref <sha>]
"""

import argparse
import dataclasses
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

from dlm.budget import Ledger
from dlm.pod import PodPlan, launch, pod_names, stop
from dlm.runpod import Client

POLL_SECS = 20
FIRST_STAGE_WARN = 240  # no STAGE line this long after launch: say so
FIRST_STAGE_ABORT = 480  # ...and give up, the bootstrap is broken
SHOW = ("STAGE", "ERROR", "WARN", "progress", "RP_REPORT", "JOB_DONE", "Traceback", "Error", "{")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def pinned_ref() -> str:
    """HEAD, which must already be pushed: the pod clones it, it cannot see local commits."""
    sha = _git("rev-parse", "HEAD")
    if not _git("branch", "-r", "--contains", sha):
        raise SystemExit(f"commit {sha[:8]} is not pushed; the pod could not clone it")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("uncommitted changes: the pod would run different code from this tree")
    return sha


def lines_of(text: str) -> list[tuple[str, str]]:
    """(source, line) pairs from the log stream: SSE `data:` records holding {"source","line"}."""
    out = []
    for raw in text.splitlines():
        raw = raw.strip()
        if raw.startswith("data:"):
            raw = raw[5:].strip()
        if not raw.startswith("{"):
            continue  # SSE bookkeeping such as "id: <timestamp>"
        try:
            rec = json.loads(raw)
            out.append((str(rec["source"]), str(rec["line"])))
        except (ValueError, KeyError):
            out.append(("raw", raw))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--ref", default="", help="git ref for the pod to run (default: pushed HEAD)")
    a = ap.parse_args()
    plan = PodPlan.from_yaml(a.plan)
    ref = a.ref or pinned_ref()
    plan = dataclasses.replace(plan, env={**plan.env, "GIT_REF": ref})
    client, ledger = Client(), Ledger(Path("budget.json"))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    signal.signal(signal.SIGALRM, lambda *_: sys.exit(142))
    signal.alarm(plan.max_lifetime_secs + 300)  # hard overall deadline, then `finally` cleans up

    print(f"pods on account before: {pod_names(client)}", flush=True)
    log_dir = Path(".pod-logs")
    log_dir.mkdir(exist_ok=True)
    snap = log_dir / f"{plan.name}-{time.strftime('%Y%m%d-%H%M%S')}.log"
    print(f"launching {plan.name} at ref {ref[:8]}; raw log -> {snap}", flush=True)

    seen: set[str] = set()
    outcome = "unknown"
    pod_id = ""
    t0 = time.monotonic()
    launched = False
    hb = 0
    try:
        info = launch(plan, ledger, client)
        launched, pod_id = True, info["pod_id"]
        print(f"pod {pod_id} at ${info['hourly_usd']}/h; ledger {ledger.summary()}", flush=True)
        saw_stage = False
        done_at = 0.0
        while True:
            elapsed = time.monotonic() - t0
            if plan.name not in pod_names(client):
                outcome = outcome if outcome != "unknown" else "pod gone"
                print(f"[{elapsed:.0f}s] pod is gone ({outcome})", flush=True)
                break
            try:
                text = client.logs_text(pod_id, secs=6)
            except Exception as e:  # the pod may not be producing logs yet
                text = ""
                print(f"[{elapsed:.0f}s] logs unavailable ({type(e).__name__})", flush=True)
            fresh = [x for src, x in lines_of(text) if x not in seen and src != "system"]
            with snap.open("a") as f:
                for x in fresh:
                    seen.add(x)
                    f.write(x + "\n")
                    if x.startswith("hb "):
                        hb += 1
                        if hb % 4:  # hb every 30 s is chatty; show one in four
                            continue
                    if x.startswith(SHOW):
                        print(f"[{elapsed:.0f}s] {x[:300]}", flush=True)
            saw_stage = saw_stage or any(x.startswith("STAGE") for x in seen)
            blob = "\n".join(seen)
            if "JOB_DONE" in blob and not done_at:
                done_at = time.monotonic()
                outcome = "ok" if "JOB_DONE status=ok" in blob else "job failed"
            if "STAGE failed" in blob and outcome == "unknown":
                outcome = "job failed"
                done_at = done_at or time.monotonic()
            if done_at and time.monotonic() - done_at > 15:  # let the final lines arrive first
                break
            if not saw_stage and elapsed > FIRST_STAGE_WARN:
                print(f"[{elapsed:.0f}s] WARN no STAGE line yet", flush=True)
            if not saw_stage and elapsed > FIRST_STAGE_ABORT:
                outcome = "bootstrap never produced output"
                break
            if elapsed > plan.max_lifetime_secs + 60:
                outcome = "driver deadline"
                break
            time.sleep(POLL_SECS)
    finally:
        if launched:
            try:
                print("terminating and verifying...", flush=True)
                print(f"ledger after stop: {stop(plan, ledger, client)}", flush=True)
            except Exception as e:
                print(f"ERROR stop failed: {type(e).__name__}: {e}", flush=True)
                raise
    print(json.dumps({"outcome": outcome, "pod_id": pod_id, "secs": round(time.monotonic() - t0)}))
    print(f"pods on account after: {pod_names(client)}", flush=True)
    return 0 if outcome == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
