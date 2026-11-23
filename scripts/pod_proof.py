"""Live proof that the pod-side dead-man's-switch works. SPENDS REAL MONEY (a few cents).

Two scenarios on a cheap CPU pod, each with its own pod:
  idle      the job goes quiet; the idle watchdog must terminate the pod itself
  lifetime  the job keeps logging; the hard max-lifetime must terminate the pod itself

The driver is the second layer: a `finally` block always terminates and verifies, so a failed
switch is caught and reported rather than left billing. Run: uv run python scripts/pod_proof.py
"""

import json
import signal
import sys
import time
from pathlib import Path
from typing import Any

from dlm.budget import Ledger
from dlm.pod import PodPlan, launch, pod_names, stop
from dlm.runpod import Client

CPU = {"id": "cpu3c", "vcpuCount": 2}
IMAGE = "python:3.12-slim"
MARGIN = 150  # seconds past the expected self-termination before the driver gives up and fails it

SCENARIOS: dict[str, dict[str, Any]] = {
    "idle": dict(
        job_cmd="echo proof: has_pod_id=${RUNPOD_POD_ID:+yes} "
        "has_api_key=${RUNPOD_API_KEY:+yes}; sleep 100000",
        idle_timeout_secs=150,
        idle_check_secs=20,
        max_lifetime_secs=900,  # backstop only, must not be the thing that fires
        expect_secs=150 + 20,
    ),
    "lifetime": dict(
        job_cmd='while true; do echo "tick $(date +%s)"; sleep 5; done',
        idle_timeout_secs=900,
        idle_check_secs=20,
        max_lifetime_secs=150,
        expect_secs=150,
    ),
}


def run_scenario(name: str, client: Client, ledger: Ledger) -> dict[str, Any]:
    cfg = dict(SCENARIOS[name])
    expect = cfg.pop("expect_secs")
    plan = PodPlan(
        name=f"dlm-proof-{name}", image=IMAGE, max_hourly_usd=0.20, cpu=CPU, disk_gb=10, **cfg
    )
    result: dict[str, Any] = {"scenario": name, "expected_secs": expect}
    t0 = time.monotonic()
    launched = False
    try:
        info = launch(plan, ledger, client)
        launched = True
        pod_id = info["pod_id"]
        result.update(pod_id=pod_id, hourly_usd=info["hourly_usd"])
        print(f"[{name}] launched {pod_id} at ${info['hourly_usd']}/h", flush=True)
        flags_checked = False
        while True:
            elapsed = time.monotonic() - t0
            present = plan.name in pod_names(client)
            if not present:
                result.update(self_terminated=True, seconds_to_gone=round(elapsed))
                print(f"[{name}] pod gone by itself after {elapsed:.0f}s", flush=True)
                break
            if elapsed > expect + MARGIN:
                result.update(self_terminated=False, seconds_to_gone=None)
                print(
                    f"[{name}] FAIL: still present after {elapsed:.0f}s; driver will terminate",
                    flush=True,
                )
                break
            if name == "idle" and not flags_checked and elapsed > 75:
                flags_checked = True
                try:
                    text = client.logs_text(pod_id)[-600:]
                except Exception as e:
                    text = f"(logs unavailable: {type(e).__name__})"
                result["logs_excerpt"] = text
                print(f"[{name}] logs: {text}", flush=True)
                if "has_api_key=yes" not in text and "logs unavailable" not in text:
                    result.update(
                        self_terminated=False, seconds_to_gone=None, note="no API key in pod"
                    )
                    print(
                        f"[{name}] FAIL: pod has no API key to terminate itself; aborting",
                        flush=True,
                    )
                    break
            time.sleep(10)
    finally:
        if launched:
            summary = stop(plan, ledger, client)  # layer 2: always terminate and verify
            result["ledger_after"] = summary
    return result


def main() -> int:
    client, ledger = Client(), Ledger(Path("budget.json"))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # so `finally` runs on TERM too
    signal.signal(signal.SIGALRM, lambda *_: sys.exit(142))  # hard overall deadline, then `finally`
    signal.alarm(1500)
    before = pod_names(client)
    print(f"pods on account before: {len(before)} {before}", flush=True)
    results = [run_scenario(n, client, ledger) for n in SCENARIOS]
    after = pod_names(client)
    print(
        json.dumps({"results": results, "pods_after": after, "ledger": ledger.summary()}, indent=1)
    )
    ok = all(r.get("self_terminated") for r in results) and not [
        n for n in after if n.startswith("dlm-proof")
    ]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
