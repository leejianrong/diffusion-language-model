"""Pod launcher. Wraps the runpod-jobs `rp` script, adding the budget gate and the ledger.

`rp` owns the pod lifecycle and bakes the pod-side dead-man's-switch (max lifetime plus an idle
watchdog) into the start command, so teardown does not depend on this machine staying connected.
This module decides whether a launch is allowed, records it, and records the stop only after
verifying the pod is really gone.
"""

import json
import os
import re
import shlex
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from dlm.budget import Ledger

CompletedLike = subprocess.CompletedProcess[str]
Runner = Callable[[list[str], Mapping[str, str]], CompletedLike]


class PodError(RuntimeError):
    pass


@dataclass(frozen=True)
class PodPlan:
    name: str  # unique per project, so we never touch another project's pods
    image: str
    job_cmd: str
    max_hourly_usd: float
    max_lifetime_secs: int
    compute_type: str = "GPU"
    gpu_types: list[str] = field(default_factory=list)
    idle_timeout_secs: int = 900
    container_disk_gb: int = 50
    volume_gb: int = 0
    cloud_type: str = "SECURE"

    @property
    def planned_usd(self) -> float:
        """Worst case: the price cap for the whole lifetime ceiling."""
        return self.max_hourly_usd * self.max_lifetime_secs / 3600

    def env(self, state_file: Path) -> dict[str, str]:
        e = {
            "RP_POD_NAME": self.name,
            "RP_POD_IMAGE": self.image,
            "RP_JOB_CMD": self.job_cmd,
            "RP_COMPUTE_TYPE": self.compute_type,
            "RP_MAX_HOURLY_USD": str(self.max_hourly_usd),
            "RP_MAX_LIFETIME_SECS": str(self.max_lifetime_secs),
            "RP_IDLE_TIMEOUT_SECS": str(self.idle_timeout_secs),
            "RP_CONTAINER_DISK_GB": str(self.container_disk_gb),
            "RP_VOLUME_GB": str(self.volume_gb),
            "RP_CLOUD_TYPE": self.cloud_type,
            "RP_INTERRUPTIBLE": "false",  # spot pods are no longer offered (runpod-jobs notes)
            "RP_STATE_FILE": str(state_file),
        }
        if self.gpu_types:
            e["RP_GPU_TYPE"] = ",".join(self.gpu_types)
        return e

    @staticmethod
    def from_yaml(path: str | Path) -> "PodPlan":
        return PodPlan(**yaml.safe_load(Path(path).read_text()))


def rp_command() -> list[str]:
    """`rp` may need the API key supplied by reference, e.g. DLM_RP='with-runpod-key rp'."""
    return shlex.split(os.environ.get("DLM_RP", "rp"))


def subprocess_runner(cmd: list[str], env: Mapping[str, str]) -> CompletedLike:
    return subprocess.run(
        cmd, env={**os.environ, **env}, capture_output=True, text=True, check=False
    )


def runpodctl_pod_names(run: Runner = subprocess_runner) -> list[str]:
    """Names of every pod on the account. Parsed from JSON and field-selected, never echoed."""
    r = run(["runpodctl", "pod", "list"], {})
    if r.returncode != 0:
        raise PodError("could not list pods; refusing to assume the account is clear")
    try:
        pods: Any = json.loads(r.stdout)
        return [str(p["name"]) for p in pods]
    except (ValueError, KeyError, TypeError) as e:
        raise PodError("unparseable pod list; refusing to assume the account is clear") from e


def _hourly_from_status(text: str, fallback: float) -> float:
    """`rp status` prints 'pod <id>: <state> <cost>'. Fall back to the cap if unparseable."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*$", text.strip())
    return float(m.group(1)) if m else fallback


def launch(
    plan: PodPlan, ledger: Ledger, state_file: Path, run: Runner = subprocess_runner
) -> dict[str, Any]:
    if ledger.running(plan.name):
        raise PodError(f"ledger already shows a running '{plan.name}' pod; stop it first")
    ledger.check(plan.planned_usd)  # raises BudgetExceededError
    env, rp = plan.env(state_file), rp_command()
    up = run([*rp, "up"], env)
    if up.returncode != 0:
        raise PodError(f"rp up failed: {up.stderr.strip()[-400:]}")
    pod_id = up.stdout.strip().splitlines()[-1]
    try:
        st = run([*rp, "status"], env)
        hourly = (
            _hourly_from_status(st.stdout, plan.max_hourly_usd)
            if st.returncode == 0
            else plan.max_hourly_usd
        )
        ledger.start(pod_id, plan.name, hourly, plan.max_lifetime_secs / 3600)
    except Exception:
        run([*rp, "down"], env)  # never leave a pod running that the ledger does not know about
        raise
    return {"pod_id": pod_id, "hourly_usd": hourly, **ledger.summary()}


def stop(
    plan: PodPlan,
    ledger: Ledger,
    state_file: Path,
    run: Runner = subprocess_runner,
    list_names: Callable[[], list[str]] | None = None,
) -> dict[str, Any]:
    """Terminate, verify the pod is gone, then record the stop. Idempotent."""
    env, rp = plan.env(state_file), rp_command()
    down = run([*rp, "down"], env)
    if down.returncode != 0:
        raise PodError(f"rp down failed, ledger left running: {down.stderr.strip()[-400:]}")
    names = (list_names or (lambda: runpodctl_pod_names(run)))()
    if plan.name in names:
        raise PodError(f"pod '{plan.name}' still present after delete, ledger left running")
    for rec in ledger.running(plan.name):
        ledger.stop(rec.pod_id)
    return ledger.summary()
