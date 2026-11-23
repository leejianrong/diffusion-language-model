"""Pod launcher on the RunPod REST v2 client, with the budget gate and ledger (ADR-0005).

Teardown has three layers, none of which depends on this machine staying connected:
1. the pod runs a dead-man's-switch as its very first act (hard max lifetime plus an idle watchdog),
   inlined as literal shell with no network fetch before it is armed;
2. `stop` terminates, then verifies the pod is really gone before the ledger records the stop;
3. the account spend limit in the RunPod console (set by the owner, outside this code).
"""

import base64
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from dlm.budget import Ledger
from dlm.runpod import Client


class PodError(RuntimeError):
    pass


@dataclass(frozen=True)
class PodPlan:
    name: str  # unique per project, so we never touch another project's pods
    image: str
    max_hourly_usd: float
    max_lifetime_secs: int
    job_cmd: str = ""
    job_file: str = ""  # a script shipped inside the create request (no fetch before arming)
    cpu: dict[str, Any] | None = None  # {"id": "cpu3c", "vcpuCount": 2}
    gpu: dict[str, Any] | None = None  # passed through to the API; validated in the smoke slice
    idle_timeout_secs: int = 900
    idle_check_secs: int = 60
    exit_grace_secs: int = 30
    disk_gb: int = 20
    cloud: str = "SECURE"
    env: dict[str, str] = field(default_factory=dict)

    @property
    def planned_usd(self) -> float:
        """Worst case: the price cap for the whole lifetime ceiling."""
        return self.max_hourly_usd * self.max_lifetime_secs / 3600

    @property
    def command(self) -> str:
        """The job: job_cmd, or job_file embedded as base64 so nothing is fetched at boot."""
        if not self.job_file:
            return self.job_cmd
        blob = base64.b64encode(Path(self.job_file).read_bytes()).decode()
        return f"echo {blob} | base64 -d > /tmp/job.sh && bash /tmp/job.sh"

    @staticmethod
    def from_yaml(path: str | Path) -> "PodPlan":
        return PodPlan(**yaml.safe_load(Path(path).read_text()))

    def start_script(self) -> str:
        """Dead-man's-switch first, then the job. Needs only bash and python3 on the image."""
        return f"""set -u
LOG=/tmp/dlm-job.log ; : > "$LOG"
terminate() {{
  echo "pod: self-terminating ($1)" >&2
  python3 - >&2 <<'PY' || echo "pod: self-terminate FAILED" >&2
import os, urllib.request as u
req = u.Request("https://api.runpod.io/v2/pods/" + os.environ["RUNPOD_POD_ID"], method="DELETE",
                headers={{"Authorization": "Bearer " + os.environ["RUNPOD_API_KEY"]}})
print("pod: terminate http", u.urlopen(req, timeout=20).status)
PY
  kill -TERM -1 2>/dev/null || true
}}
( sleep {self.max_lifetime_secs} ; terminate "max-lifetime {self.max_lifetime_secs}s" ) &
( {self.command} 2>&1 | tee "$LOG" ) &
JOB=$!
(
  last=0 ; changed=$(date +%s)
  while kill -0 "$JOB" 2>/dev/null; do
    sleep {self.idle_check_secs}
    size=$(wc -c < "$LOG" 2>/dev/null || echo 0) ; now=$(date +%s)
    if [ "$size" != "$last" ]; then last=$size ; changed=$now
    elif [ $((now - changed)) -ge {self.idle_timeout_secs} ]; then
      terminate "idle {self.idle_timeout_secs}s" ; break
    fi
  done
) &
wait "$JOB" ; sleep {self.exit_grace_secs} ; terminate "job exited"
"""

    def create_body(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "name": self.name,
            "image": self.image,
            "cloud": self.cloud,
            "disk": self.disk_gb,
            "entrypoint": ["bash", "-c"],
            "cmd": [self.start_script()],
        }
        if self.cpu:
            body["cpu"] = self.cpu
        if self.gpu:
            body["gpu"] = self.gpu
        if self.env:
            body["env"] = self.env
        return body


def _hourly(pod: dict[str, Any], fallback: float) -> float:
    """Best-effort hourly price from the pod record; fall back to the plan's cap (conservative)."""
    cost = pod.get("cost")
    if isinstance(cost, int | float):
        return float(cost)
    if isinstance(cost, dict):
        for k in ("perHr", "hourly", "perHour", "costPerHr", "price"):
            if isinstance(cost.get(k), int | float):
                return float(cost[k])
    return fallback


def pod_names(client: Client) -> list[str]:
    try:
        return [str(p["name"]) for p in client.list_pods()]
    except Exception as e:  # fail closed: an unreadable list never counts as "clear"
        raise PodError(
            f"could not list pods ({type(e).__name__}); not assuming the account is clear"
        ) from e


def launch(plan: PodPlan, ledger: Ledger, client: Client) -> dict[str, Any]:
    if ledger.running(plan.name):
        raise PodError(f"ledger already shows a running '{plan.name}' pod; stop it first")
    ledger.check(plan.planned_usd)  # raises BudgetExceededError before any API call
    pod = client.create_pod(plan.create_body())
    pod_id = str(pod["id"])
    try:
        info = client.get_pod(pod_id)
        hourly = _hourly(info, plan.max_hourly_usd)
        if hourly > plan.max_hourly_usd:
            raise PodError(f"pod costs ${hourly}/h, above the ${plan.max_hourly_usd}/h cap")
        ledger.start(pod_id, plan.name, hourly, plan.max_lifetime_secs / 3600)
    except Exception:
        client.terminate(pod_id)  # never leave a pod running that the ledger does not know about
        raise
    return {"pod_id": pod_id, "hourly_usd": hourly, **ledger.summary()}


def stop(
    plan: PodPlan,
    ledger: Ledger,
    client: Client,
    *,
    verify_secs: float = 90.0,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Terminate, verify the pod is gone, then record the stop. Idempotent."""
    ids = {r.pod_id for r in ledger.running(plan.name)}
    for pid in ids:
        client.terminate(pid)
    deadline = time.monotonic() + verify_secs
    while True:
        gone = plan.name not in pod_names(client)
        if gone:
            break
        if time.monotonic() >= deadline:
            raise PodError(f"pod '{plan.name}' still present after terminate, ledger left running")
        sleep(3)
    for pid in ids:
        ledger.stop(pid)
    return ledger.summary()
