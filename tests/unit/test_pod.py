import subprocess
from pathlib import Path
from typing import Any

import pytest

from dlm.budget import BudgetExceededError, Ledger
from dlm.pod import PodError, PodPlan, launch, stop
from dlm.runpod import ApiError


class FakeClient:
    def __init__(self, cost: Any = 1.5, still_listed_for: int = 0) -> None:
        self.cost, self.listed_polls, self.calls = cost, still_listed_for, []  # type: ignore[var-annotated]
        self.pods: dict[str, str] = {}

    def create_pod(self, body: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("create")
        self.pods["pod123"] = body["name"]
        return {"id": "pod123"}

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        self.calls.append("get")
        return {"id": pod_id, "cost": self.cost}

    def terminate(self, pod_id: str) -> None:
        self.calls.append("terminate")
        if self.listed_polls <= 0:
            self.pods.pop(pod_id, None)

    def list_pods(self) -> list[dict[str, Any]]:
        self.calls.append("list")
        if self.listed_polls > 0:
            self.listed_polls -= 1
            if self.listed_polls == 0:
                self.pods.clear()
        return [{"id": i, "name": n} for i, n in self.pods.items()]


def plan(hourly: float = 2.0, secs: int = 3600) -> PodPlan:
    return PodPlan(name="dlm-test", image="img", job_cmd="echo hi", max_hourly_usd=hourly,
                   max_lifetime_secs=secs, cpu={"id": "cpu3c", "vcpuCount": 2})  # fmt: skip


def new_ledger(tmp_path: Path, cap: float = 60.0) -> Ledger:
    return Ledger(tmp_path / "budget.json", cap_usd=cap, now=lambda: 1_000_000.0)


def test_start_script_is_valid_bash_and_arms_the_switch_before_the_job() -> None:
    script = plan(secs=900).start_script()
    assert (
        subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True).returncode == 0
    )
    assert script.index("sleep 900") < script.index("echo hi")  # max-lifetime armed before the job
    assert script.index("terminate()") < script.index("echo hi")
    assert (
        "curl" not in script and "wget" not in script and "apt" not in script
    )  # nothing to hang on first


def test_create_body_requests_only_what_the_plan_says() -> None:
    b = plan().create_body()
    assert (
        b["name"] == "dlm-test" and b["cpu"] == {"id": "cpu3c", "vcpuCount": 2} and "gpu" not in b
    )
    assert "env" not in b  # no secrets leave the machine unless the plan asks for them


def test_launch_over_budget_never_calls_the_api(tmp_path: Path) -> None:
    c = FakeClient()
    with pytest.raises(BudgetExceededError):
        launch(plan(hourly=100.0), new_ledger(tmp_path), c)  # type: ignore[arg-type]
    assert c.calls == []


def test_launch_records_the_actual_hourly_rate(tmp_path: Path) -> None:
    c, lg = FakeClient(cost=1.5), new_ledger(tmp_path)
    out = launch(plan(), lg, c)  # type: ignore[arg-type]
    assert out["pod_id"] == "pod123" and out["hourly_usd"] == 1.5
    assert [p.hourly_usd for p in lg.running("dlm-test")] == [1.5]


def test_unreadable_price_falls_back_to_the_cap(tmp_path: Path) -> None:
    out = launch(plan(hourly=2.0), new_ledger(tmp_path), FakeClient(cost="n/a"))  # type: ignore[arg-type]
    assert out["hourly_usd"] == 2.0


def test_a_pod_above_the_price_cap_is_terminated(tmp_path: Path) -> None:
    c, lg = FakeClient(cost=9.0), new_ledger(tmp_path)
    with pytest.raises(PodError, match="above"):
        launch(plan(hourly=2.0), lg, c)  # type: ignore[arg-type]
    assert "terminate" in c.calls and not lg.pods


def test_ledger_failure_after_create_terminates_the_pod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c, lg = FakeClient(), new_ledger(tmp_path)

    def boom(*a: object, **k: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(lg, "start", boom)
    with pytest.raises(OSError, match="disk full"):
        launch(plan(), lg, c)  # type: ignore[arg-type]
    assert "terminate" in c.calls


def test_second_launch_while_one_is_running_is_refused(tmp_path: Path) -> None:
    lg = new_ledger(tmp_path)
    launch(plan(), lg, FakeClient())  # type: ignore[arg-type]
    with pytest.raises(PodError, match="already shows a running"):
        launch(plan(), lg, FakeClient())  # type: ignore[arg-type]


def test_stop_waits_for_the_pod_to_disappear_then_records(tmp_path: Path) -> None:
    c, lg = FakeClient(still_listed_for=2), new_ledger(tmp_path)
    launch(plan(), lg, c)  # type: ignore[arg-type]
    sleeps: list[float] = []
    out = stop(plan(), lg, c, sleep=sleeps.append)  # type: ignore[arg-type]
    assert out["running"] == [] and len(sleeps) >= 1


def test_stop_never_records_while_the_pod_is_still_listed(tmp_path: Path) -> None:
    c, lg = FakeClient(still_listed_for=10**6), new_ledger(tmp_path)
    launch(plan(), lg, c)  # type: ignore[arg-type]
    with pytest.raises(PodError, match="still present"):
        stop(plan(), lg, c, verify_secs=0, sleep=lambda s: None)  # type: ignore[arg-type]
    assert lg.running("dlm-test")  # still billing as far as we know


def test_an_unreadable_pod_list_is_not_treated_as_clear(tmp_path: Path) -> None:
    class Broken(FakeClient):
        def list_pods(self) -> list[dict[str, Any]]:
            raise ApiError(500, "down")

    lg = new_ledger(tmp_path)
    launch(plan(), lg, Broken())  # type: ignore[arg-type]
    with pytest.raises(PodError, match="could not list"):
        stop(plan(), lg, Broken(), verify_secs=0, sleep=lambda s: None)  # type: ignore[arg-type]
    assert lg.running("dlm-test")
