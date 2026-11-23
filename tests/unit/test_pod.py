import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest

from dlm.budget import BudgetExceededError, Ledger
from dlm.pod import PodError, PodPlan, launch, runpodctl_pod_names, stop

STATE = Path(".rp-state-test")


class FakeRunner:
    """Scripted replacement for subprocess: keyed on the rp subcommand (last arg)."""

    def __init__(self, replies: dict[str, tuple[int, str, str]] | None = None) -> None:
        self.replies = {"up": (0, "pod123\n", ""), "status": (0, "pod pod123: RUNNING 1.5\n", ""),
                        "down": (0, "", "")}  # fmt: skip
        self.replies.update(replies or {})
        self.calls: list[str] = []

    def __call__(self, cmd: list[str], env: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
        self.calls.append(cmd[-1])
        rc, out, err = self.replies.get(cmd[-1], (0, "", ""))
        return subprocess.CompletedProcess(cmd, rc, out, err)


def plan(hourly: float = 2.0, secs: int = 3600) -> PodPlan:
    return PodPlan(name="dlm-test", image="img", job_cmd="true", max_hourly_usd=hourly,
                   max_lifetime_secs=secs, gpu_types=["NVIDIA A100 80GB PCIe"])  # fmt: skip


def new_ledger(tmp_path: Path, cap: float = 60.0) -> Ledger:
    return Ledger(tmp_path / "budget.json", cap_usd=cap, now=lambda: 1_000_000.0)


def test_plan_worst_case_and_env_never_requests_spot() -> None:
    p = plan(hourly=1.5, secs=7200)
    assert p.planned_usd == pytest.approx(3.0)
    env = p.env(STATE)
    assert env["RP_INTERRUPTIBLE"] == "false" and env["RP_POD_NAME"] == "dlm-test"
    assert env["RP_MAX_LIFETIME_SECS"] == "7200" and "RUNPOD_API_KEY" not in env


def test_launch_over_budget_never_calls_rp(tmp_path: Path) -> None:
    run = FakeRunner()
    with pytest.raises(BudgetExceededError):
        launch(plan(hourly=100.0), new_ledger(tmp_path), STATE, run)
    assert run.calls == []


def test_launch_records_actual_hourly_rate_from_status(tmp_path: Path) -> None:
    run, lg = FakeRunner(), new_ledger(tmp_path)
    out = launch(plan(), lg, STATE, run)
    assert out["pod_id"] == "pod123" and out["hourly_usd"] == 1.5
    assert [p.hourly_usd for p in lg.running("dlm-test")] == [1.5]


def test_launch_falls_back_to_the_price_cap_when_status_is_unreadable(tmp_path: Path) -> None:
    run = FakeRunner({"status": (0, "garbage", "")})
    out = launch(plan(hourly=2.0), new_ledger(tmp_path), STATE, run)
    assert out["hourly_usd"] == 2.0


def test_failed_rp_up_leaves_ledger_untouched(tmp_path: Path) -> None:
    run, lg = FakeRunner({"up": (1, "", "no instances available")}), new_ledger(tmp_path)
    with pytest.raises(PodError, match="no instances"):
        launch(plan(), lg, STATE, run)
    assert not lg.pods


def test_ledger_failure_after_launch_tears_the_pod_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, lg = FakeRunner(), new_ledger(tmp_path)

    def boom(*a: object, **k: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(lg, "start", boom)
    with pytest.raises(OSError, match="disk full"):
        launch(plan(), lg, STATE, run)
    assert "down" in run.calls  # never leave a pod the ledger does not know about


def test_second_launch_with_a_running_pod_is_refused(tmp_path: Path) -> None:
    lg = new_ledger(tmp_path)
    launch(plan(), lg, STATE, FakeRunner())
    with pytest.raises(PodError, match="already shows a running"):
        launch(plan(), lg, STATE, FakeRunner())


def test_stop_records_only_after_the_pod_is_verified_gone(tmp_path: Path) -> None:
    lg = new_ledger(tmp_path)
    launch(plan(), lg, STATE, FakeRunner())
    with pytest.raises(PodError, match="still present"):
        stop(plan(), lg, STATE, FakeRunner(), list_names=lambda: ["dlm-test"])
    assert lg.running("dlm-test")  # still billing as far as we know
    out = stop(plan(), lg, STATE, FakeRunner(), list_names=lambda: ["someone-elses-pod"])
    assert out["running"] == [] and not lg.running()


def test_stop_failure_keeps_ledger_running(tmp_path: Path) -> None:
    lg = new_ledger(tmp_path)
    launch(plan(), lg, STATE, FakeRunner())
    with pytest.raises(PodError, match="rp down failed"):
        stop(plan(), lg, STATE, FakeRunner({"down": (1, "", "api down")}), list_names=lambda: [])
    assert lg.running()


def test_pod_list_is_fail_closed() -> None:
    ok = FakeRunner({"list": (0, '[{"id":"a","name":"x","env":{"K":"secret"}}]', "")})
    assert runpodctl_pod_names(ok) == ["x"]
    with pytest.raises(PodError):
        runpodctl_pod_names(FakeRunner({"list": (1, "", "")}))
    with pytest.raises(PodError):
        runpodctl_pod_names(FakeRunner({"list": (0, "not json", "")}))
