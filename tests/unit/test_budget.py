from pathlib import Path

import pytest

from dlm.budget import BudgetExceededError, Ledger

H = 3600.0


class Clock:
    def __init__(self) -> None:
        self.t = 1_000_000.0

    def __call__(self) -> float:
        return self.t


def ledger(tmp_path: Path, clock: Clock, cap: float = 60.0) -> Ledger:
    return Ledger(tmp_path / "budget.json", cap_usd=cap, now=clock)


def test_running_pod_with_no_recorded_stop_costs_up_to_now(tmp_path: Path) -> None:
    c = Clock()
    lg = ledger(tmp_path, c)
    lg.start("p1", "train", hourly_usd=2.0, planned_hours=10)
    c.t += 3 * H
    assert lg.spent() == pytest.approx(6.0)
    lg.stop("p1")
    c.t += 5 * H
    assert lg.spent() == pytest.approx(6.0)  # stopped: the clock no longer runs


def test_launch_refused_when_spent_plus_planned_exceeds_cap_accepted_when_equal(
    tmp_path: Path,
) -> None:
    c = Clock()
    lg = ledger(tmp_path, c)
    lg.start("old", "x", hourly_usd=1.0, planned_hours=100)
    c.t += 59 * H  # $59 spent
    lg.stop("old")
    with pytest.raises(BudgetExceededError, match="exceeds"):
        lg.check(planned_usd=2.0)
    lg.check(planned_usd=1.0)  # exactly at the cap is allowed


def test_still_running_pods_count_their_remaining_worst_case(tmp_path: Path) -> None:
    c = Clock()
    lg = ledger(tmp_path, c, cap=12.0)
    lg.start("p", "x", hourly_usd=1.0, planned_hours=10)
    c.t += 2 * H  # spent 2, up to 8 more to come
    assert lg.committed() == pytest.approx(10.0)
    with pytest.raises(BudgetExceededError):
        lg.check(planned_usd=3.0)
    lg.check(planned_usd=2.0)


def test_stop_is_idempotent_and_ignores_unknown_pods(tmp_path: Path) -> None:
    c = Clock()
    lg = ledger(tmp_path, c)
    lg.start("p", "x", 1.0, 1)
    c.t += H
    lg.stop("p")
    first = lg.spent()
    c.t += H
    lg.stop("p")
    lg.stop("nope")
    assert lg.spent() == first and not lg.running()


def test_ledger_persists_across_instances(tmp_path: Path) -> None:
    c = Clock()
    lg = ledger(tmp_path, c, cap=42.0)
    lg.start("p", "x", 1.5, 2)
    again = Ledger(tmp_path / "budget.json", now=c)
    assert again.cap_usd == 42.0 and [p.pod_id for p in again.running()] == ["p"]
