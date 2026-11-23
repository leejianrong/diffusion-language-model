"""Spend ledger and launch gate for rented GPUs (ADR-0005).

The cap is enforced by code, not memory. Before any pod launches, the gate adds up what has been
spent, what the pods still running could yet spend inside their lifetime ceiling, and what the new
pod could spend at its own ceiling, and refuses if that exceeds the cap.
"""

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_CAP_USD = 60.0


class BudgetExceededError(RuntimeError):
    pass


@dataclass
class PodRecord:
    pod_id: str
    label: str
    start: float  # unix seconds
    hourly_usd: float
    planned_hours: float  # lifetime ceiling the pod was launched with
    stop: float | None = None

    def cost(self, now: float) -> float:
        end = self.stop if self.stop is not None else now
        return max(0.0, end - self.start) / 3600 * self.hourly_usd

    def remaining_exposure(self, now: float) -> float:
        """Most a still-running pod can yet cost before its own lifetime ceiling ends it."""
        if self.stop is not None:
            return 0.0
        left_h = max(0.0, self.planned_hours - (now - self.start) / 3600)
        return left_h * self.hourly_usd


class Ledger:
    def __init__(
        self, path: Path, cap_usd: float = DEFAULT_CAP_USD, now: Callable[[], float] = time.time
    ) -> None:
        self.path, self.now = path, now
        self.cap_usd = cap_usd
        self.pods: list[PodRecord] = []
        if path.exists():
            data = json.loads(path.read_text())
            self.cap_usd = float(data["cap_usd"])
            self.pods = [PodRecord(**p) for p in data["pods"]]

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"cap_usd": self.cap_usd, "pods": [asdict(p) for p in self.pods]}, indent=2)
        )
        tmp.replace(self.path)

    def running(self, label: str | None = None) -> list[PodRecord]:
        return [p for p in self.pods if p.stop is None and (label is None or p.label == label)]

    def spent(self) -> float:
        t = self.now()
        return sum(p.cost(t) for p in self.pods)

    def committed(self) -> float:
        """Spent so far plus the worst case still ahead for pods that are running."""
        t = self.now()
        return self.spent() + sum(p.remaining_exposure(t) for p in self.pods)

    def check(self, planned_usd: float) -> None:
        after = self.committed() + planned_usd
        if after > self.cap_usd + 1e-9:
            raise BudgetExceededError(
                f"refusing launch: committed ${self.committed():.2f} + planned ${planned_usd:.2f} "
                f"= ${after:.2f} exceeds the ${self.cap_usd:.2f} cap"
            )

    def start(self, pod_id: str, label: str, hourly_usd: float, planned_hours: float) -> None:
        self.pods.append(PodRecord(pod_id, label, self.now(), hourly_usd, planned_hours))
        self.save()

    def stop(self, pod_id: str) -> None:
        """Idempotent: stopping an already-stopped or unknown pod is a no-op."""
        for p in self.pods:
            if p.pod_id == pod_id and p.stop is None:
                p.stop = self.now()
        self.save()

    def summary(self) -> dict[str, Any]:
        return {
            "cap_usd": self.cap_usd,
            "spent_usd": round(self.spent(), 4),
            "committed_usd": round(self.committed(), 4),
            "running": [p.pod_id for p in self.running()],
        }
