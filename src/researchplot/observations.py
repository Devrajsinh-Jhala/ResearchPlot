"""Typed measurements produced by live-figure and file inspectors."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Observation:
    """One measured fact that can be evaluated by one or more profile rules."""

    probe: str
    value: object = None
    available: bool = True
    phase: str = "live"
    detail: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "probe": self.probe,
            "value": self.value,
            "available": self.available,
            "phase": self.phase,
            "detail": self.detail,
        }


class ObservationSet:
    """Indexed, immutable collection of observations."""

    def __init__(self, observations: Iterable[Observation]) -> None:
        values = tuple(observations)
        duplicates = {
            probe for probe, count in Counter(item.probe for item in values).items() if count > 1
        }
        if duplicates:
            raise ValueError(f"Duplicate observations: {', '.join(sorted(duplicates))}.")
        self._values = values
        self._by_probe = {item.probe: item for item in values}

    def get(self, probe: str) -> Observation | None:
        return self._by_probe.get(probe)

    def __iter__(self) -> Iterator[Observation]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def to_dict(self) -> dict[str, object]:
        return {item.probe: item.to_dict() for item in self._values}
