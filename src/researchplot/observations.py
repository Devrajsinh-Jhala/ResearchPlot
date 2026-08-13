"""Typed measurements produced by live-figure and file inspectors."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from .api_types import EvidenceConfidence, EvidencePhase
from .units import normalize_unit


@dataclass(frozen=True, slots=True)
class Observation:
    """One typed fact that can be evaluated by one or more profile rules.

    Producers declare the evidence phase, confidence, unit, and any format or
    phase restrictions.  This metadata prevents a measurement from being reused
    as stronger or broader evidence than the producer actually supports.
    """

    probe: str
    value: object = None
    available: bool = True
    phase: str = "live"
    detail: str | None = None
    unit: str | None = None
    confidence: EvidenceConfidence = EvidenceConfidence.DETERMINISTIC
    producer: str = "researchplot"
    supported_phases: tuple[EvidencePhase, ...] = ()
    supported_formats: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.probe, str) or not self.probe.strip():
            raise ValueError("Observation probe must be a non-empty string.")
        try:
            phase = EvidencePhase(self.phase)
        except ValueError as exc:
            choices = ", ".join(item.value for item in EvidencePhase)
            raise ValueError(
                f"Unknown observation phase {self.phase!r}; choose from: {choices}."
            ) from exc
        object.__setattr__(self, "phase", phase.value)
        try:
            confidence = EvidenceConfidence(self.confidence)
        except ValueError as exc:
            choices = ", ".join(item.value for item in EvidenceConfidence)
            raise ValueError(
                f"Unknown observation confidence {self.confidence!r}; choose from: {choices}."
            ) from exc
        object.__setattr__(self, "confidence", confidence)
        if self.unit is not None:
            object.__setattr__(self, "unit", normalize_unit(self.unit))
        if not isinstance(self.producer, str) or not self.producer.strip():
            raise ValueError("Observation producer must be a non-empty string.")
        phases = tuple(EvidencePhase(item) for item in self.supported_phases)
        object.__setattr__(self, "supported_phases", phases)
        if phases and phase not in phases:
            raise ValueError(
                f"Observation producer {self.producer!r} does not support phase {phase.value!r}."
            )
        formats = tuple(dict.fromkeys(item.casefold() for item in self.supported_formats))
        if any(not item for item in formats):
            raise ValueError("Observation supported formats must be non-empty strings.")
        object.__setattr__(self, "supported_formats", formats)

    def to_dict(self) -> dict[str, object]:
        return {
            "probe": self.probe,
            "value": self.value,
            "available": self.available,
            "phase": self.phase,
            "detail": self.detail,
            "unit": self.unit,
            "confidence": self.confidence.value,
            "producer": self.producer,
            "supported_phases": [item.value for item in self.supported_phases],
            "supported_formats": list(self.supported_formats),
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
