"""Stable v2 public enums that are independent of individual inspectors."""

from __future__ import annotations

from enum import StrEnum


class CheckStatus(StrEnum):
    """Normalized status used by v2 report consumers."""

    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    INFO = "info"
    SKIPPED = "skipped"
    NOT_APPLICABLE = "not_applicable"


class EvidencePhase(StrEnum):
    """Stage from which a compliance observation originated."""

    LIVE = "live"
    FILE = "file"
    BUNDLE = "bundle"
    MANUSCRIPT = "manuscript"


class EvidenceConfidence(StrEnum):
    """How strongly an observation establishes the stated fact."""

    DETERMINISTIC = "deterministic"
    HEURISTIC = "heuristic"
    MANUAL = "manual"


class ArtworkType(StrEnum):
    """High-level artifact representation selected for export or audit."""

    VECTOR = "vector"
    LINE_ART = "line_art"
    HALFTONE = "halftone"
    COMBINATION = "combination"
    PHOTOGRAPH = "photograph"


__all__ = ["ArtworkType", "CheckStatus", "EvidenceConfidence", "EvidencePhase"]
