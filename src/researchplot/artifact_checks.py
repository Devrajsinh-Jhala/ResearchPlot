"""Profile evaluation for normalized exported-artifact observations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from .compliance import Finding, Outcome, Report, RuleEngine
from .inspectors import ArtifactInspection, inspect_artifact
from .models import RuleLevel
from .observations import Observation, ObservationSet
from .target import Target


def _observation_set(inspection: ArtifactInspection) -> ObservationSet:
    observations = [
        Observation(item.key, item.value, phase="file") for item in inspection.observations
    ]
    metadata = inspection.metadata
    dpi_values = [
        float(value)
        for key in ("raster.dpi_x", "raster.dpi_y")
        if isinstance((value := metadata.get(key)), (int, float)) and not isinstance(value, bool)
    ]
    observations.append(
        Observation(
            "artifact.dpi",
            min(dpi_values) if dpi_values else None,
            available=bool(dpi_values),
            phase="file",
            detail="Raster DPI metadata is unavailable for this artifact."
            if not dpi_values
            else None,
        )
    )
    font_names = metadata.get("pdf.font_names")
    observations.append(
        Observation(
            "font.families",
            tuple(str(name).lstrip("/").split("+")[-1] for name in font_names)
            if isinstance(font_names, tuple)
            else None,
            available=isinstance(font_names, tuple) and bool(font_names),
            phase="file",
            detail="Computed font families are unavailable for this artifact format.",
        )
    )
    return ObservationSet(observations)


def audit_target(
    path: str | Path,
    *,
    target: Target,
    attestations: Mapping[str, str] | None = None,
) -> Report:
    """Inspect one artifact and evaluate all applicable file rules."""

    inspection = inspect_artifact(path)
    context = target.context(inspection.format)
    report = RuleEngine().evaluate(
        target.profile,
        _observation_set(inspection),
        context,
        phase="file",
        attestations=attestations,
    )
    artifact_name = inspection.path.name
    report = replace(
        report,
        findings=tuple(replace(finding, artifact=artifact_name) for finding in report.findings),
    )
    extras: list[Finding] = []
    for index, warning in enumerate(inspection.warnings, start=1):
        extras.append(
            Finding(
                f"inspector.warning.{index}",
                Outcome.SKIP,
                RuleLevel.INFERRED,
                "file",
                warning,
                verification="automated",
                artifact=artifact_name,
            )
        )
    for index, caveat in enumerate(target.profile.caveats, start=1):
        extras.append(
            Finding(
                f"profile.caveat.{index}",
                Outcome.SKIP,
                RuleLevel.INFERRED,
                "file",
                caveat,
                verification="manual",
                artifact=artifact_name,
            )
        )
    return replace(report, findings=report.findings + tuple(extras))
