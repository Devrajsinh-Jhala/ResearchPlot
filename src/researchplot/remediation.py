"""Deterministic, non-mutating remediation guidance for inspected artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .inspectors import ArtifactInspection


class RemediationKind(StrEnum):
    """Stable category for a corrective action."""

    ACTIVE_CONTENT = "active-content"
    EXTERNAL_REFERENCE = "external-reference"
    FONT_EMBEDDING = "font-embedding"
    TYPE3_FONT = "type3-font"
    TRANSPARENCY = "transparency"
    ORIENTATION = "orientation"
    MULTI_FRAME = "multi-frame"
    PHYSICAL_SIZE = "physical-size"
    FILE_NAMING = "file-naming"


@dataclass(frozen=True, slots=True)
class Remediation:
    """One traceable human action; ResearchPlot never applies it implicitly."""

    id: str
    kind: RemediationKind
    severity: str
    summary: str
    guidance: str
    evidence_keys: tuple[str, ...]
    automatic: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "severity": self.severity,
            "summary": self.summary,
            "guidance": self.guidance,
            "evidence_keys": list(self.evidence_keys),
            "automatic": self.automatic,
        }


@dataclass(frozen=True, slots=True)
class RemediationPlan:
    """Ordered remediation guidance derived only from measured facts."""

    artifact_format: str
    remediations: tuple[Remediation, ...]

    @property
    def empty(self) -> bool:
        return not self.remediations

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_format": self.artifact_format,
            "remediations": [item.to_dict() for item in self.remediations],
        }

    def to_markdown(self) -> str:
        if not self.remediations:
            return "No artifact-level remediations were identified."
        lines = ["## Artifact remediation plan", ""]
        for index, item in enumerate(self.remediations, start=1):
            lines.extend(
                (
                    f"{index}. **{item.summary}** (`{item.severity}`)",
                    "",
                    f"   {item.guidance}",
                    "",
                )
            )
        return "\n".join(lines).rstrip()


_SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def _count(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return default


def _sequence(value: object) -> tuple[object, ...]:
    return value if isinstance(value, tuple) else ()


def plan_remediation(inspection: ArtifactInspection) -> RemediationPlan:
    """Classify passive inspector observations into deterministic human guidance."""

    items: list[Remediation] = []
    get = inspection.get
    active_key = f"{inspection.format}.active_content"
    active_types_key = f"{inspection.format}.active_content_types"
    if bool(get(active_key, False)):
        active_types = (
            ", ".join(str(item) for item in _sequence(get(active_types_key, ()))) or "unknown"
        )
        items.append(
            Remediation(
                "remove-active-content",
                RemediationKind.ACTIVE_CONTENT,
                "critical",
                "Remove active or embedded content",
                f"Re-export as a static artifact and confirm that these indicators are absent: "
                f"{active_types}. Do not merely rename the file.",
                (active_key, active_types_key),
            )
        )

    external_key = (
        "svg.external_link_count" if inspection.format == "svg" else "pdf.external_uri_count"
    )
    if _count(get(external_key, 0)) > 0:
        items.append(
            Remediation(
                "embed-external-resources",
                RemediationKind.EXTERNAL_REFERENCE,
                "high",
                "Remove external resource dependencies",
                "Embed approved static resources or remove links so the artifact remains complete "
                "and reviewable offline.",
                (external_key,),
            )
        )

    if _count(get("pdf.unembedded_font_count", 0)) > 0:
        items.append(
            Remediation(
                "embed-pdf-fonts",
                RemediationKind.FONT_EMBEDDING,
                "high",
                "Embed all PDF fonts",
                "Re-export from the authoring application with font embedding enabled, then audit "
                "the new PDF. Converting text to outlines can reduce editability and accessibility.",
                ("pdf.unembedded_font_count", "pdf.font_details"),
            )
        )
    if _count(get("pdf.type3_font_count", 0)) > 0:
        items.append(
            Remediation(
                "replace-type3-fonts",
                RemediationKind.TYPE3_FONT,
                "medium",
                "Replace Type 3 fonts",
                "Use an embeddable OpenType or TrueType font and regenerate the PDF.",
                ("pdf.type3_font_count", "pdf.font_details"),
            )
        )
    if bool(get("pdf.has_transparency", False)):
        items.append(
            Remediation(
                "review-pdf-transparency",
                RemediationKind.TRANSPARENCY,
                "medium",
                "Review PDF transparency",
                "If the selected venue prohibits transparency, flatten it using the source "
                "application and visually compare the result before submission.",
                ("pdf.has_transparency", "pdf.transparency_reasons"),
            )
        )
    if bool(get("raster.requires_orientation_transform", False)):
        items.append(
            Remediation(
                "normalize-raster-orientation",
                RemediationKind.ORIENTATION,
                "medium",
                "Normalize raster orientation",
                "Apply the EXIF orientation to the pixels and save with orientation 1 so all "
                "submission systems display the same geometry.",
                ("raster.exif_orientation", "raster.requires_orientation_transform"),
            )
        )
    if _count(get("raster.frame_count", 1), 1) > 1:
        items.append(
            Remediation(
                "export-static-raster-frame",
                RemediationKind.MULTI_FRAME,
                "high",
                "Export one static raster frame",
                "Select the intended frame and export it as a single-frame artifact; retain the "
                "animation separately only when the venue explicitly accepts it.",
                ("raster.frame_count", "raster.frame_sizes"),
            )
        )
    if get("artifact.width_mm") is None or get("artifact.height_mm") is None:
        items.append(
            Remediation(
                "declare-physical-size",
                RemediationKind.PHYSICAL_SIZE,
                "medium",
                "Declare a verifiable physical size",
                "Export one static figure with absolute dimensions (or valid DPI for raster "
                "artifacts) so physical size can be audited.",
                ("artifact.width_mm", "artifact.height_mm"),
            )
        )
    if get("artifact.extension_matches_content") is False:
        items.append(
            Remediation(
                "align-extension-with-content",
                RemediationKind.FILE_NAMING,
                "low",
                "Align the filename extension with the encoded format",
                "Rename the artifact to the extension matching its measured content signature.",
                ("artifact.extension", "artifact.extension_matches_content", "artifact.format"),
            )
        )
    ordered = tuple(
        sorted(
            items,
            key=lambda item: (_SEVERITY_ORDER[item.severity], item.kind.value, item.id),
        )
    )
    return RemediationPlan(inspection.format, ordered)


__all__ = ["Remediation", "RemediationKind", "RemediationPlan", "plan_remediation"]
