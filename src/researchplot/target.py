"""Resolved submission targets: the primary ResearchPlot 1.0 workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from matplotlib.figure import Figure

from .compliance import Policy, Report, RuleEngine, TargetContext
from .figure_inspector import inspect_figure
from .models import ContentKind, FigureRole, OutputFormat, VenueProfile
from .registry import resolve_profile
from .style import StyleContext

if TYPE_CHECKING:
    from .transactional_export import ExportResult


def _enum_value(value: str) -> str:
    return value.strip().casefold().replace("-", "_").replace(" ", "_")


def coerce_content(value: ContentKind | str) -> ContentKind:
    if isinstance(value, ContentKind):
        return value
    aliases = {
        "photo": "photograph",
        "vector": "line_art",
        "data": "data_visualization",
    }
    normalized = aliases.get(_enum_value(value), _enum_value(value))
    try:
        return ContentKind(normalized)
    except ValueError as exc:
        choices = ", ".join(item.value.replace("_", "-") for item in ContentKind)
        raise ValueError(f"Unknown content kind {value!r}. Choose from: {choices}.") from exc


def coerce_role(value: FigureRole | str) -> FigureRole:
    if isinstance(value, FigureRole):
        return value
    try:
        return FigureRole(_enum_value(value))
    except ValueError as exc:
        choices = ", ".join(item.value.replace("_", "-") for item in FigureRole)
        raise ValueError(f"Unknown figure role {value!r}. Choose from: {choices}.") from exc


def coerce_format(value: OutputFormat | str) -> OutputFormat:
    if isinstance(value, OutputFormat):
        return value
    normalized = _enum_value(value).lstrip(".")
    normalized = {"jpg": "jpeg", "tif": "tiff"}.get(normalized, normalized)
    try:
        return OutputFormat(normalized)
    except ValueError as exc:
        choices = ", ".join(item.value for item in OutputFormat)
        raise ValueError(f"Unknown output format {value!r}. Choose from: {choices}.") from exc


@dataclass(frozen=True, slots=True)
class Target:
    """A resolved profile plus all metadata needed to select conditional rules."""

    profile: VenueProfile
    role: FigureRole
    width: str | None
    content: ContentKind

    def __post_init__(self) -> None:
        if self.width is not None:
            self.profile.width_mm(self.width)

    @property
    def coordinate(self) -> str:
        return str(getattr(self.profile, "coordinate", self.profile.id))

    def context(self, output_format: OutputFormat | str | None = None) -> TargetContext:
        selected = coerce_format(output_format).value if output_format is not None else None
        return TargetContext(
            role=self.role.value,
            width=self.width,
            content=self.content.value,
            output_format=selected,
        )

    def style(
        self,
        *,
        latex: bool = False,
        overrides: dict[str, Any] | None = None,
    ) -> StyleContext:
        """Create a reversible Matplotlib style context for this target."""

        if self.width is None:
            raise ValueError(
                f"Profile {self.coordinate!r} does not define a physical figure width; "
                "its requirements can be used for bundle checks, but not for styling."
            )

        return StyleContext(
            self.profile,
            width=self.width,
            latex=latex,
            overrides=overrides,
        )

    def validate(
        self,
        fig: Figure,
        *,
        attestations: dict[str, str] | None = None,
    ) -> Report:
        """Evaluate all applicable live-figure rules."""

        return RuleEngine().evaluate(
            self.profile,
            inspect_figure(fig),
            self.context(),
            phase="live",
            attestations=attestations,
        )

    def audit(
        self,
        path: str | Path,
        *,
        attestations: dict[str, str] | None = None,
    ) -> Report:
        """Inspect an exported artifact and evaluate file-phase rules."""

        from .artifact_checks import audit_target

        return audit_target(path, target=self, attestations=attestations)

    def export(
        self,
        fig: Figure,
        target_path: str | Path,
        *,
        formats: tuple[OutputFormat | str, ...] | list[OutputFormat | str] | None = None,
        policy: Policy | str = Policy.COMPLETE,
        dpi: int | None = None,
        overwrite: bool = False,
        attestations: dict[str, str] | None = None,
        metadata: dict[str, object] | None = None,
        **savefig_kwargs: Any,
    ) -> ExportResult:
        """Transactionally export and post-audit this target."""

        from .transactional_export import export_target

        return export_target(
            fig,
            target_path,
            target=self,
            formats=formats,
            policy=policy,
            dpi=dpi,
            overwrite=overwrite,
            attestations=attestations,
            metadata=metadata,
            savefig_kwargs=savefig_kwargs,
        )


def target(
    profile: str | VenueProfile,
    *,
    role: FigureRole | str = FigureRole.MAIN,
    width: str | None = None,
    content: ContentKind | str = ContentKind.DATA_VISUALIZATION,
) -> Target:
    """Resolve an immutable profile revision into a submission target."""

    resolved = resolve_profile(profile)
    return Target(
        resolved,
        coerce_role(role),
        width if width is not None else resolved.default_width,
        coerce_content(content),
    )
