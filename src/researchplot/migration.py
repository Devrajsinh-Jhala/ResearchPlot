"""Explicit migration bridges from the stable ResearchPlot 1.x API."""

from __future__ import annotations

import re
import warnings
from pathlib import Path

from .project import ProjectConfig
from .project_api import Project
from .specs import DeliverableSpec, FigureSpec, ProjectSpec
from .target import Target, coerce_format


def _legacy_id(value: str, *, fallback: str) -> str:
    normalized = re.sub(r"[^a-z0-9_-]+", "-", value.casefold()).strip("-_")
    if not normalized:
        normalized = fallback
    if not normalized[0].isalnum():
        normalized = f"figure-{normalized}"
    if not normalized[-1].isalnum():
        normalized = normalized.rstrip("-_")
    return normalized or fallback


def figure_spec_from_target(
    target: Target,
    path: str | Path,
    *,
    figure_id: str | None = None,
    deliverable_id: str = "artifact",
) -> FigureSpec:
    """Represent one legacy target/path pair as an immutable v3 figure spec."""

    artifact = Path(path).expanduser().resolve()
    if not artifact.suffix:
        raise ValueError("A migrated legacy artifact path must have a supported suffix.")
    output_format = coerce_format(artifact.suffix)
    selected_id = _legacy_id(
        figure_id or artifact.stem,
        fallback="figure",
    )
    return FigureSpec(
        id=selected_id,
        role=target.role,
        width=target.width,
        content=target.content,
        deliverables=(
            DeliverableSpec(
                id=_legacy_id(deliverable_id, fallback="artifact"),
                format=output_format,
                path=artifact,
                required=True,
                preferred=True,
            ),
        ),
    )


def project_spec_from_v1(
    config: ProjectConfig,
    *,
    warn: bool = True,
) -> ProjectSpec:
    """Convert a loaded v1 ``ProjectConfig`` without changing the source file."""

    if warn:
        warnings.warn(
            "ProjectConfig is a ResearchPlot 1.x compatibility model; migrate the file "
            "to schema_version = 3 and ProjectSpec.",
            DeprecationWarning,
            stacklevel=2,
        )
    figures: list[FigureSpec] = []
    used_ids: set[str] = set()
    for index, item in enumerate(config.figures, start=1):
        base = _legacy_id(item.path.stem, fallback=f"figure-{index}")
        selected_id = base
        suffix = 2
        while selected_id in used_ids:
            selected_id = f"{base}-{suffix}"
            suffix += 1
        used_ids.add(selected_id)
        if not item.path.suffix:
            raise ValueError(
                f"Cannot migrate {item.path}: a supported artifact suffix is required."
            )
        output_format = coerce_format(item.path.suffix)
        figures.append(
            FigureSpec(
                id=selected_id,
                role=item.role,
                width=item.width,
                content=item.content,
                caption=item.caption,
                alt_text=item.alt_text,
                source_data=(item.source_data,) if item.source_data is not None else (),
                deliverables=(
                    DeliverableSpec(
                        id="artifact",
                        format=output_format,
                        path=item.path,
                        required=True,
                        preferred=True,
                    ),
                ),
            )
        )
    return ProjectSpec(
        profile=config.profile.coordinate,
        policy=config.policy,
        figures=tuple(figures),
        config_path=config.path,
    )


def project_from_v1(
    config: ProjectConfig | str | Path,
    *,
    warn: bool = True,
) -> Project:
    """Load a v1 configuration through the explicit deprecated compatibility bridge."""

    selected = ProjectConfig.load(config) if isinstance(config, (str, Path)) else config
    return Project(project_spec_from_v1(selected, warn=warn))
