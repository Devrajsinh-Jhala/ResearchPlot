"""Staged multi-figure submission bundle construction."""

from __future__ import annotations

import json
import shutil
import tempfile
import unicodedata
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure

from .compliance import CompliancePolicyError, Policy, Report, RuleEngine
from .inspectors import inspect_artifact
from .manifest import ArtifactRecord, _package_version, profile_digest
from .models import ContentKind, FigureRole, OutputFormat, VenueProfile
from .observations import Observation, ObservationSet
from .registry import resolve_profile
from .target import Target, coerce_content, coerce_role
from .target import target as make_target
from .transactional_export import ExportResult, _output_paths, _permitted_formats

_WINDOWS_RESERVED_NAMES = {
    "aux",
    "con",
    "nul",
    "prn",
    *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
}
_WINDOWS_FORBIDDEN_CHARACTERS = frozenset('<>:"/\\|?*')


def _portable_component_key(component: str) -> str:
    normalized = unicodedata.normalize("NFC", component)
    if not normalized or normalized in {".", ".."}:
        raise ValueError("Submission paths must not contain empty or dot components.")
    if normalized.endswith((" ", ".")):
        raise ValueError(f"Submission path component {component!r} has a trailing dot or space.")
    if any(
        ord(character) < 32 or character in _WINDOWS_FORBIDDEN_CHARACTERS
        for character in normalized
    ):
        raise ValueError(f"Submission path component {component!r} is not portable.")
    device_name = normalized.split(".", 1)[0].casefold()
    if device_name in _WINDOWS_RESERVED_NAMES:
        raise ValueError(f"Submission path component {component!r} is a reserved device name.")
    return normalized.casefold()


def _portable_path_key(path: Path) -> str:
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError("Submission paths must be relative and remain inside the bundle.")
    return "/".join(_portable_component_key(part) for part in path.parts)


def _copy_exclusive(source: Path, destination: Path) -> None:
    """Copy a file without ever replacing a previously written bundle artifact."""

    created = False
    try:
        writer = destination.open("xb")
        created = True
        with source.open("rb") as reader, writer:
            shutil.copyfileobj(reader, writer)
        shutil.copystat(source, destination)
    except BaseException:
        if created:
            destination.unlink(missing_ok=True)
        raise


def _path_lexists(path: Path) -> bool:
    """Return whether a path entry exists without following symbolic links."""

    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


@dataclass(frozen=True, slots=True)
class SubmissionItemResult:
    name: str
    paths: tuple[Path, ...]
    report: Report
    metadata: dict[str, object]

    def to_dict(self, *, relative_to: Path) -> dict[str, object]:
        return {
            "name": self.name,
            "paths": [path.relative_to(relative_to).as_posix() for path in self.paths],
            "metadata": self.metadata,
            "report": self.report.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class BundleResult:
    path: Path
    manifest_path: Path
    items: tuple[SubmissionItemResult, ...]

    @property
    def passed(self) -> bool:
        return all(item.report.passed for item in self.items)

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path.as_posix(),
            "manifest_path": self.manifest_path.as_posix(),
            "passed": self.passed,
            "items": [item.to_dict(relative_to=self.path) for item in self.items],
        }


@dataclass(slots=True)
class _SubmissionEntry:
    name: str
    asset: Figure | Path
    role: FigureRole
    width: str | None
    content: ContentKind
    formats: tuple[OutputFormat | str, ...] | None
    metadata: dict[str, object]
    attestations: dict[str, str]
    source_data: Path | None


@dataclass(slots=True)
class Submission:
    """Collect figures and build one audited directory through staging."""

    profile: VenueProfile | str
    output_dir: Path | str = Path("submission")
    policy: Policy | str = Policy.COMPLETE
    _entries: list[_SubmissionEntry] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.profile = resolve_profile(self.profile)
        self.output_dir = Path(self.output_dir)
        self.policy = Policy(self.policy)

    def add(
        self,
        name: str,
        asset: Figure | str | Path,
        *,
        role: FigureRole | str = FigureRole.MAIN,
        width: str | None = None,
        content: ContentKind | str = ContentKind.DATA_VISUALIZATION,
        formats: tuple[OutputFormat | str, ...] | list[OutputFormat | str] | None = None,
        alt_text: str | None = None,
        caption: str | None = None,
        source_data: str | Path | None = None,
        attestations: dict[str, str] | None = None,
    ) -> Submission:
        """Add one live figure or existing artifact to the pending bundle."""

        if alt_text is not None and not isinstance(alt_text, str):
            raise TypeError("alt_text must be a string or null.")
        if caption is not None and not isinstance(caption, str):
            raise TypeError("caption must be a string or null.")
        path_name = Path(name)
        if path_name.is_absolute() or ".." in path_name.parts or len(path_name.parts) != 1:
            raise ValueError("Submission item names must be a single relative filename or stem.")
        name_key = _portable_path_key(path_name)
        if any(_portable_path_key(Path(entry.name)) == name_key for entry in self._entries):
            raise ValueError(f"Submission item {name!r} already exists.")
        resolved_profile = self.profile
        assert isinstance(resolved_profile, VenueProfile)
        selected_width = width if width is not None else resolved_profile.default_width
        if selected_width is not None:
            resolved_profile.width_mm(selected_width)
        stored_asset: Figure | Path = (
            Path(asset).resolve() if isinstance(asset, (str, Path)) else asset
        )
        source_path = Path(source_data).resolve() if source_data is not None else None
        source_name = (
            f"source-data/{path_name.stem}{source_path.suffix.casefold()}"
            if source_path is not None
            else None
        )
        metadata: dict[str, object] = {
            "alt_text": alt_text,
            "caption": caption,
            "source_data": source_name,
        }
        self._entries.append(
            _SubmissionEntry(
                name,
                stored_asset,
                coerce_role(role),
                selected_width,
                coerce_content(content),
                tuple(formats) if formats is not None else None,
                metadata,
                dict(attestations or {}),
                source_path,
            )
        )
        return self

    def _bundle_report(self, entry: _SubmissionEntry, target: Target) -> Report:
        metadata = entry.metadata
        alt_text = metadata.get("alt_text")
        caption = metadata.get("caption")
        normalized_alt = str(alt_text).strip().casefold() if alt_text else ""
        normalized_caption = str(caption).strip().casefold() if caption else ""
        observations = ObservationSet(
            (
                Observation("metadata.alt_text.present", bool(normalized_alt), phase="bundle"),
                Observation("metadata.caption.present", bool(normalized_caption), phase="bundle"),
                Observation(
                    "metadata.source_data.present",
                    bool(metadata.get("source_data")),
                    phase="bundle",
                ),
                Observation(
                    "metadata.alt_text.distinct_from_caption",
                    bool(normalized_alt) and normalized_alt != normalized_caption,
                    phase="bundle",
                ),
            )
        )
        return RuleEngine().evaluate(
            target.profile,
            observations,
            target.context(),
            phase="bundle",
            attestations=entry.attestations,
        )

    def build(self) -> BundleResult:
        """Build the complete bundle and clean staging after handled failures."""

        if not self._entries:
            raise ValueError("A submission must contain at least one figure.")
        output = Path(self.output_dir).resolve()
        if _path_lexists(output):
            raise FileExistsError(f"Submission output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        resolved_profile = self.profile
        selected_policy = self.policy
        assert isinstance(resolved_profile, VenueProfile)
        assert isinstance(selected_policy, Policy)

        claimed_paths: dict[str, str] = {}

        def claim(path: Path, owner: str) -> None:
            key = _portable_path_key(path)
            previous = claimed_paths.get(key)
            if previous is not None:
                raise ValueError(
                    f"Bundle path {path.as_posix()!r} for {owner} conflicts with {previous}."
                )
            claimed_paths[key] = owner

        claim(Path("researchplot-manifest.json"), "the bundle manifest")
        planned_entries: list[tuple[_SubmissionEntry, Target, tuple[Path, ...]]] = []
        for entry in self._entries:
            target = make_target(
                resolved_profile,
                role=entry.role,
                width=entry.width,
                content=entry.content,
            )
            desired = Path(entry.name)
            if isinstance(entry.asset, Figure):
                planned_paths = _output_paths(desired, entry.formats, _permitted_formats(target))
                for path in planned_paths:
                    claim(path, f"figure {entry.name!r}")
                claim(
                    desired.with_suffix(".researchplot.json"),
                    f"temporary manifest for figure {entry.name!r}",
                )
            else:
                planned_paths = (
                    desired if desired.suffix else desired.with_suffix(entry.asset.suffix),
                )
                claim(planned_paths[0], f"artifact {entry.name!r}")
            if entry.source_data is not None:
                claim(Path(str(entry.metadata["source_data"])), f"source data for {entry.name!r}")
            planned_entries.append((entry, target, planned_paths))

        with tempfile.TemporaryDirectory(prefix=".researchplot-bundle-", dir=output.parent) as temp:
            staging = Path(temp) / output.name
            staging.mkdir()
            staged_results: list[SubmissionItemResult] = []
            artifact_formats: dict[Path, str] = {}
            for entry, target, planned_paths in planned_entries:
                desired = staging / entry.name
                if isinstance(entry.asset, Figure):
                    exported: ExportResult = target.export(
                        entry.asset,
                        desired,
                        formats=entry.formats,
                        policy=selected_policy,
                        attestations=entry.attestations,
                        metadata=entry.metadata,
                    )
                    paths = exported.paths
                    artifact_report = exported.report
                    exported.manifest_path.unlink()
                    artifact_formats.update((path, inspect_artifact(path).format) for path in paths)
                else:
                    if not entry.asset.is_file():
                        raise FileNotFoundError(f"Submission artifact not found: {entry.asset}")
                    destination = staging / planned_paths[0]
                    _copy_exclusive(entry.asset, destination)
                    paths = (destination,)
                    artifact_formats[destination] = inspect_artifact(destination).format
                    artifact_report = target.audit(destination, attestations=entry.attestations)
                bundle_report = self._bundle_report(entry, target)
                report = replace(
                    artifact_report,
                    findings=artifact_report.findings + bundle_report.findings,
                )
                if report.blocks(selected_policy):
                    raise CompliancePolicyError(report, selected_policy)
                if entry.source_data is not None:
                    if not entry.source_data.is_file():
                        raise FileNotFoundError(
                            f"Submission source data not found: {entry.source_data}"
                        )
                    source_destination = staging / str(entry.metadata["source_data"])
                    source_destination.parent.mkdir(exist_ok=True)
                    if source_destination.exists():
                        raise ValueError(
                            f"Source-data destination is duplicated: {source_destination.name}"
                        )
                    _copy_exclusive(entry.source_data, source_destination)
                staged_results.append(
                    SubmissionItemResult(entry.name, paths, report, dict(entry.metadata))
                )

            artifact_records = tuple(
                ArtifactRecord.from_path(
                    path,
                    relative_to=staging,
                    artifact_format=artifact_formats.get(path),
                )
                for path in sorted(staging.rglob("*"))
                if path.is_file()
            )
            manifest_data: dict[str, Any] = {
                "schema_version": 1,
                "researchplot_version": _package_version(),
                "profile": str(getattr(resolved_profile, "coordinate", resolved_profile.id)),
                "profile_digest": profile_digest(resolved_profile),
                "sources": [source.to_dict() for source in resolved_profile.sources],
                "caveats": list(resolved_profile.caveats),
                "artifacts": [record.to_dict() for record in artifact_records],
                "figures": [result.to_dict(relative_to=staging) for result in staged_results],
            }
            staged_manifest = staging / "researchplot-manifest.json"
            staged_manifest.write_text(
                json.dumps(manifest_data, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            if _path_lexists(output):
                raise FileExistsError(
                    f"Submission output was created while the bundle was staged: {output}"
                )
            staging.replace(output)

        final_items = tuple(
            SubmissionItemResult(
                item.name,
                tuple(output / path.relative_to(staging) for path in item.paths),
                item.report,
                item.metadata,
            )
            for item in staged_results
        )
        return BundleResult(output, output / "researchplot-manifest.json", final_items)
