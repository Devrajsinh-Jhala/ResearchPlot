"""Transactional Matplotlib export with mandatory post-export inspection."""

from __future__ import annotations

import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib as mpl
from matplotlib.figure import Figure
from PIL import Image

from .compliance import CompliancePolicyError, Policy, Report
from .manifest import ArtifactRecord, ExportManifest, profile_digest
from .models import ConstraintOperator, OutputFormat, VenueRule
from .target import Target, coerce_format


@dataclass(frozen=True, slots=True)
class ExportResult:
    """Audited paths and evidence produced by a successful export."""

    paths: tuple[Path, ...]
    report: Report
    manifest: ExportManifest
    manifest_path: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "paths": [path.as_posix() for path in self.paths],
            "manifest_path": self.manifest_path.as_posix(),
            "report": self.report.to_dict(),
            "manifest": self.manifest.to_dict(),
        }


def _rule_values(rule: VenueRule) -> tuple[str, ...]:
    value = rule.value
    if isinstance(value, tuple):
        return tuple(str(item).casefold() for item in value)
    return ()


def _permitted_formats(target: Target) -> tuple[OutputFormat, ...]:
    selected: list[OutputFormat] = []
    for rule in target.profile.rules:
        if not rule.id.startswith("export.formats."):
            continue
        applicability = getattr(rule, "applies_to", None)
        if applicability is not None and not applicability.matches(
            role=target.role, content_kind=target.content, width=target.width
        ):
            continue
        for item in _rule_values(rule):
            try:
                output = coerce_format(item)
            except ValueError:
                continue
            if output not in selected:
                selected.append(output)
    return tuple(selected)


def _minimum_dpi(target: Target, output_format: OutputFormat) -> int | None:
    candidates: list[int] = []
    for rule in target.profile.rules:
        if (
            not rule.id.startswith("export.min_dpi.")
            and getattr(rule, "probe", "") != "artifact.dpi"
        ):
            continue
        applicability = getattr(rule, "applies_to", None)
        if applicability is None or applicability.matches(
            role=target.role,
            content_kind=target.content,
            output_format=output_format,
            width=target.width,
        ):
            if isinstance(rule.value, (int, float)) and not isinstance(rule.value, bool):
                candidates.append(int(rule.value))
            elif (
                rule.constraint.operator is ConstraintOperator.BETWEEN
                and isinstance(rule.value, tuple)
                and len(rule.value) == 2
                and isinstance(rule.value[0], (int, float))
            ):
                candidates.append(int(rule.value[0]))
    return max(candidates) if candidates else None


def _required_compression(target: Target, output_format: OutputFormat) -> str | None:
    for rule in target.profile.rules:
        if rule.probe != "raster.compression":
            continue
        if not rule.applies_to.matches(
            role=target.role,
            content_kind=target.content,
            output_format=output_format,
            width=target.width,
        ):
            continue
        if rule.constraint.operator is ConstraintOperator.EQ and isinstance(rule.value, str):
            return rule.value
    return None


def _allowed_raster_modes(target: Target, output_format: OutputFormat) -> tuple[str, ...]:
    for rule in target.profile.rules:
        if rule.probe != "raster.mode":
            continue
        if not rule.applies_to.matches(
            role=target.role,
            content_kind=target.content,
            output_format=output_format,
            width=target.width,
        ):
            continue
        if rule.constraint.operator is ConstraintOperator.IN and isinstance(rule.value, tuple):
            return tuple(str(item) for item in rule.value)
    return ()


def _normalize_raster(
    path: Path,
    *,
    target: Target,
    output_format: OutputFormat,
    compression: str | None,
) -> None:
    allowed_modes = _allowed_raster_modes(target, output_format)
    if not allowed_modes:
        return
    with Image.open(path) as image:
        if image.mode in allowed_modes:
            return
        if "RGB" not in allowed_modes or image.mode not in {"RGBA", "P", "LA"}:
            return
        image.load()
        converted = image.convert("RGB")
        dpi_value = image.info.get("dpi")
        icc_profile = image.info.get("icc_profile")
    normalized = path.with_name(f".{path.stem}-normalized{path.suffix}")
    options: dict[str, object] = {}
    if dpi_value is not None:
        options["dpi"] = dpi_value
    if icc_profile is not None:
        options["icc_profile"] = icc_profile
    if output_format is OutputFormat.TIFF and compression == "lzw":
        options["compression"] = "tiff_lzw"
    converted.save(normalized, format=output_format.value, **options)
    normalized.replace(path)


def _supported_formats(fig: Figure) -> set[str]:
    supported: set[str] = set()
    for name in fig.canvas.get_supported_filetypes():
        try:
            supported.add(coerce_format(name).value)
        except ValueError:
            continue
    return supported


def _output_paths(
    requested: str | Path,
    formats: Sequence[OutputFormat | str] | None,
    permitted: tuple[OutputFormat, ...],
) -> tuple[Path, ...]:
    path = Path(requested)
    if path.suffix:
        if formats:
            raise ValueError("Do not provide formats when target_path already has a suffix.")
        coerce_format(path.suffix)
        return (path,)
    selected = tuple(coerce_format(item) for item in formats) if formats else permitted
    if not selected:
        raise ValueError(
            "No output format was requested and the profile has no applicable format rule."
        )
    return tuple(path.with_suffix(f".{item.value}") for item in selected)


def _combined_report(live: Report, file_reports: Sequence[Report]) -> Report:
    return replace(
        live,
        findings=live.findings
        + tuple(finding for report in file_reports for finding in report.findings),
    )


def _existing_target_is_regular(path: Path) -> bool | None:
    """Return whether an existing path is a regular file, without following links."""

    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except FileNotFoundError:
        return None


def _reject_non_regular_targets(paths: Sequence[Path]) -> None:
    invalid = [path for path in paths if _existing_target_is_regular(path) is False]
    if invalid:
        raise IsADirectoryError(
            "ResearchPlot only replaces existing regular files; refusing target(s): "
            + ", ".join(str(path) for path in invalid)
        )


def _commit_staged(staged: Sequence[Path], final: Sequence[Path], *, overwrite: bool) -> None:
    """Commit a staged file set and restore previous files if replacement fails."""

    if len(staged) != len(final):
        raise ValueError("Staged and final output counts do not match.")
    if len(set(final)) != len(final):
        raise ValueError("Transactional output paths must be unique.")
    backup_root = staged[0].parent / ".backup"
    backup_root.mkdir()
    backups: list[tuple[Path, Path]] = []
    claims: list[Path] = []
    committed: list[Path] = []
    try:
        for index, destination in enumerate(final):
            existing_regular = _existing_target_is_regular(destination)
            if existing_regular is False:
                raise IsADirectoryError(
                    "ResearchPlot only replaces existing regular files; "
                    f"refusing target: {destination}"
                )
            if existing_regular is True:
                if not overwrite:
                    raise FileExistsError(f"Refusing to overwrite existing output: {destination}")
                backup = backup_root / f"{index:04d}-{destination.name}"
                destination.replace(backup)
                if _existing_target_is_regular(backup) is not True:
                    backup.replace(destination)
                    raise IsADirectoryError(
                        "ResearchPlot only replaces existing regular files; "
                        f"refusing target: {destination}"
                    )
                backups.append((backup, destination))
            else:
                try:
                    destination.open("xb").close()
                except FileExistsError:
                    if _existing_target_is_regular(destination) is False:
                        raise IsADirectoryError(
                            "ResearchPlot only replaces existing regular files; "
                            f"refusing target: {destination}"
                        ) from None
                    raise FileExistsError(
                        f"Refusing to overwrite concurrently created output: {destination}"
                    ) from None
                claims.append(destination)
        for source, destination in zip(staged, final, strict=True):
            source.replace(destination)
            committed.append(destination)
    except BaseException:
        for destination in reversed(committed):
            destination.unlink(missing_ok=True)
        for destination in reversed(claims):
            destination.unlink(missing_ok=True)
        for backup, destination in reversed(backups):
            if backup.exists():
                backup.replace(destination)
        raise


def export_target(
    fig: Figure,
    target_path: str | Path,
    *,
    target: Target,
    formats: Sequence[OutputFormat | str] | None = None,
    policy: Policy | str = Policy.COMPLETE,
    dpi: int | None = None,
    overwrite: bool = False,
    attestations: Mapping[str, str] | None = None,
    metadata: Mapping[str, object] | None = None,
    savefig_kwargs: Mapping[str, Any] | None = None,
) -> ExportResult:
    """Write, inspect, and commit requested outputs with handled-error rollback."""

    selected_policy = Policy(policy)
    permitted = _permitted_formats(target)
    outputs = _output_paths(target_path, formats, permitted)
    manifest_path = Path(target_path).with_suffix(".researchplot.json")
    targets = (*outputs, manifest_path)
    _reject_non_regular_targets(targets)
    collisions = [path for path in targets if _existing_target_is_regular(path) is True]
    if collisions and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing output(s): "
            + ", ".join(str(path) for path in collisions)
        )

    live = target.validate(fig, attestations=dict(attestations or {}))
    if live.blocks(selected_policy):
        raise CompliancePolicyError(live, selected_policy)

    output_parent = outputs[0].parent.resolve()
    if any(path.parent.resolve() != output_parent for path in outputs):
        raise ValueError("All transactional outputs must share one parent directory.")
    output_parent.mkdir(parents=True, exist_ok=True)
    supported = _supported_formats(fig)
    unsupported = [
        path.suffix for path in outputs if coerce_format(path.suffix).value not in supported
    ]
    if unsupported:
        raise ValueError(f"Matplotlib cannot export: {', '.join(unsupported)}.")

    with tempfile.TemporaryDirectory(prefix=".researchplot-", dir=output_parent) as temporary:
        staging = Path(temporary)
        staged_paths = tuple(staging / path.name for path in outputs)
        options_base = dict(savefig_kwargs or {})
        options_base.setdefault("bbox_inches", None)
        export_rc = target.style().rc if target.width is not None else {}
        with mpl.rc_context(rc=export_rc):
            for staged, final in zip(staged_paths, outputs, strict=True):
                output_format = coerce_format(final.suffix)
                options = dict(options_base)
                options["format"] = output_format.value
                if output_format in {OutputFormat.PNG, OutputFormat.JPEG, OutputFormat.TIFF}:
                    required_dpi = _minimum_dpi(target, output_format)
                    options.setdefault(
                        "dpi", int(dpi if dpi is not None else required_dpi or fig.dpi)
                    )
                if output_format is OutputFormat.TIFF:
                    compression = _required_compression(target, output_format)
                    if compression == "lzw":
                        pil_kwargs = dict(options.get("pil_kwargs") or {})
                        pil_kwargs.setdefault("compression", "tiff_lzw")
                        options["pil_kwargs"] = pil_kwargs
                fig.savefig(staged, **options)
                if output_format in {OutputFormat.PNG, OutputFormat.JPEG, OutputFormat.TIFF}:
                    _normalize_raster(
                        staged,
                        target=target,
                        output_format=output_format,
                        compression=_required_compression(target, output_format),
                    )
        file_reports = tuple(
            target.audit(path, attestations=dict(attestations or {})) for path in staged_paths
        )
        report = _combined_report(live, file_reports)
        if report.blocks(selected_policy):
            raise CompliancePolicyError(report, selected_policy)

        artifacts = tuple(
            ArtifactRecord.from_path(path, relative_to=staging) for path in staged_paths
        )
        manifest = ExportManifest(
            profile=target.coordinate,
            profile_digest=profile_digest(target.profile),
            target=target.context(),
            artifacts=artifacts,
            report=report,
            metadata=dict(metadata or {}),
            sources=target.profile.sources,
            caveats=target.profile.caveats,
        )
        staged_manifest = staging / manifest_path.name
        manifest.write(staged_manifest)
        _commit_staged(
            (*staged_paths, staged_manifest),
            (*outputs, manifest_path),
            overwrite=overwrite,
        )

    return ExportResult(outputs, report, manifest, manifest_path)
