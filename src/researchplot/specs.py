"""Immutable ResearchPlot 2 project and deliverable specifications.

The specification layer is deliberately independent from Matplotlib and artifact
parsers.  It describes intent; :mod:`researchplot.project_api` resolves that intent
against an installed venue profile and turns it into a compliance plan.
"""

from __future__ import annotations

import copy
import json
import math
import re
import tomllib
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import TypeAlias, cast

from .compliance import Policy
from .models import ContentKind, FigureRole, OutputFormat

PROJECT_SCHEMA_VERSION = 3
_SPEC_ID = re.compile(r"^[a-z0-9]+(?:[a-z0-9_-]*[a-z0-9])?$")
_RULE_ID = re.compile(r"^[a-z0-9_]+(?:\.[a-z0-9_-]+)+$")
_PROFILE_COORDINATE = re.compile(
    r"^(?:[a-z0-9][a-z0-9._-]*/)?[a-z0-9]+(?:-[a-z0-9]+)*@"
    r"[0-9]{4}\.[0-9]{2}\.[0-9]+(?:#sha256:[0-9a-f]{64})?$"
)

JsonScalar: TypeAlias = str | int | float | bool | None
FrozenJson: TypeAlias = JsonScalar | tuple["FrozenJson", ...] | Mapping[str, "FrozenJson"]


class ManuscriptFormat(StrEnum):
    """Manuscript containers represented by a project specification."""

    PDF = "pdf"
    LATEX = "latex"
    DOCX = "docx"


def project_schema() -> dict[str, object]:
    """Return an independent copy of the bundled schema-v3 project contract."""

    resource = files("researchplot").joinpath("project.schema.json")
    try:
        payload = json.loads(resource.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:  # pragma: no cover - packaging guard.
        raise RuntimeError("ResearchPlot was installed without a valid project schema.") from exc
    if not isinstance(payload, dict):  # pragma: no cover - guarded by the bundled contract tests.
        raise RuntimeError("The bundled ResearchPlot project schema is not an object.")
    return cast(dict[str, object], copy.deepcopy(payload))


def _non_empty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    return value.strip()


def _spec_id(value: object, label: str) -> str:
    selected = _non_empty_string(value, label)
    if not _SPEC_ID.fullmatch(selected):
        raise ValueError(
            f"{label} must contain lowercase letters, numbers, underscores, or hyphens."
        )
    return selected


def _reject_unknown(value: Mapping[str, object], allowed: set[str], label: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{label} contains unknown fields: {', '.join(sorted(unknown))}.")


def _freeze_json(value: object, label: str) -> FrozenJson:
    if value is None or isinstance(value, (str, bool, int)):
        return cast(JsonScalar, value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite number.")
        return value
    if isinstance(value, Mapping):
        result: dict[str, FrozenJson] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"{label} keys must be non-empty strings.")
            result[key] = _freeze_json(item, f"{label}.{key}")
        return MappingProxyType(result)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json(item, f"{label}[]") for item in value)
    raise ValueError(f"{label} contains unsupported value {type(value).__name__}.")


def _thaw_json(value: FrozenJson) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _metadata(value: object, label: str) -> Mapping[str, FrozenJson]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a table/object.")
    frozen = _freeze_json(value, label)
    assert isinstance(frozen, Mapping)
    return frozen


def _iso_date(value: object, label: str) -> str:
    selected = _non_empty_string(value, label)
    try:
        date.fromisoformat(selected)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO date (YYYY-MM-DD).") from exc
    return selected


@dataclass(frozen=True, slots=True)
class ManualAttestation:
    """Reviewer-authored evidence for a profile rule classified as manual."""

    rule_id: str
    reviewer: str
    date: str
    rationale: str
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        rule_id = _non_empty_string(self.rule_id, "attestation.rule_id")
        if not _RULE_ID.fullmatch(rule_id):
            raise ValueError(f"Invalid attestation rule id {rule_id!r}.")
        object.__setattr__(self, "rule_id", rule_id)
        object.__setattr__(
            self, "reviewer", _non_empty_string(self.reviewer, "attestation.reviewer")
        )
        object.__setattr__(self, "date", _iso_date(self.date, "attestation.date"))
        object.__setattr__(
            self,
            "rationale",
            _non_empty_string(self.rationale, "attestation.rationale"),
        )
        evidence = tuple(
            dict.fromkeys(
                _non_empty_string(item, "attestation.evidence[]") for item in self.evidence
            )
        )
        object.__setattr__(self, "evidence", evidence)

    def to_dict(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "reviewer": self.reviewer,
            "date": self.date,
            "rationale": self.rationale,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True, slots=True)
class Waiver:
    """Reviewable workflow exception that never changes the venue verdict."""

    rule_id: str
    profile_digest: str
    reviewer: str
    reason: str
    expires_on: str

    def __post_init__(self) -> None:
        rule_id = _non_empty_string(self.rule_id, "waiver.rule_id")
        if not _RULE_ID.fullmatch(rule_id):
            raise ValueError(f"Invalid waiver rule id {rule_id!r}.")
        object.__setattr__(self, "rule_id", rule_id)
        if not re.fullmatch(r"[0-9a-f]{64}", self.profile_digest):
            raise ValueError("waiver.profile_digest must be a lowercase SHA-256 digest.")
        object.__setattr__(self, "reviewer", _non_empty_string(self.reviewer, "waiver.reviewer"))
        object.__setattr__(self, "reason", _non_empty_string(self.reason, "waiver.reason"))
        object.__setattr__(self, "expires_on", _iso_date(self.expires_on, "waiver.expires_on"))

    @property
    def expired(self) -> bool:
        return date.fromisoformat(self.expires_on) < date.today()

    def to_dict(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "profile_digest": self.profile_digest,
            "reviewer": self.reviewer,
            "reason": self.reason,
            "expires_on": self.expires_on,
            "expired": self.expired,
        }


AttestationValue: TypeAlias = str | ManualAttestation
WaiverValue: TypeAlias = str | Waiver


def _attestations(value: object, label: str) -> Mapping[str, AttestationValue]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a table/object.")
    result: dict[str, AttestationValue] = {}
    for key, item in value.items():
        rule_id = _non_empty_string(key, f"{label} key")
        if not _RULE_ID.fullmatch(rule_id):
            raise ValueError(f"{label} contains invalid rule id {rule_id!r}.")
        if isinstance(item, ManualAttestation):
            if item.rule_id != rule_id:
                raise ValueError(f"{label}.{rule_id} carries a different rule id.")
            result[rule_id] = item
            continue
        if isinstance(item, str):
            warnings.warn(
                "String-only attestations are a v1 compatibility form; use reviewer/date/"
                "rationale/evidence fields in schema-v3 projects.",
                DeprecationWarning,
                stacklevel=3,
            )
            result[rule_id] = _non_empty_string(item, f"{label}.{rule_id}")
            continue
        if not isinstance(item, Mapping):
            raise ValueError(f"{label}.{rule_id} must be a statement or attestation table.")
        _reject_unknown(item, {"reviewer", "date", "rationale", "evidence"}, f"{label}.{rule_id}")
        raw_evidence = item.get("evidence", [])
        if not isinstance(raw_evidence, Sequence) or isinstance(
            raw_evidence, (str, bytes, bytearray)
        ):
            raise ValueError(f"{label}.{rule_id}.evidence must be an array of strings.")
        result[rule_id] = ManualAttestation(
            rule_id,
            _non_empty_string(item.get("reviewer"), f"{label}.{rule_id}.reviewer"),
            _iso_date(item.get("date"), f"{label}.{rule_id}.date"),
            _non_empty_string(item.get("rationale"), f"{label}.{rule_id}.rationale"),
            tuple(
                _non_empty_string(entry, f"{label}.{rule_id}.evidence[]") for entry in raw_evidence
            ),
        )
    return MappingProxyType(result)


def _waivers(value: object, label: str) -> Mapping[str, WaiverValue]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a table/object.")
    result: dict[str, WaiverValue] = {}
    for key, item in value.items():
        rule_id = _non_empty_string(key, f"{label} key")
        if not _RULE_ID.fullmatch(rule_id):
            raise ValueError(f"{label} contains invalid rule id {rule_id!r}.")
        if isinstance(item, Waiver):
            if item.rule_id != rule_id:
                raise ValueError(f"{label}.{rule_id} carries a different rule id.")
            result[rule_id] = item
            continue
        if isinstance(item, str):
            warnings.warn(
                "String-only waivers are a v1 compatibility form; use profile_digest/"
                "reviewer/reason/expires_on fields in schema-v3 projects.",
                DeprecationWarning,
                stacklevel=3,
            )
            result[rule_id] = _non_empty_string(item, f"{label}.{rule_id}")
            continue
        if not isinstance(item, Mapping):
            raise ValueError(f"{label}.{rule_id} must be a reason or waiver table.")
        _reject_unknown(
            item,
            {"profile_digest", "reviewer", "reason", "expires_on"},
            f"{label}.{rule_id}",
        )
        result[rule_id] = Waiver(
            rule_id,
            _non_empty_string(item.get("profile_digest"), f"{label}.{rule_id}.profile_digest"),
            _non_empty_string(item.get("reviewer"), f"{label}.{rule_id}.reviewer"),
            _non_empty_string(item.get("reason"), f"{label}.{rule_id}.reason"),
            _iso_date(item.get("expires_on"), f"{label}.{rule_id}.expires_on"),
        )
    return MappingProxyType(result)


def _coerce_role(value: FigureRole | str) -> FigureRole:
    if isinstance(value, FigureRole):
        return value
    return FigureRole(_non_empty_string(value, "role").casefold().replace("-", "_"))


def _coerce_content(value: ContentKind | str) -> ContentKind:
    if isinstance(value, ContentKind):
        return value
    normalized = _non_empty_string(value, "content").casefold().replace("-", "_")
    normalized = {"photo": "photograph", "vector": "line_art", "data": "data_visualization"}.get(
        normalized, normalized
    )
    return ContentKind(normalized)


def _coerce_format(value: OutputFormat | str) -> OutputFormat:
    if isinstance(value, OutputFormat):
        return value
    normalized = _non_empty_string(value, "format").casefold().lstrip(".")
    normalized = {"jpg": "jpeg", "tif": "tiff"}.get(normalized, normalized)
    return OutputFormat(normalized)


def _resolve_path(value: object, root: Path, label: str, *, required: bool) -> Path | None:
    if value is None and not required:
        return None
    raw = _non_empty_string(value, label)
    path = Path(raw)
    if path.is_absolute() or path.drive or path.root:
        raise ValueError(f"{label} must be relative to the declared project root.")
    if ".." in path.parts:
        raise ValueError(f"{label} must not contain parent-directory traversal ('..').")
    project_root = root.resolve()
    resolved = (project_root / path).resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"{label} resolves outside the declared project root.") from exc
    return resolved


def _paths(value: object, root: Path, label: str) -> tuple[Path, ...]:
    if value is None:
        return ()
    raw_values: Sequence[object] = (
        [value] if isinstance(value, str) else cast(Sequence[object], value)
    )
    if not isinstance(value, (str, Sequence)) or isinstance(value, (bytes, bytearray)):
        raise ValueError(f"{label} must be a string or array of strings.")
    result = tuple(
        cast(Path, _resolve_path(item, root, label, required=True)) for item in raw_values
    )
    keys = [str(path.resolve()).casefold() for path in result]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{label} repeats a path.")
    return result


def _optional_text(value: object, label: str) -> str | None:
    return _non_empty_string(value, label) if value is not None else None


def _expect_text_array(value: object, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be an array of non-empty strings.")
    values = tuple(_non_empty_string(item, f"{label}[]") for item in value)
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must not contain duplicates.")
    return values


def _positive_integer(value: object, label: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer or null.")
    return value


def _figure_number(value: object, label: str) -> int | str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer or non-empty string.")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{label} must be positive.")
        return value
    return _non_empty_string(value, label)


def _page_numbers(value: object, label: str) -> tuple[int, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be an array of positive page numbers.")
    result: list[int] = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
            raise ValueError(f"{label} must contain only positive page numbers.")
        if item not in result:
            result.append(item)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class PanelSpec:
    """Metadata and evidence paths for one panel in a logical figure."""

    id: str
    label: str | None = None
    order: int | None = None
    description: str | None = None
    alt_text: str | None = None
    long_description: str | None = None
    source_data: tuple[Path, ...] = ()
    metadata: Mapping[str, FrozenJson] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _spec_id(self.id, "panel.id"))
        object.__setattr__(self, "label", _optional_text(self.label, "panel.label"))
        if self.order is not None and (
            not isinstance(self.order, int) or isinstance(self.order, bool) or self.order <= 0
        ):
            raise ValueError("panel.order must be a positive integer or null.")
        object.__setattr__(
            self,
            "description",
            _optional_text(self.description, "panel.description"),
        )
        object.__setattr__(self, "alt_text", _optional_text(self.alt_text, "panel.alt_text"))
        object.__setattr__(
            self,
            "long_description",
            _optional_text(self.long_description, "panel.long_description"),
        )
        source_data = tuple(Path(path) for path in self.source_data)
        keys = [str(path.resolve()).casefold() for path in source_data]
        if len(keys) != len(set(keys)):
            raise ValueError(f"panel {self.id!r} repeats a source-data path.")
        object.__setattr__(self, "source_data", source_data)
        object.__setattr__(self, "metadata", _metadata(self.metadata, "panel.metadata"))

    def to_dict(self, *, relative_to: Path | None = None) -> dict[str, object]:
        paths: list[str] = []
        for path in self.source_data:
            selected = path
            if relative_to is not None:
                try:
                    selected = path.relative_to(relative_to)
                except ValueError:
                    pass
            paths.append(selected.as_posix())
        payload: dict[str, object] = {
            "id": self.id,
            "label": self.label,
            "order": self.order,
            "description": self.description,
            "alt_text": self.alt_text,
            "long_description": self.long_description,
            "source_data": paths,
            "metadata": _thaw_json(cast(FrozenJson, self.metadata)),
        }
        return {key: value for key, value in payload.items() if value is not None}


@dataclass(frozen=True, slots=True)
class ManuscriptMatchHint:
    """Conservative hints for locating a figure in a compiled manuscript."""

    figure_id: str
    pages: tuple[int, ...] = ()
    number: int | str | None = None
    caption: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "figure_id", _spec_id(self.figure_id, "matching_hint.figure"))
        object.__setattr__(self, "pages", _page_numbers(self.pages, "matching_hint.pages"))
        object.__setattr__(self, "number", _figure_number(self.number, "matching_hint.number"))
        object.__setattr__(
            self,
            "caption",
            _optional_text(self.caption, "matching_hint.caption"),
        )
        if not self.pages and self.number is None and self.caption is None:
            raise ValueError(
                f"matching hint for {self.figure_id!r} must provide pages, number, or caption."
            )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "figure": self.figure_id,
            "pages": list(self.pages),
            "number": self.number,
            "caption": self.caption,
        }
        return {key: value for key, value in payload.items() if value is not None and value != []}


@dataclass(frozen=True, slots=True)
class DeliverableSpec:
    """One concrete representation of a logical research figure."""

    id: str
    format: OutputFormat | str
    path: Path | None = None
    required: bool = True
    preferred: bool = False
    metadata: Mapping[str, FrozenJson] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _spec_id(self.id, "deliverable.id"))
        try:
            selected_format = _coerce_format(self.format)
            object.__setattr__(self, "format", selected_format)
        except ValueError as exc:
            raise ValueError(f"deliverable {self.id!r} has an unsupported format.") from exc
        if self.path is not None:
            selected_path = Path(self.path)
            if selected_path.suffix:
                try:
                    suffix_format = _coerce_format(selected_path.suffix)
                except ValueError as exc:
                    raise ValueError(
                        f"deliverable {self.id!r} has an unsupported filename suffix."
                    ) from exc
                if suffix_format is not selected_format:
                    raise ValueError(
                        f"deliverable {self.id!r} declares {selected_format.value!r} but its "
                        f"path uses {selected_path.suffix!r}."
                    )
            object.__setattr__(self, "path", selected_path)
        if not isinstance(self.required, bool) or not isinstance(self.preferred, bool):
            raise TypeError("deliverable required/preferred flags must be booleans.")
        object.__setattr__(self, "metadata", _metadata(self.metadata, "deliverable.metadata"))

    def to_dict(self, *, relative_to: Path | None = None) -> dict[str, object]:
        path: str | None = None
        if self.path is not None:
            selected = self.path
            if relative_to is not None:
                try:
                    selected = selected.relative_to(relative_to)
                except ValueError:
                    pass
            path = selected.as_posix()
        selected_format = cast(OutputFormat, self.format)
        payload: dict[str, object] = {
            "id": self.id,
            "format": selected_format.value,
            "path": path,
            "required": self.required,
            "preferred": self.preferred,
            "metadata": _thaw_json(cast(FrozenJson, self.metadata)),
        }
        return {key: value for key, value in payload.items() if value is not None}


@dataclass(frozen=True, slots=True)
class FigureSpec:
    """A logical figure and all representations intended for submission."""

    id: str
    deliverables: tuple[DeliverableSpec, ...]
    role: FigureRole | str = FigureRole.MAIN
    width: str | None = None
    content: ContentKind | str = ContentKind.DATA_VISUALIZATION
    number: int | str | None = None
    caption: str | None = None
    alt_text: str | None = None
    long_description: str | None = None
    key_trends: tuple[str, ...] = ()
    panels: tuple[PanelSpec, ...] = ()
    source_data: tuple[Path, ...] = ()
    data_table: Path | None = None
    attachments: tuple[Path, ...] = ()
    attestations: Mapping[str, AttestationValue] = field(default_factory=dict)
    waivers: Mapping[str, WaiverValue] = field(default_factory=dict)
    metadata: Mapping[str, FrozenJson] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _spec_id(self.id, "figure.id"))
        object.__setattr__(self, "role", _coerce_role(self.role))
        object.__setattr__(self, "content", _coerce_content(self.content))
        object.__setattr__(self, "number", _figure_number(self.number, "figure.number"))
        if self.width is not None:
            object.__setattr__(self, "width", _non_empty_string(self.width, "figure.width"))
        deliverables = tuple(self.deliverables)
        if not deliverables:
            raise ValueError(f"figure {self.id!r} must define at least one deliverable.")
        if not all(isinstance(item, DeliverableSpec) for item in deliverables):
            raise TypeError("figure.deliverables must contain DeliverableSpec values.")
        ids = [item.id for item in deliverables]
        if len(ids) != len(set(ids)):
            raise ValueError(f"figure {self.id!r} repeats a deliverable id.")
        if not any(item.required for item in deliverables):
            raise ValueError(f"figure {self.id!r} must have at least one required deliverable.")
        preferred = [item.id for item in deliverables if item.preferred]
        if len(preferred) > 1:
            raise ValueError(f"figure {self.id!r} has more than one preferred deliverable.")
        object.__setattr__(self, "deliverables", deliverables)
        panels = tuple(self.panels)
        if not all(isinstance(item, PanelSpec) for item in panels):
            raise TypeError("figure.panels must contain PanelSpec values.")
        panel_ids = [item.id for item in panels]
        if len(panel_ids) != len(set(panel_ids)):
            raise ValueError(f"figure {self.id!r} repeats a panel id.")
        panel_labels = [item.label.casefold() for item in panels if item.label is not None]
        if len(panel_labels) != len(set(panel_labels)):
            raise ValueError(f"figure {self.id!r} repeats a panel label.")
        panel_orders = [item.order for item in panels if item.order is not None]
        if panel_orders and len(panel_orders) != len(panels):
            raise ValueError(
                f"figure {self.id!r} must give every panel an order when any order is set."
            )
        if len(panel_orders) != len(set(panel_orders)):
            raise ValueError(f"figure {self.id!r} repeats a panel order.")
        object.__setattr__(self, "panels", panels)
        source_data = tuple(Path(path) for path in self.source_data)
        source_keys = [str(path.resolve()).casefold() for path in source_data]
        if len(source_keys) != len(set(source_keys)):
            raise ValueError(f"figure {self.id!r} repeats a source-data path.")
        object.__setattr__(self, "source_data", source_data)
        attachments = tuple(Path(path) for path in self.attachments)
        attachment_keys = [str(path.resolve()).casefold() for path in attachments]
        if len(attachment_keys) != len(set(attachment_keys)):
            raise ValueError(f"figure {self.id!r} repeats an attachment path.")
        object.__setattr__(self, "attachments", attachments)
        if self.data_table is not None:
            object.__setattr__(self, "data_table", Path(self.data_table))
        object.__setattr__(self, "caption", _optional_text(self.caption, "figure.caption"))
        object.__setattr__(self, "alt_text", _optional_text(self.alt_text, "figure.alt_text"))
        object.__setattr__(
            self,
            "long_description",
            _optional_text(self.long_description, "figure.long_description"),
        )
        key_trends = tuple(
            dict.fromkeys(
                _non_empty_string(item, "figure.key_trends[]") for item in self.key_trends
            )
        )
        object.__setattr__(self, "key_trends", key_trends)
        object.__setattr__(self, "attestations", _attestations(self.attestations, "attestations"))
        object.__setattr__(self, "waivers", _waivers(self.waivers, "waivers"))
        object.__setattr__(self, "metadata", _metadata(self.metadata, "figure.metadata"))

    @property
    def attestation_statements(self) -> Mapping[str, str]:
        """Compatibility projection consumed by the manual-rule engine."""

        return MappingProxyType(
            {
                rule_id: value if isinstance(value, str) else value.rationale
                for rule_id, value in self.attestations.items()
            }
        )

    @property
    def active_waiver_rule_ids(self) -> tuple[str, ...]:
        """Return non-expired waiver IDs without changing compliance semantics."""

        return tuple(
            rule_id
            for rule_id, value in self.waivers.items()
            if isinstance(value, str) or not value.expired
        )

    def to_dict(self, *, relative_to: Path | None = None) -> dict[str, object]:
        role = cast(FigureRole, self.role)
        content = cast(ContentKind, self.content)
        source_data: list[str] = []
        for path in self.source_data:
            selected = path
            if relative_to is not None:
                try:
                    selected = path.relative_to(relative_to)
                except ValueError:
                    pass
            source_data.append(selected.as_posix())
        attachments: list[str] = []
        for path in self.attachments:
            selected = path
            if relative_to is not None:
                try:
                    selected = path.relative_to(relative_to)
                except ValueError:
                    pass
            attachments.append(selected.as_posix())
        data_table: str | None = None
        if self.data_table is not None:
            selected_table = self.data_table
            if relative_to is not None:
                try:
                    selected_table = selected_table.relative_to(relative_to)
                except ValueError:
                    pass
            data_table = selected_table.as_posix()
        payload: dict[str, object] = {
            "id": self.id,
            "role": role.value,
            "width": self.width,
            "content": content.value,
            "number": self.number,
            "caption": self.caption,
            "alt_text": self.alt_text,
            "long_description": self.long_description,
            "key_trends": list(self.key_trends),
            "panels": [item.to_dict(relative_to=relative_to) for item in self.panels],
            "source_data": source_data,
            "data_table": data_table,
            "attachments": attachments,
            "attestations": {
                rule_id: value
                if isinstance(value, str)
                else {
                    "reviewer": value.reviewer,
                    "date": value.date,
                    "rationale": value.rationale,
                    "evidence": list(value.evidence),
                }
                for rule_id, value in self.attestations.items()
            },
            "waivers": {
                rule_id: value
                if isinstance(value, str)
                else {
                    "profile_digest": value.profile_digest,
                    "reviewer": value.reviewer,
                    "reason": value.reason,
                    "expires_on": value.expires_on,
                }
                for rule_id, value in self.waivers.items()
            },
            "metadata": _thaw_json(cast(FrozenJson, self.metadata)),
            "deliverables": [item.to_dict(relative_to=relative_to) for item in self.deliverables],
        }
        return {key: value for key, value in payload.items() if value is not None}


@dataclass(frozen=True, slots=True)
class ManuscriptSpec:
    """An optional compiled manuscript plus conservative figure matching hints."""

    path: Path
    format: ManuscriptFormat | str
    required: bool = False
    matching_hints: tuple[ManuscriptMatchHint, ...] = ()
    metadata: Mapping[str, FrozenJson] = field(default_factory=dict)

    def __post_init__(self) -> None:
        selected_path = Path(self.path)
        object.__setattr__(self, "path", selected_path)
        if not isinstance(self.format, ManuscriptFormat):
            object.__setattr__(
                self,
                "format",
                ManuscriptFormat(_non_empty_string(self.format, "manuscript.format").casefold()),
            )
        selected_format = cast(ManuscriptFormat, self.format)
        expected_suffixes = {
            ManuscriptFormat.PDF: {".pdf"},
            ManuscriptFormat.DOCX: {".docx"},
            ManuscriptFormat.LATEX: {".tex", ".ltx"},
        }[selected_format]
        if selected_path.suffix.casefold() not in expected_suffixes:
            choices = ", ".join(sorted(expected_suffixes))
            raise ValueError(
                f"manuscript format {selected_format.value!r} requires a {choices} path."
            )
        if not isinstance(self.required, bool):
            raise TypeError("manuscript.required must be a boolean.")
        hints = tuple(self.matching_hints)
        if not all(isinstance(item, ManuscriptMatchHint) for item in hints):
            raise TypeError("manuscript.matching_hints must contain ManuscriptMatchHint values.")
        hint_ids = [item.figure_id for item in hints]
        if len(hint_ids) != len(set(hint_ids)):
            raise ValueError("manuscript.matching_hints may contain only one hint per figure.")
        object.__setattr__(self, "matching_hints", hints)
        object.__setattr__(self, "metadata", _metadata(self.metadata, "manuscript.metadata"))

    def to_dict(self, *, relative_to: Path | None = None) -> dict[str, object]:
        selected = self.path
        if relative_to is not None:
            try:
                selected = selected.relative_to(relative_to)
            except ValueError:
                pass
        manuscript_format = cast(ManuscriptFormat, self.format)
        payload: dict[str, object] = {
            "path": selected.as_posix(),
            "format": manuscript_format.value,
            "required": self.required,
            "matching_hints": [item.to_dict() for item in self.matching_hints],
            "metadata": _thaw_json(cast(FrozenJson, self.metadata)),
        }
        return payload


@dataclass(frozen=True, slots=True)
class ProjectSpec:
    """Strict schema-v3 project intent, independent of an installed profile object."""

    profile: str
    figures: tuple[FigureSpec, ...]
    policy: Policy | str = Policy.COMPLETE
    manuscript: ManuscriptSpec | None = None
    lock_path: Path | None = None
    metadata: Mapping[str, FrozenJson] = field(default_factory=dict)
    config_path: Path | None = field(default=None, compare=False, repr=False)
    schema_version: int = PROJECT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        profile = _non_empty_string(self.profile, "profile")
        if not _PROFILE_COORDINATE.fullmatch(profile):
            raise ValueError(
                "schema-v3 projects require an exact "
                "'[namespace/]profile-id@YYYY.MM.PATCH[#sha256:<digest>]' coordinate."
            )
        object.__setattr__(self, "profile", profile)
        try:
            object.__setattr__(self, "policy", Policy(self.policy))
        except ValueError as exc:
            choices = ", ".join(item.value for item in Policy)
            raise ValueError(f"policy must be one of: {choices}.") from exc
        if self.schema_version != PROJECT_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {PROJECT_SCHEMA_VERSION}, not {self.schema_version!r}."
            )
        figures = tuple(self.figures)
        if not figures:
            raise ValueError("A schema-v3 project must contain at least one figure.")
        if not all(isinstance(item, FigureSpec) for item in figures):
            raise TypeError("figures must contain FigureSpec values.")
        ids = [item.id for item in figures]
        if len(ids) != len(set(ids)):
            raise ValueError("Project figure ids must be unique.")
        paths = [
            deliverable.path
            for figure in figures
            for deliverable in figure.deliverables
            if deliverable.path is not None
        ]
        normalized = [str(path.resolve()).casefold() for path in paths]
        if len(normalized) != len(set(normalized)):
            raise ValueError("Project deliverable paths must be unique.")
        object.__setattr__(self, "figures", figures)
        if self.manuscript is not None:
            if not isinstance(self.manuscript, ManuscriptSpec):
                raise TypeError("manuscript must be a ManuscriptSpec value or null.")
            unknown_hints = {item.figure_id for item in self.manuscript.matching_hints} - set(ids)
            if unknown_hints:
                raise ValueError(
                    "Manuscript matching hints reference unknown figures: "
                    + ", ".join(sorted(unknown_hints))
                    + "."
                )
        if self.lock_path is not None:
            object.__setattr__(self, "lock_path", Path(self.lock_path))
        if self.config_path is not None:
            object.__setattr__(self, "config_path", Path(self.config_path))
        object.__setattr__(self, "metadata", _metadata(self.metadata, "project.metadata"))

    @property
    def root(self) -> Path:
        return self.config_path.parent if self.config_path is not None else Path.cwd()

    def figure(self, figure_id: str) -> FigureSpec:
        """Return one figure by id or raise an actionable error."""

        try:
            return next(item for item in self.figures if item.id == figure_id)
        except StopIteration as exc:
            choices = ", ".join(item.id for item in self.figures)
            raise KeyError(f"Unknown figure {figure_id!r}. Available figures: {choices}.") from exc

    def to_dict(self) -> dict[str, object]:
        policy = cast(Policy, self.policy)
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "profile": self.profile,
            "policy": policy.value,
            "lock": self.lock_path.as_posix() if self.lock_path is not None else None,
            "metadata": _thaw_json(cast(FrozenJson, self.metadata)),
            "manuscript": self.manuscript.to_dict(relative_to=self.root)
            if self.manuscript is not None
            else None,
            "figures": [item.to_dict(relative_to=self.root) for item in self.figures],
        }
        return {key: value for key, value in payload.items() if value is not None}

    @classmethod
    def load(cls, path: str | Path = "researchplot.toml") -> ProjectSpec:
        """Load and strictly validate a schema-v3 TOML project."""

        config_path = Path(path).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"ResearchPlot configuration not found: {config_path}")
        try:
            payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ValueError(f"Could not parse {config_path}: {exc}") from exc
        if not isinstance(payload, dict):  # pragma: no cover - tomllib always returns a dict.
            raise ValueError("Project configuration must be a TOML table.")
        root_payload: object = None
        tool = payload.get("tool")
        if isinstance(tool, dict):
            root_payload = tool.get("researchplot")
        settings = root_payload if isinstance(root_payload, dict) else payload
        return cls._from_mapping(cast(dict[str, object], settings), config_path=config_path)

    @classmethod
    def _from_mapping(cls, settings: Mapping[str, object], *, config_path: Path) -> ProjectSpec:
        _reject_unknown(
            settings,
            {"schema_version", "profile", "policy", "lock", "metadata", "manuscript", "figures"},
            "project",
        )
        version = settings.get("schema_version")
        if not isinstance(version, int) or isinstance(version, bool) or version != 3:
            raise ValueError("project.schema_version must be the integer 3.")
        profile = _non_empty_string(settings.get("profile"), "project.profile")
        policy = _non_empty_string(settings.get("policy", Policy.COMPLETE.value), "project.policy")
        root = config_path.parent
        lock_path = _resolve_path(settings.get("lock"), root, "project.lock", required=False)

        raw_manuscript = settings.get("manuscript")
        manuscript: ManuscriptSpec | None = None
        if raw_manuscript is not None:
            if not isinstance(raw_manuscript, Mapping):
                raise ValueError("project.manuscript must be a table.")
            _reject_unknown(
                raw_manuscript,
                {"path", "format", "required", "matching_hints", "metadata"},
                "project.manuscript",
            )
            raw_required = raw_manuscript.get("required", False)
            if not isinstance(raw_required, bool):
                raise ValueError("project.manuscript.required must be a boolean.")
            raw_hints = raw_manuscript.get("matching_hints", [])
            if not isinstance(raw_hints, list):
                raise ValueError("project.manuscript.matching_hints must be an array of tables.")
            hints: list[ManuscriptMatchHint] = []
            for hint_index, raw_hint in enumerate(raw_hints):
                hint_label = f"project.manuscript.matching_hints[{hint_index}]"
                if not isinstance(raw_hint, Mapping):
                    raise ValueError(f"{hint_label} must be a table.")
                _reject_unknown(raw_hint, {"figure", "pages", "number", "caption"}, hint_label)
                hints.append(
                    ManuscriptMatchHint(
                        figure_id=_spec_id(raw_hint.get("figure"), f"{hint_label}.figure"),
                        pages=_page_numbers(raw_hint.get("pages"), f"{hint_label}.pages"),
                        number=_figure_number(raw_hint.get("number"), f"{hint_label}.number"),
                        caption=_optional_text(raw_hint.get("caption"), f"{hint_label}.caption"),
                    )
                )
            manuscript = ManuscriptSpec(
                path=cast(
                    Path,
                    _resolve_path(
                        raw_manuscript.get("path"), root, "project.manuscript.path", required=True
                    ),
                ),
                format=_non_empty_string(raw_manuscript.get("format"), "project.manuscript.format"),
                required=raw_required,
                matching_hints=tuple(hints),
                metadata=_metadata(raw_manuscript.get("metadata"), "project.manuscript.metadata"),
            )

        raw_figures = settings.get("figures")
        if not isinstance(raw_figures, list) or not raw_figures:
            raise ValueError("project.figures must be a non-empty array of tables.")
        figures = tuple(
            cls._parse_figure(raw, index=index, root=root) for index, raw in enumerate(raw_figures)
        )
        return cls(
            profile=profile,
            figures=figures,
            policy=policy,
            manuscript=manuscript,
            lock_path=lock_path,
            metadata=_metadata(settings.get("metadata"), "project.metadata"),
            config_path=config_path,
            schema_version=version,
        )

    @staticmethod
    def _parse_figure(raw: object, *, index: int, root: Path) -> FigureSpec:
        label = f"project.figures[{index}]"
        if not isinstance(raw, Mapping):
            raise ValueError(f"{label} must be a table.")
        _reject_unknown(
            raw,
            {
                "id",
                "role",
                "width",
                "content",
                "number",
                "caption",
                "alt_text",
                "long_description",
                "key_trends",
                "panels",
                "source_data",
                "data_table",
                "attachments",
                "attestations",
                "waivers",
                "metadata",
                "deliverables",
            },
            label,
        )
        raw_deliverables = raw.get("deliverables")
        if not isinstance(raw_deliverables, list) or not raw_deliverables:
            raise ValueError(f"{label}.deliverables must be a non-empty array of tables.")
        deliverables = tuple(
            ProjectSpec._parse_deliverable(
                item, label=f"{label}.deliverables[{item_index}]", root=root
            )
            for item_index, item in enumerate(raw_deliverables)
        )
        source_data = _paths(raw.get("source_data"), root, f"{label}.source_data")
        attachments = _paths(raw.get("attachments"), root, f"{label}.attachments")
        data_table = _resolve_path(
            raw.get("data_table"), root, f"{label}.data_table", required=False
        )
        raw_panels = raw.get("panels", [])
        if not isinstance(raw_panels, list):
            raise ValueError(f"{label}.panels must be an array of tables.")
        panels = tuple(
            ProjectSpec._parse_panel(
                item,
                label=f"{label}.panels[{panel_index}]",
                root=root,
            )
            for panel_index, item in enumerate(raw_panels)
        )
        width_value = raw.get("width")
        width = (
            _non_empty_string(width_value, f"{label}.width") if width_value is not None else None
        )
        caption_value = raw.get("caption")
        alt_value = raw.get("alt_text")
        return FigureSpec(
            id=_spec_id(raw.get("id"), f"{label}.id"),
            role=_non_empty_string(raw.get("role", FigureRole.MAIN.value), f"{label}.role"),
            width=width,
            content=_non_empty_string(
                raw.get("content", ContentKind.DATA_VISUALIZATION.value), f"{label}.content"
            ),
            number=_figure_number(raw.get("number"), f"{label}.number"),
            caption=_non_empty_string(caption_value, f"{label}.caption")
            if caption_value is not None
            else None,
            alt_text=_non_empty_string(alt_value, f"{label}.alt_text")
            if alt_value is not None
            else None,
            long_description=_optional_text(
                raw.get("long_description"), f"{label}.long_description"
            ),
            key_trends=_expect_text_array(raw.get("key_trends"), f"{label}.key_trends"),
            panels=panels,
            source_data=source_data,
            data_table=data_table,
            attachments=attachments,
            attestations=_attestations(raw.get("attestations"), f"{label}.attestations"),
            waivers=_waivers(raw.get("waivers"), f"{label}.waivers"),
            metadata=_metadata(raw.get("metadata"), f"{label}.metadata"),
            deliverables=deliverables,
        )

    @staticmethod
    def _parse_panel(raw: object, *, label: str, root: Path) -> PanelSpec:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{label} must be a table.")
        _reject_unknown(
            raw,
            {
                "id",
                "label",
                "order",
                "description",
                "alt_text",
                "long_description",
                "source_data",
                "metadata",
            },
            label,
        )
        return PanelSpec(
            id=_spec_id(raw.get("id"), f"{label}.id"),
            label=_optional_text(raw.get("label"), f"{label}.label"),
            order=_positive_integer(raw.get("order"), f"{label}.order"),
            description=_optional_text(raw.get("description"), f"{label}.description"),
            alt_text=_optional_text(raw.get("alt_text"), f"{label}.alt_text"),
            long_description=_optional_text(
                raw.get("long_description"), f"{label}.long_description"
            ),
            source_data=_paths(raw.get("source_data"), root, f"{label}.source_data"),
            metadata=_metadata(raw.get("metadata"), f"{label}.metadata"),
        )

    @staticmethod
    def _parse_deliverable(raw: object, *, label: str, root: Path) -> DeliverableSpec:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{label} must be a table.")
        _reject_unknown(raw, {"id", "format", "path", "required", "preferred", "metadata"}, label)
        required = raw.get("required", True)
        preferred = raw.get("preferred", False)
        if not isinstance(required, bool) or not isinstance(preferred, bool):
            raise ValueError(f"{label}.required and .preferred must be booleans.")
        return DeliverableSpec(
            id=_spec_id(raw.get("id"), f"{label}.id"),
            format=_non_empty_string(raw.get("format"), f"{label}.format"),
            path=_resolve_path(raw.get("path"), root, f"{label}.path", required=False),
            required=required,
            preferred=preferred,
            metadata=_metadata(raw.get("metadata"), f"{label}.metadata"),
        )
