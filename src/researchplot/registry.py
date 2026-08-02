"""Offline profile-schema validation, loading, and name resolution."""

from __future__ import annotations

import copy
import difflib
import hashlib
import json
import math
import re
import warnings
from collections.abc import Iterable, Mapping
from datetime import date, timedelta
from functools import lru_cache
from importlib.metadata import entry_points
from importlib.resources import files
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

from .models import (
    ConstraintOperator,
    ContentKind,
    FigureRole,
    OutputFormat,
    RuleApplicability,
    RuleConstraint,
    RuleLevel,
    RulePhase,
    RuleValue,
    SourceRef,
    VenueKind,
    VenueProfile,
    VenueResolutionWarning,
    VenueRule,
    VerificationMode,
)

_PROFILE_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_PROFILE_REVISION = re.compile(r"^[0-9]{4}\.[0-9]{2}\.[0-9]+$")
_RULE_ID = re.compile(r"^[a-z0-9_]+(?:\.[a-z0-9_-]+)+$")
_PROBE_ID = re.compile(r"^[a-z0-9_]+(?:\.[a-z0-9_]+)+$")
_LEGACY_ONLY = {"cell", "pnas", "science", "springer"}
_MAX_PROFILE_BYTES = 1_000_000
_NUMERIC_OPERATORS = {
    ConstraintOperator.GT,
    ConstraintOperator.GTE,
    ConstraintOperator.LT,
    ConstraintOperator.LTE,
    ConstraintOperator.APPROX,
}
_SEQUENCE_OPERATORS = {
    ConstraintOperator.IN,
    ConstraintOperator.NOT_IN,
    ConstraintOperator.BETWEEN,
    ConstraintOperator.SUBSET,
}


def normalize_venue_name(value: str) -> str:
    """Normalize punctuation and spacing for unpinned profile lookup."""

    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _expect_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return cast(dict[str, Any], value)


def _reject_unknown_keys(value: dict[str, Any], *, allowed: set[str], label: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{label} contains unknown fields: {', '.join(sorted(unknown))}.")


def _expect_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    return value


def _expect_strings(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"{label} must be an array of non-empty strings.")
    result = tuple(cast(list[str], value))
    if len(result) != len(set(result)):
        raise ValueError(f"{label} must not contain duplicates.")
    return result


def _expect_enum_tuple(
    value: object,
    enum_type: type[FigureRole] | type[ContentKind] | type[OutputFormat] | type[RulePhase],
    label: str,
) -> tuple[Any, ...]:
    raw_values = _expect_strings(value, label)
    try:
        return tuple(enum_type(item) for item in raw_values)
    except ValueError as exc:
        choices = ", ".join(item.value for item in enum_type)
        raise ValueError(f"{label} contains an invalid value; choose from: {choices}.") from exc


def _freeze_rule_value(value: object, label: str) -> RuleValue:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{label} must be finite.")
        return value
    if isinstance(value, list):
        if not value:
            raise ValueError(f"{label} must not be an empty array.")
        if all(isinstance(item, str) and item for item in value):
            result = tuple(cast(list[str], value))
            if len(result) != len(set(result)):
                raise ValueError(f"{label} must not contain duplicates.")
            return result
        if all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
            numeric = tuple(float(item) for item in cast(list[float], value))
            if not all(math.isfinite(item) for item in numeric):
                raise ValueError(f"{label} must contain only finite numbers.")
            if len(numeric) != len(set(numeric)):
                raise ValueError(f"{label} must not contain duplicates.")
            return numeric
    raise ValueError(f"{label} contains an unsupported rule value.")


def _validate_date(value: str, label: str) -> str:
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must use YYYY-MM-DD format.") from exc
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON numeric value {value!r} is not finite.")


@lru_cache(maxsize=1)
def _load_profile_schema() -> dict[str, Any]:
    resource = files("researchplot").joinpath("profile.schema.json")
    try:
        schema = json.loads(
            resource.read_text(encoding="utf-8"), parse_constant=_reject_json_constant
        )
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise RuntimeError("ResearchPlot was installed without a valid profile schema.") from exc
    return _expect_mapping(schema, "profile.schema.json")


def profile_schema() -> dict[str, Any]:
    """Return a copy of the bundled JSON Schema (no network access required)."""

    return copy.deepcopy(_load_profile_schema())


def _validate_schema_envelope(data: dict[str, Any], filename: str) -> None:
    """Validate the schema envelope before semantic profile validation.

    ResearchPlot intentionally avoids a runtime dependency on ``jsonschema``.
    The bundled schema is authoritative for external tools, while this loader
    implements the same constraints and the cross-reference checks that JSON
    Schema cannot express.
    """

    schema = _load_profile_schema()
    required = set(cast(list[str], schema["required"]))
    missing = required - set(data)
    if missing:
        raise ValueError(f"{filename} is missing required fields: {', '.join(sorted(missing))}.")
    allowed = set(cast(dict[str, Any], schema["properties"]))
    _reject_unknown_keys(data, allowed=allowed, label=filename)
    if data.get("schema_version") != 2:
        raise ValueError(f"{filename} uses an unsupported profile schema; expected version 2.")


def _parse_source(raw_source: object, filename: str, index: int) -> SourceRef:
    label = f"{filename}.sources[{index}]"
    source = _expect_mapping(raw_source, label)
    _reject_unknown_keys(
        source,
        allowed={"id", "title", "url", "locator", "retrieved_on", "verified_on"},
        label=label,
    )
    url = _expect_string(source.get("url"), f"{label}.url")
    parsed_url = urlsplit(url)
    if (
        parsed_url.scheme.casefold() != "https"
        or not parsed_url.hostname
        or parsed_url.username is not None
        or parsed_url.password is not None
    ):
        raise ValueError(f"{label}.url must use HTTPS.")
    return SourceRef(
        id=_expect_string(source.get("id"), f"{label}.id"),
        title=_expect_string(source.get("title"), f"{label}.title"),
        url=url,
        locator=_expect_string(source.get("locator"), f"{label}.locator"),
        retrieved_on=_validate_date(
            _expect_string(source.get("retrieved_on"), f"{label}.retrieved_on"),
            f"{label}.retrieved_on",
        ),
        verified_on=_validate_date(
            _expect_string(source.get("verified_on"), f"{label}.verified_on"),
            f"{label}.verified_on",
        ),
    )


def _parse_constraint(raw_constraint: object, label: str) -> RuleConstraint:
    constraint = _expect_mapping(raw_constraint, label)
    _reject_unknown_keys(
        constraint,
        allowed={"operator", "value", "unit", "tolerance"},
        label=label,
    )
    try:
        operator = ConstraintOperator(
            _expect_string(constraint.get("operator"), f"{label}.operator")
        )
    except ValueError as exc:
        raise ValueError(f"{label}.operator is not supported.") from exc
    value = _freeze_rule_value(constraint.get("value"), f"{label}.value")
    unit = constraint.get("unit")
    if unit is not None and (not isinstance(unit, str) or not unit.strip()):
        raise ValueError(f"{label}.unit must be a non-empty string or null.")
    tolerance = constraint.get("tolerance")
    if tolerance is not None and (
        not isinstance(tolerance, (int, float))
        or isinstance(tolerance, bool)
        or not math.isfinite(float(tolerance))
        or float(tolerance) < 0
    ):
        raise ValueError(f"{label}.tolerance must be a non-negative number or null.")
    if operator in _NUMERIC_OPERATORS and (
        not isinstance(value, (int, float)) or isinstance(value, bool)
    ):
        raise ValueError(f"{label}.{operator.value} requires a numeric value.")
    if operator in _SEQUENCE_OPERATORS and not isinstance(value, tuple):
        raise ValueError(f"{label}.{operator.value} requires an array value.")
    if operator is ConstraintOperator.BETWEEN and (
        len(cast(tuple[object, ...], value)) != 2
        or not all(isinstance(item, (int, float)) for item in cast(tuple[object, ...], value))
    ):
        raise ValueError(f"{label}.between requires exactly two numeric bounds.")
    if operator is ConstraintOperator.BETWEEN:
        bounds = cast(tuple[float, float], value)
        if bounds[0] > bounds[1]:
            raise ValueError(f"{label}.between lower bound must not exceed its upper bound.")
    if operator is not ConstraintOperator.APPROX and tolerance is not None:
        raise ValueError(f"{label}.tolerance is only valid with the approx operator.")
    return RuleConstraint(
        operator=operator,
        value=value,
        unit=unit,
        tolerance=float(tolerance) if tolerance is not None else None,
    )


def _parse_applicability(raw: object, label: str) -> RuleApplicability:
    applicability = _expect_mapping(raw, label)
    _reject_unknown_keys(
        applicability,
        allowed={"roles", "content_kinds", "output_formats", "widths"},
        label=label,
    )
    return RuleApplicability(
        roles=cast(
            tuple[FigureRole, ...],
            _expect_enum_tuple(applicability.get("roles", []), FigureRole, f"{label}.roles"),
        ),
        content_kinds=cast(
            tuple[ContentKind, ...],
            _expect_enum_tuple(
                applicability.get("content_kinds", []),
                ContentKind,
                f"{label}.content_kinds",
            ),
        ),
        output_formats=cast(
            tuple[OutputFormat, ...],
            _expect_enum_tuple(
                applicability.get("output_formats", []),
                OutputFormat,
                f"{label}.output_formats",
            ),
        ),
        widths=_expect_strings(applicability.get("widths", []), f"{label}.widths"),
    )


def _parse_rule(raw_rule: object, filename: str, index: int, source_ids: set[str]) -> VenueRule:
    label = f"{filename}.rules[{index}]"
    rule = _expect_mapping(raw_rule, label)
    _reject_unknown_keys(
        rule,
        allowed={
            "id",
            "probe",
            "constraint",
            "applies_to",
            "verification",
            "phases",
            "level",
            "source_ids",
            "description",
        },
        label=label,
    )
    rule_id = _expect_string(rule.get("id"), f"{label}.id")
    if not _RULE_ID.fullmatch(rule_id):
        raise ValueError(f"{label}.id is not a valid dotted rule identifier.")
    probe = _expect_string(rule.get("probe"), f"{label}.probe")
    if not _PROBE_ID.fullmatch(probe):
        raise ValueError(f"{label}.probe is not a valid dotted probe identifier.")
    linked_sources = _expect_strings(rule.get("source_ids"), f"{label}.source_ids")
    if not linked_sources:
        raise ValueError(f"{filename}.{rule_id} must cite at least one source.")
    missing_sources = set(linked_sources) - source_ids
    if missing_sources:
        raise ValueError(
            f"{filename}.{rule_id} references unknown sources: "
            f"{', '.join(sorted(missing_sources))}."
        )
    try:
        level = RuleLevel(_expect_string(rule.get("level"), f"{label}.level"))
        verification = VerificationMode(
            _expect_string(rule.get("verification"), f"{label}.verification")
        )
    except ValueError as exc:
        raise ValueError(f"{filename}.{rule_id} has invalid rule metadata.") from exc
    phases = cast(
        tuple[RulePhase, ...],
        _expect_enum_tuple(rule.get("phases"), RulePhase, f"{label}.phases"),
    )
    if not phases:
        raise ValueError(f"{filename}.{rule_id} must define at least one validation phase.")
    return VenueRule(
        id=rule_id,
        probe=probe,
        constraint=_parse_constraint(rule.get("constraint"), f"{label}.constraint"),
        applies_to=_parse_applicability(rule.get("applies_to"), f"{label}.applies_to"),
        verification=verification,
        level=level,
        source_ids=linked_sources,
        description=_expect_string(rule.get("description"), f"{label}.description"),
        phases=phases,
    )


def _canonical_digest(data: dict[str, Any]) -> str:
    canonical = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(canonical).hexdigest()


def _parse_profile(
    payload: object, filename: str, *, require_filename_match: bool = True
) -> VenueProfile:
    """Parse and semantically validate one schema-v2 profile payload."""

    data = _expect_mapping(payload, filename)
    _validate_schema_envelope(data, filename)

    profile_id = _expect_string(data.get("id"), f"{filename}.id")
    if not _PROFILE_ID.fullmatch(profile_id):
        raise ValueError(f"{filename}.id must be a lowercase kebab-case identifier.")
    revision = _expect_string(data.get("revision"), f"{filename}.revision")
    if not _PROFILE_REVISION.fullmatch(revision):
        raise ValueError(f"{filename}.revision must use YYYY.MM.PATCH format.")
    raw_year = data.get("year")
    if raw_year is not None and (not isinstance(raw_year, int) or isinstance(raw_year, bool)):
        raise ValueError(f"{filename}.year must be an integer or null.")

    raw_sources = data.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError(f"{filename}.sources must be a non-empty array.")
    sources = tuple(
        _parse_source(raw_source, filename, index) for index, raw_source in enumerate(raw_sources)
    )
    source_ids = [source.id for source in sources]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError(f"{filename} repeats a source id.")

    raw_rules = data.get("rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise ValueError(f"{filename}.rules must be a non-empty array.")
    rules = tuple(
        _parse_rule(raw_rule, filename, index, set(source_ids))
        for index, raw_rule in enumerate(raw_rules)
    )
    rule_ids = [rule.id for rule in rules]
    if len(rule_ids) != len(set(rule_ids)):
        raise ValueError(f"{filename} repeats a rule id.")

    filename_stem = filename.removesuffix(".json").split("@", maxsplit=1)[0]
    if require_filename_match and profile_id != filename_stem:
        raise ValueError(f"{filename}.id must match its filename.")
    for profile_rule in rules:
        if profile_rule.id.startswith("figure.width.") and (
            not isinstance(profile_rule.value, (int, float))
            or isinstance(profile_rule.value, bool)
            or not math.isfinite(float(profile_rule.value))
            or float(profile_rule.value) <= 0
            or profile_rule.unit != "mm"
        ):
            raise ValueError(
                f"{filename}.{profile_rule.id} must be a positive numeric millimetre value."
            )
        if profile_rule.id.startswith("export.formats.") and (
            not isinstance(profile_rule.value, tuple) or not profile_rule.value
        ):
            raise ValueError(f"{filename}.{profile_rule.id} must list one or more formats.")
        if profile_rule.id.startswith("export.min_dpi.") and (
            not isinstance(profile_rule.value, (int, float))
            or isinstance(profile_rule.value, bool)
            or not math.isfinite(float(profile_rule.value))
            or float(profile_rule.value) <= 0
            or profile_rule.unit != "dpi"
        ):
            raise ValueError(f"{filename}.{profile_rule.id} must be a positive DPI value.")

    raw_default_width = data.get("default_width")
    default_width = (
        _expect_string(raw_default_width, f"{filename}.default_width")
        if raw_default_width is not None
        else None
    )
    width_options = {
        rule.id.removeprefix("figure.width.")
        for rule in rules
        if rule.id.startswith("figure.width.")
        and rule.constraint.operator in {ConstraintOperator.EQ, ConstraintOperator.APPROX}
        and isinstance(rule.value, (int, float))
        and not isinstance(rule.value, bool)
    }
    if default_width is not None and default_width not in width_options:
        raise ValueError(f"{filename} default width {default_width!r} is not defined.")
    for rule in rules:
        unknown_widths = set(rule.applies_to.widths) - width_options
        if unknown_widths:
            raise ValueError(
                f"{filename}.{rule.id} applies to undefined widths: "
                f"{', '.join(sorted(unknown_widths))}."
            )
    aliases = _expect_strings(data.get("aliases"), f"{filename}.aliases")
    return VenueProfile(
        id=profile_id,
        name=_expect_string(data.get("name"), f"{filename}.name"),
        kind=VenueKind(_expect_string(data.get("kind"), f"{filename}.kind")),
        year=raw_year,
        aliases=aliases,
        scope=_expect_string(data.get("scope"), f"{filename}.scope"),
        default_width=default_width,
        verified_on=_validate_date(
            _expect_string(data.get("verified_on"), f"{filename}.verified_on"),
            f"{filename}.verified_on",
        ),
        sources=sources,
        rules=rules,
        caveats=_expect_strings(data.get("caveats"), f"{filename}.caveats"),
        schema_version=2,
        revision=revision,
        effective_date=_validate_date(
            _expect_string(data.get("effective_date"), f"{filename}.effective_date"),
            f"{filename}.effective_date",
        ),
        digest=_canonical_digest(data),
    )


def validate_profile_data(payload: object, *, filename: str = "profile.json") -> VenueProfile:
    """Validate a JSON-compatible profile and return its immutable model."""

    return _parse_profile(payload, filename, require_filename_match=False)


def load_profile(path: str | Path) -> VenueProfile:
    """Load and validate a schema-v2 profile from a local JSON file."""

    profile_path = Path(path)
    try:
        if profile_path.stat().st_size > _MAX_PROFILE_BYTES:
            raise ValueError(
                f"Profile {profile_path} exceeds the {_MAX_PROFILE_BYTES}-byte size limit."
            )
        payload = json.loads(
            profile_path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Profile {profile_path} is not valid JSON at line {exc.lineno}, column {exc.colno}."
        ) from exc
    except OSError as exc:
        raise ValueError(f"Unable to read profile {profile_path}: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"Profile {profile_path} is invalid: {exc}") from exc
    return _parse_profile(payload, profile_path.name, require_filename_match=False)


def _revision_key(revision: str) -> tuple[int, int, int]:
    year, month, patch = revision.split(".")
    return int(year), int(month), int(patch)


def _coerce_plugin_profiles(
    value: object, *, label: str, depth: int = 0
) -> tuple[VenueProfile, ...]:
    """Normalize the deliberately small profile-pack entry-point contract."""

    if depth > 16:
        raise ValueError(f"Profile entry point {label!r} exceeded the nesting limit.")
    if callable(value):
        return _coerce_plugin_profiles(value(), label=label, depth=depth + 1)
    if isinstance(value, VenueProfile):
        return (value,)
    if isinstance(value, Mapping):
        return (validate_profile_data(dict(value), filename=f"{normalize_venue_name(label)}.json"),)
    if isinstance(value, (str, Path)):
        profile_path = Path(value)
        if profile_path.is_dir():
            paths = sorted(profile_path.glob("*.json"))
            if len(paths) > 256:
                raise ValueError(f"Profile entry point {label!r} exceeds 256 JSON files.")
            return tuple(load_profile(path) for path in paths)
        return (load_profile(profile_path),)
    if isinstance(value, Iterable):
        profiles: list[VenueProfile] = []
        for index, item in enumerate(value):
            if index >= 256:
                raise ValueError(f"Profile entry point {label!r} exceeds 256 items.")
            profiles.extend(
                _coerce_plugin_profiles(item, label=f"{label}-{index}", depth=depth + 1)
            )
            if len(profiles) > 256:
                raise ValueError(f"Profile entry point {label!r} exceeds 256 profiles.")
        return tuple(profiles)
    raise TypeError(
        f"Profile entry point {label!r} must expose a VenueProfile, profile mapping, "
        "JSON path, directory, iterable of those values, or a zero-argument callable."
    )


def _load_plugin_profiles() -> tuple[VenueProfile, ...]:
    loaded: list[VenueProfile] = []
    for entry_point in entry_points().select(group="researchplot.profiles"):
        label = f"{entry_point.module}:{entry_point.attr or ''}".rstrip(":")
        try:
            loaded.extend(_coerce_plugin_profiles(entry_point.load(), label=label))
        except Exception as exc:
            warnings.warn(
                f"Ignoring ResearchPlot profile pack {entry_point.name!r} from {label}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
    return tuple(loaded)


@lru_cache(maxsize=1)
def _load_profiles() -> tuple[VenueProfile, ...]:
    profile_root = files("researchplot.profiles")
    bundled: list[VenueProfile] = []
    for resource in sorted(profile_root.iterdir(), key=lambda item: item.name):
        if resource.name.endswith(".json"):
            bundled.append(
                _parse_profile(
                    json.loads(
                        resource.read_text(encoding="utf-8"),
                        parse_constant=_reject_json_constant,
                    ),
                    resource.name,
                )
            )
    if not bundled:
        raise RuntimeError("ResearchPlot was installed without its venue profiles.")
    loaded: list[VenueProfile] = []
    coordinates: set[str] = set()
    aliases: dict[str, str] = {}

    def add_profile(profile: VenueProfile) -> None:
        if profile.coordinate in coordinates:
            raise ValueError(f"Venue profile coordinate {profile.coordinate!r} is duplicated.")
        for candidate in (profile.id, profile.name, *profile.aliases):
            normalized = normalize_venue_name(candidate)
            previous = aliases.get(normalized)
            if previous is not None and previous != profile.id:
                raise ValueError(
                    f"Profile alias {candidate!r} collides between {previous!r} and {profile.id!r}."
                )
            aliases[normalized] = profile.id
        coordinates.add(profile.coordinate)
        loaded.append(profile)

    for profile in bundled:
        add_profile(profile)
    for profile in _load_plugin_profiles():
        try:
            add_profile(profile)
        except ValueError as exc:
            warnings.warn(
                f"Ignoring conflicting ResearchPlot profile {profile.coordinate!r}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
    return tuple(loaded)


def list_profiles(
    *, kind: VenueKind | str | None = None, year: int | None = None
) -> list[VenueProfile]:
    """Return bundled and installed profile revisions, optionally filtered."""

    selected_kind = VenueKind(kind) if isinstance(kind, str) else kind
    return [
        profile
        for profile in _load_profiles()
        if (selected_kind is None or profile.kind is selected_kind)
        and (year is None or profile.year == year)
    ]


def list_venues(
    *, kind: VenueKind | str | None = None, year: int | None = None
) -> list[VenueProfile]:
    """Compatibility alias for :func:`list_profiles`."""

    return list_profiles(kind=kind, year=year)


def search_venues(query: str) -> list[VenueProfile]:
    """Return profile revisions whose IDs, names, or aliases contain ``query``."""

    needle = normalize_venue_name(query)
    if not needle:
        return list(_load_profiles())
    return [
        profile
        for profile in _load_profiles()
        if any(
            needle in normalize_venue_name(candidate)
            for candidate in (profile.coordinate, profile.id, profile.name, *profile.aliases)
        )
    ]


def search_profiles(query: str) -> list[VenueProfile]:
    """Return profile revisions whose IDs, names, or aliases contain ``query``."""

    return search_venues(query)


def _select_latest(profiles: list[VenueProfile]) -> VenueProfile:
    return max(profiles, key=lambda profile: _revision_key(profile.revision))


def _resolve_pinned(query: str, profiles: tuple[VenueProfile, ...]) -> VenueProfile:
    base, separator, revision = query.strip().rpartition("@")
    if not separator or not base or not revision:
        raise ValueError(f"Invalid profile coordinate {query!r}; use '<profile-id>@YYYY.MM.PATCH'.")
    normalized_base = normalize_venue_name(base)
    candidates = [
        profile
        for profile in profiles
        if normalized_base == normalize_venue_name(profile.id) and profile.revision == revision
    ]
    if len(candidates) == 1:
        return candidates[0]
    base_profiles = [
        profile for profile in profiles if normalized_base == normalize_venue_name(profile.id)
    ]
    if base_profiles:
        revisions = ", ".join(
            profile.coordinate
            for profile in sorted(base_profiles, key=lambda item: _revision_key(item.revision))
        )
        raise ValueError(
            f"Unknown revision {revision!r} for {base!r}. Available coordinates: {revisions}."
        )
    raise ValueError(
        f"Unknown profile id {base!r}. Coordinates must use an exact profile id before '@'."
    )


def resolve_profile(query: str | VenueProfile) -> VenueProfile:
    """Resolve an exact coordinate or a warned, unpinned human venue name."""

    if isinstance(query, VenueProfile):
        return query
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Venue query cannot be empty.")
    profiles = _load_profiles()
    if "@" in query:
        return _resolve_pinned(query, profiles)

    normalized = normalize_venue_name(query)
    if normalized in _LEGACY_ONLY:
        raise ValueError(
            f"{query!r} is available only as an unverified legacy plotting style. "
            "Pin researchplot-venues[plots]==0.2.1 in a separate environment if needed; "
            "no compliance profile is claimed."
        )

    exact_ids = [profile for profile in profiles if normalized == normalize_venue_name(profile.id)]
    candidates: list[VenueProfile] = []
    for profile in profiles:
        names = (profile.id, profile.name, *profile.aliases)
        if normalized in {normalize_venue_name(name) for name in names}:
            candidates.append(profile)

    selected_profile: VenueProfile | None = None
    if exact_ids:
        selected_profile = _select_latest(exact_ids)
    elif candidates:
        candidate_ids = {item.id for item in candidates}
        if len(candidate_ids) == 1:
            selected_profile = _select_latest(candidates)
        elif all(item.kind is VenueKind.CONFERENCE for item in candidates):
            newest_year = max(item.year or 0 for item in candidates)
            newest = [item for item in candidates if (item.year or 0) == newest_year]
            newest_ids = {item.id for item in newest}
            if len(newest_ids) == 1:
                selected_profile = _select_latest(newest)
        if selected_profile is None:
            choices = ", ".join(sorted(item.coordinate for item in candidates))
            raise ValueError(
                f"Ambiguous venue {query!r}. Use one of these exact coordinates: {choices}."
            )

    if selected_profile is not None:
        profile = selected_profile
        warnings.warn(
            f"Unpinned profile {query!r} resolved to {profile.coordinate!r}. "
            "Pin the exact coordinate for reproducible output.",
            VenueResolutionWarning,
            stacklevel=2,
        )
        if profile.id == "elsevier-generic" and normalized != normalize_venue_name(profile.id):
            warnings.warn(
                "'elsevier' resolved to the generic artwork profile; the target journal may "
                "override these rules.",
                VenueResolutionWarning,
                stacklevel=2,
            )
        if profile.kind is not VenueKind.CONFERENCE:
            verified_on = date.fromisoformat(profile.verified_on)
            if date.today() - verified_on > timedelta(days=365):
                warnings.warn(
                    f"Profile {profile.coordinate!r} was last verified on "
                    f"{profile.verified_on}; re-check its official sources before submission.",
                    VenueResolutionWarning,
                    stacklevel=2,
                )
        return profile

    lookup: dict[str, str] = {}
    for profile in profiles:
        for candidate in (profile.id, profile.name, *profile.aliases):
            lookup[normalize_venue_name(candidate)] = profile.coordinate
    matches = difflib.get_close_matches(normalized, lookup, n=3, cutoff=0.45)
    hint = ""
    if matches:
        hint = " Did you mean: " + ", ".join(dict.fromkeys(lookup[item] for item in matches)) + "?"
    raise ValueError(f"Unknown venue {query!r}.{hint}")


def resolve_venue(query: str | VenueProfile) -> VenueProfile:
    """Compatibility alias for :func:`resolve_profile`."""

    return resolve_profile(query)


def clear_profile_cache() -> None:
    """Clear profile and schema caches (primarily useful for installation tests)."""

    _load_profiles.cache_clear()
    _load_profile_schema.cache_clear()
