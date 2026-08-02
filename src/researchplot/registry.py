"""Offline venue profile loading and name resolution."""

from __future__ import annotations

import difflib
import json
import re
import warnings
from datetime import date, timedelta
from functools import lru_cache
from importlib.resources import files
from typing import Any, cast

from .models import (
    RuleLevel,
    RuleValue,
    SourceRef,
    VenueKind,
    VenueProfile,
    VenueResolutionWarning,
    VenueRule,
)

_PROFILE_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_LEGACY_ONLY = {"cell", "pnas", "science", "springer"}


def normalize_venue_name(value: str) -> str:
    """Normalize punctuation and spacing for venue lookup."""

    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _expect_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return cast(dict[str, Any], value)


def _expect_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    return value


def _expect_strings(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be an array of strings.")
    return tuple(cast(list[str], value))


def _freeze_rule_value(value: object, label: str) -> RuleValue:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, list):
        if all(isinstance(item, str) for item in value):
            return tuple(cast(list[str], value))
        if all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
            return tuple(float(item) for item in cast(list[float], value))
    raise ValueError(f"{label} contains an unsupported rule value.")


def _validate_date(value: str, label: str) -> str:
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must use YYYY-MM-DD format.") from exc
    return value


def _parse_profile(payload: object, filename: str) -> VenueProfile:
    data = _expect_mapping(payload, filename)
    if data.get("schema_version") != 1:
        raise ValueError(f"{filename} uses an unsupported profile schema.")

    profile_id = _expect_string(data.get("id"), f"{filename}.id")
    if not _PROFILE_ID.fullmatch(profile_id):
        raise ValueError(f"{filename}.id must be a lowercase kebab-case identifier.")

    raw_year = data.get("year")
    if raw_year is not None and (not isinstance(raw_year, int) or isinstance(raw_year, bool)):
        raise ValueError(f"{filename}.year must be an integer or null.")

    sources: list[SourceRef] = []
    source_ids: set[str] = set()
    raw_sources = data.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError(f"{filename}.sources must be a non-empty array.")
    for index, raw_source in enumerate(raw_sources):
        source = _expect_mapping(raw_source, f"{filename}.sources[{index}]")
        source_id = _expect_string(source.get("id"), f"{filename}.sources[{index}].id")
        if source_id in source_ids:
            raise ValueError(f"{filename} repeats source id {source_id!r}.")
        source_ids.add(source_id)
        sources.append(
            SourceRef(
                id=source_id,
                title=_expect_string(source.get("title"), f"{filename}.sources[{index}].title"),
                url=_expect_string(source.get("url"), f"{filename}.sources[{index}].url"),
                verified_on=_validate_date(
                    _expect_string(
                        source.get("verified_on"),
                        f"{filename}.sources[{index}].verified_on",
                    ),
                    f"{filename}.sources[{index}].verified_on",
                ),
            )
        )
        if not sources[-1].url.startswith("https://"):
            raise ValueError(f"{filename}.sources[{index}].url must use HTTPS.")

    rules: list[VenueRule] = []
    rule_ids: set[str] = set()
    raw_rules = data.get("rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise ValueError(f"{filename}.rules must be a non-empty array.")
    for index, raw_rule in enumerate(raw_rules):
        rule = _expect_mapping(raw_rule, f"{filename}.rules[{index}]")
        rule_id = _expect_string(rule.get("id"), f"{filename}.rules[{index}].id")
        if rule_id in rule_ids:
            raise ValueError(f"{filename} repeats rule id {rule_id!r}.")
        rule_ids.add(rule_id)
        linked_sources = _expect_strings(
            rule.get("source_ids"), f"{filename}.rules[{index}].source_ids"
        )
        if not linked_sources:
            raise ValueError(f"{filename}.{rule_id} must cite at least one source.")
        missing_sources = set(linked_sources) - source_ids
        if missing_sources:
            raise ValueError(
                f"{filename}.{rule_id} references unknown sources: "
                f"{', '.join(sorted(missing_sources))}."
            )
        try:
            level = RuleLevel(_expect_string(rule.get("level"), "rule level"))
        except ValueError as exc:
            raise ValueError(f"{filename}.{rule_id} has an invalid rule level.") from exc
        unit = rule.get("unit")
        if unit is not None and not isinstance(unit, str):
            raise ValueError(f"{filename}.{rule_id}.unit must be a string or null.")
        rules.append(
            VenueRule(
                id=rule_id,
                value=_freeze_rule_value(rule.get("value"), f"{filename}.{rule_id}"),
                unit=unit,
                level=level,
                source_ids=linked_sources,
                description=_expect_string(
                    rule.get("description"), f"{filename}.{rule_id}.description"
                ),
            )
        )

    filename_stem = filename.removesuffix(".json")
    if profile_id != filename_stem:
        raise ValueError(f"{filename}.id must match its filename.")
    for profile_rule in rules:
        if profile_rule.id.startswith("figure.width."):
            if (
                not isinstance(profile_rule.value, (int, float))
                or isinstance(profile_rule.value, bool)
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
            or float(profile_rule.value) <= 0
            or profile_rule.unit != "dpi"
        ):
            raise ValueError(f"{filename}.{profile_rule.id} must be a positive DPI value.")

    default_width = _expect_string(data.get("default_width"), f"{filename}.default_width")
    if f"figure.width.{default_width}" not in rule_ids:
        raise ValueError(f"{filename} default width {default_width!r} is not defined.")
    if not any(rule_id.startswith("export.formats.") for rule_id in rule_ids):
        raise ValueError(f"{filename} must define at least one export format rule.")

    return VenueProfile(
        id=profile_id,
        name=_expect_string(data.get("name"), f"{filename}.name"),
        kind=VenueKind(_expect_string(data.get("kind"), f"{filename}.kind")),
        year=raw_year,
        aliases=_expect_strings(data.get("aliases", []), f"{filename}.aliases"),
        scope=_expect_string(data.get("scope"), f"{filename}.scope"),
        default_width=default_width,
        verified_on=_validate_date(
            _expect_string(data.get("verified_on"), f"{filename}.verified_on"),
            f"{filename}.verified_on",
        ),
        sources=tuple(sources),
        rules=tuple(rules),
        caveats=_expect_strings(data.get("caveats", []), f"{filename}.caveats"),
    )


@lru_cache(maxsize=1)
def _load_profiles() -> tuple[VenueProfile, ...]:
    profile_root = files("researchplot.profiles")
    loaded = []
    for resource in sorted(profile_root.iterdir(), key=lambda item: item.name):
        if resource.name.endswith(".json"):
            loaded.append(
                _parse_profile(json.loads(resource.read_text(encoding="utf-8")), resource.name)
            )
    if not loaded:
        raise RuntimeError("ResearchPlot was installed without its venue profiles.")
    profile_ids = [profile.id for profile in loaded]
    if len(profile_ids) != len(set(profile_ids)):
        raise ValueError("Venue profile identifiers must be unique.")
    return tuple(loaded)


def list_venues(
    *, kind: VenueKind | str | None = None, year: int | None = None
) -> list[VenueProfile]:
    """Return bundled verified venue profiles, optionally filtered."""

    selected_kind = VenueKind(kind) if isinstance(kind, str) else kind
    return [
        profile
        for profile in _load_profiles()
        if (selected_kind is None or profile.kind is selected_kind)
        and (year is None or profile.year == year)
    ]


def search_venues(query: str) -> list[VenueProfile]:
    """Return profiles whose identifiers, names, or aliases contain ``query``."""

    needle = normalize_venue_name(query)
    if not needle:
        return list(_load_profiles())
    return [
        profile
        for profile in _load_profiles()
        if any(
            needle in normalize_venue_name(candidate)
            for candidate in (profile.id, profile.name, *profile.aliases)
        )
    ]


def resolve_venue(query: str | VenueProfile) -> VenueProfile:
    """Resolve a human venue name without any silent fallback."""

    if isinstance(query, VenueProfile):
        return query
    normalized = normalize_venue_name(query)
    if not normalized:
        raise ValueError("Venue query cannot be empty.")
    if normalized in _LEGACY_ONLY:
        raise ValueError(
            f"{query!r} is available only as an unverified legacy plotting style. "
            "Use researchplot.plots for compatibility; no compliance profile is claimed."
        )

    profiles = _load_profiles()
    exact_ids = [profile for profile in profiles if normalized == normalize_venue_name(profile.id)]
    candidates: list[VenueProfile] = []
    for profile in profiles:
        names = (profile.id, profile.name, *profile.aliases)
        if normalized in {normalize_venue_name(name) for name in names}:
            candidates.append(profile)

    selected_profile: VenueProfile | None
    if exact_ids:
        selected_profile = exact_ids[0]
    elif len(candidates) == 1:
        selected_profile = candidates[0]
    elif candidates and all(item.kind is VenueKind.CONFERENCE for item in candidates):
        newest_year = max(item.year or 0 for item in candidates)
        newest = [item for item in candidates if (item.year or 0) == newest_year]
        if len(newest) != 1:
            choices = ", ".join(sorted(item.id for item in candidates))
            raise ValueError(f"Ambiguous venue {query!r}. Use one of these exact IDs: {choices}.")
        selected_profile = newest[0]
    elif candidates:
        choices = ", ".join(sorted(item.id for item in candidates))
        raise ValueError(f"Ambiguous venue {query!r}. Use one of these exact IDs: {choices}.")
    else:
        selected_profile = None

    if selected_profile is not None:
        profile = selected_profile
        if profile.year is not None and not re.search(r"\b\d{4}\b", query):
            warnings.warn(
                f"{query!r} resolved to {profile.id!r}. Pin the year-specific id for "
                "reproducible output.",
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
                    f"Profile {profile.id!r} was last verified on {profile.verified_on}; "
                    "re-check its official sources before submission.",
                    VenueResolutionWarning,
                    stacklevel=2,
                )
        return profile

    lookup: dict[str, str] = {}
    for profile in _load_profiles():
        for candidate in (profile.id, profile.name, *profile.aliases):
            lookup[normalize_venue_name(candidate)] = profile.id
    matches = difflib.get_close_matches(normalized, lookup, n=3, cutoff=0.45)
    hint = ""
    if matches:
        hint = " Did you mean: " + ", ".join(dict.fromkeys(lookup[item] for item in matches)) + "?"
    raise ValueError(f"Unknown venue {query!r}.{hint}")


def clear_profile_cache() -> None:
    """Clear the bundled profile cache (primarily useful to test installations)."""

    _load_profiles.cache_clear()
