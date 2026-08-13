"""Deterministic, explicit composition for schema-v3 profiles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

from .models import SourceRef, VenueProfile, VenueRule


def _merge_sources(
    base: tuple[SourceRef, ...], child: tuple[SourceRef, ...]
) -> tuple[SourceRef, ...]:
    merged = {source.id: source for source in base}
    for source in child:
        previous = merged.get(source.id)
        if previous is not None and previous != source:
            raise ValueError(
                f"Source {source.id!r} changes inherited evidence; use a new source id."
            )
        merged[source.id] = source
    return tuple(merged[key] for key in sorted(merged))


def _merge_rules(
    base: tuple[VenueRule, ...], child: tuple[VenueRule, ...]
) -> tuple[VenueRule, ...]:
    merged = {rule.id: rule for rule in base}
    for rule in child:
        previous = merged.get(rule.id)
        if previous is not None and previous.id not in rule.supersedes:
            raise ValueError(
                f"Rule {rule.id!r} silently overrides inherited guidance; list it in supersedes."
            )
        merged[rule.id] = rule
    return tuple(merged[key] for key in sorted(merged))


def compose_profile(base: VenueProfile, child: VenueProfile) -> VenueProfile:
    """Compose one exact parent into a child and calculate a resolved digest."""

    if base.coordinate not in child.extends:
        raise ValueError(f"Child {child.coordinate!r} does not declare parent {base.coordinate!r}.")
    sources = _merge_sources(base.sources, child.sources)
    rules = _merge_rules(base.rules, child.rules)
    caveats = tuple(dict.fromkeys((*base.caveats, *child.caveats)))
    digest_payload = {
        "base": base.digest,
        "child": child.document_digest or child.digest,
        "rules": [rule.to_dict() for rule in rules],
        "sources": [source.to_dict() for source in sources],
    }
    digest = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    result = replace(
        child,
        aliases=child.aliases,
        caveats=caveats,
        sources=sources,
        rules=rules,
        digest=digest,
    )
    if result.default_width is not None and result.default_width not in result.width_options:
        raise ValueError(
            f"Composed profile {result.coordinate!r} has undefined default width "
            f"{result.default_width!r}."
        )
    for rule in result.rules:
        unknown_widths = set(rule.applies_to.widths) - set(result.width_options)
        if unknown_widths:
            raise ValueError(
                f"Composed rule {rule.id!r} applies to undefined widths: "
                f"{', '.join(sorted(unknown_widths))}."
            )
    return result


def resolve_composition(profiles: tuple[VenueProfile, ...]) -> tuple[VenueProfile, ...]:
    """Resolve a profile graph, rejecting cycles and missing exact parents."""

    by_coordinate = {profile.coordinate: profile for profile in profiles}
    resolved: dict[str, VenueProfile] = {}
    active: set[str] = set()

    def visit(profile: VenueProfile) -> VenueProfile:
        if profile.coordinate in resolved:
            return resolved[profile.coordinate]
        if profile.coordinate in active:
            raise ValueError(f"Profile composition cycle includes {profile.coordinate!r}.")
        active.add(profile.coordinate)
        result = profile
        for parent_coordinate in profile.extends:
            parent = by_coordinate.get(parent_coordinate)
            if parent is None:
                raise ValueError(
                    f"Profile {profile.coordinate!r} extends missing exact coordinate {parent_coordinate!r}."
                )
            result = compose_profile(visit(parent), result)
        active.remove(profile.coordinate)
        resolved[profile.coordinate] = result
        return result

    return tuple(visit(profile) for profile in profiles)
