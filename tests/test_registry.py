from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from importlib.resources import files

import pytest

import researchplot as rp
from researchplot import registry

EXPECTED_WIDTHS = {
    "ieee-journal": {"single": 88.9, "double": 181.864},
    "nature": {"single": 89.0, "double": 183.0},
    "elsevier-generic": {
        "minimal": 30.0,
        "single": 90.0,
        "one-and-half": 140.0,
        "double": 190.0,
    },
    "neurips-2026": {"full": 139.7},
    "icml-2026": {"single": 82.55, "double": 171.45},
    "cvpr-2026": {"single": 83.34375, "double": 174.625},
    "acl-2026": {"single": 77.0, "double": 160.0},
}


def test_catalog_has_only_verified_initial_profiles() -> None:
    profiles = rp.list_venues()
    assert {profile.id for profile in profiles} == set(EXPECTED_WIDTHS)
    for profile in profiles:
        assert date.fromisoformat(profile.verified_on)
        assert profile.sources
        assert profile.rules
        for source in profile.sources:
            assert source.url.startswith("https://")
            assert date.fromisoformat(source.verified_on)
        for rule in profile.rules:
            assert rule.source_ids
            assert rule.level in set(rp.RuleLevel)


@pytest.mark.parametrize(
    ("profile_id", "width", "expected"),
    [
        (profile_id, width, expected)
        for profile_id, widths in EXPECTED_WIDTHS.items()
        for width, expected in widths.items()
    ],
)
def test_exact_widths(profile_id: str, width: str, expected: float) -> None:
    assert rp.resolve_venue(profile_id).width_mm(width) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("  CVPR--2026 ", "cvpr-2026"),
        ("I.C.M.L. 2026", "icml-2026"),
        ("IEEE Transactions", "ieee-journal"),
        ("Nature Journal", "nature"),
    ],
)
def test_alias_normalization(query: str, expected: str) -> None:
    assert rp.resolve_venue(query).id == expected


def test_bare_conference_resolves_latest_and_reports_id() -> None:
    with pytest.warns(rp.VenueResolutionWarning, match="cvpr-2026"):
        assert rp.resolve_venue("cvpr").id == "cvpr-2026"


def test_elsevier_alias_warns_about_generic_scope() -> None:
    with pytest.warns(rp.VenueResolutionWarning, match="generic"):
        assert rp.resolve_venue("elsevier").id == "elsevier-generic"


def test_unknown_and_legacy_names_are_actionable() -> None:
    with pytest.raises(ValueError, match="Did you mean.*cvpr-2026"):
        rp.resolve_venue("cvppr")
    with pytest.raises(ValueError, match="unverified legacy"):
        rp.resolve_venue("science")


def test_ambiguous_alias_requires_exact_id(monkeypatch: pytest.MonkeyPatch) -> None:
    first = rp.resolve_venue("nature")
    second = replace(first, id="nature-methods", name="Nature Methods")
    monkeypatch.setattr(registry, "_load_profiles", lambda: (first, second))
    with pytest.raises(ValueError, match="Ambiguous venue"):
        registry.resolve_venue("nature journal")


def test_list_and_search_filters() -> None:
    conferences = rp.list_venues(kind="conference", year=2026)
    assert {profile.id for profile in conferences} == {
        "acl-2026",
        "cvpr-2026",
        "icml-2026",
        "neurips-2026",
    }
    assert [profile.id for profile in rp.search_venues("electrical")] == ["ieee-journal"]


def test_models_are_frozen_and_serializable() -> None:
    profile = rp.resolve_venue("nature")
    with pytest.raises(AttributeError):
        profile.name = "Changed"  # type: ignore[misc]
    payload = profile.to_dict()
    assert payload["width_options"] == ["single", "double"]
    assert payload["rules"][0]["source_ids"]


def test_profile_schema_rejects_invalid_width_and_missing_provenance() -> None:
    resource = files("researchplot.profiles").joinpath("nature.json")
    payload = json.loads(resource.read_text(encoding="utf-8"))
    payload["rules"][0]["value"] = -1
    with pytest.raises(ValueError, match="positive numeric millimetre"):
        registry._parse_profile(payload, "nature.json")

    payload = json.loads(resource.read_text(encoding="utf-8"))
    payload["rules"][0]["source_ids"] = []
    with pytest.raises(ValueError, match="cite at least one source"):
        registry._parse_profile(payload, "nature.json")
