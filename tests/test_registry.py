from __future__ import annotations

import json
import warnings
from dataclasses import replace
from datetime import date
from importlib.metadata import version
from importlib.resources import files

import pytest

import researchplot as rp
from researchplot import registry
from researchplot.models import (
    ConstraintOperator,
    FigureRole,
    RuleLevel,
    RulePhase,
    VerificationMode,
)

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
    "plos-biology": {"text-column": 132.0, "full": 190.5},
}

EXPECTED_PROFILE_IDS = set(EXPECTED_WIDTHS) | {"acm-acmart"}


def test_version_matches_distribution_metadata() -> None:
    assert rp.__version__ == version("researchplot-venues")


def test_catalog_contains_source_backed_schema_v2_profiles() -> None:
    profiles = rp.list_profiles()
    assert {profile.id for profile in profiles} == EXPECTED_PROFILE_IDS
    for profile in profiles:
        assert profile.schema_version == 2
        assert profile.coordinate == f"{profile.id}@2026.08.0"
        assert len(profile.digest) == 64
        assert date.fromisoformat(profile.verified_on)
        assert date.fromisoformat(profile.effective_date)
        assert profile.sources
        assert profile.rules
        for source in profile.sources:
            assert source.url.startswith("https://")
            assert source.locator
            assert date.fromisoformat(source.retrieved_on)
            assert date.fromisoformat(source.verified_on)
        for rule in profile.rules:
            assert rule.source_ids
            assert rule.level in set(RuleLevel)
            assert rule.probe
            assert rule.phases
            assert rule.verification in set(VerificationMode)


@pytest.mark.parametrize(
    ("profile_id", "width", "expected"),
    [
        (profile_id, width, expected)
        for profile_id, widths in EXPECTED_WIDTHS.items()
        for width, expected in widths.items()
    ],
)
def test_exact_widths(profile_id: str, width: str, expected: float) -> None:
    coordinate = f"{profile_id}@2026.08.0"
    assert registry.resolve_profile(coordinate).width_mm(width) == pytest.approx(expected)


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
    with pytest.warns(rp.VenueResolutionWarning, match="Unpinned profile"):
        assert rp.resolve_profile(query).id == expected


def test_bare_conference_resolves_latest_and_reports_id() -> None:
    with pytest.warns(rp.VenueResolutionWarning, match="cvpr-2026"):
        assert rp.resolve_profile("cvpr").id == "cvpr-2026"


def test_elsevier_alias_warns_about_generic_scope() -> None:
    with pytest.warns(rp.VenueResolutionWarning, match="generic"):
        assert rp.resolve_profile("elsevier").id == "elsevier-generic"


def test_unknown_and_legacy_names_are_actionable() -> None:
    with pytest.raises(ValueError, match="Did you mean.*cvpr-2026"):
        rp.resolve_profile("cvppr")
    with pytest.raises(ValueError, match="unverified legacy"):
        rp.resolve_profile("science")


def test_ambiguous_alias_requires_exact_id(monkeypatch: pytest.MonkeyPatch) -> None:
    first = registry.resolve_profile("nature@2026.08.0")
    second = replace(first, id="nature-methods", name="Nature Methods")
    monkeypatch.setattr(registry, "_load_profiles", lambda: (first, second))
    with pytest.raises(ValueError, match="Ambiguous venue"):
        registry.resolve_venue("nature journal")


def test_list_and_search_filters() -> None:
    conferences = rp.list_profiles(kind="conference", year=2026)
    assert {profile.id for profile in conferences} == {
        "acl-2026",
        "cvpr-2026",
        "icml-2026",
        "neurips-2026",
    }
    assert [profile.id for profile in rp.search_profiles("electrical")] == ["ieee-journal"]


def test_models_are_frozen_and_serializable() -> None:
    profile = registry.resolve_profile("nature@2026.08.0")
    with pytest.raises(AttributeError):
        profile.name = "Changed"  # type: ignore[misc]
    payload = profile.to_dict()
    assert payload["coordinate"] == "nature@2026.08.0"
    assert payload["digest"] == profile.digest
    assert payload["width_options"] == ["single", "double"]
    assert payload["rules"][0]["source_ids"]
    assert payload["rules"][0]["constraint"]["operator"] == "approx"


def test_profile_schema_rejects_invalid_width_and_missing_provenance() -> None:
    resource = files("researchplot.profiles").joinpath("nature.json")
    payload = json.loads(resource.read_text(encoding="utf-8"))
    payload["rules"][0]["constraint"]["value"] = -1
    with pytest.raises(ValueError, match="positive numeric millimetre"):
        registry._parse_profile(payload, "nature.json")

    payload = json.loads(resource.read_text(encoding="utf-8"))
    payload["rules"][0]["source_ids"] = []
    with pytest.raises(ValueError, match="cite at least one source"):
        registry._parse_profile(payload, "nature.json")


def test_profile_schema_rejects_non_finite_and_reversed_numeric_constraints() -> None:
    nature = json.loads(
        files("researchplot.profiles").joinpath("nature.json").read_text(encoding="utf-8")
    )
    nature["rules"][0]["constraint"]["value"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        registry.validate_profile_data(nature)

    nature = json.loads(
        files("researchplot.profiles").joinpath("nature.json").read_text(encoding="utf-8")
    )
    nature["rules"][0]["constraint"]["tolerance"] = float("inf")
    with pytest.raises(ValueError, match="non-negative number"):
        registry.validate_profile_data(nature)

    plos = json.loads(
        files("researchplot.profiles").joinpath("plos-biology.json").read_text(encoding="utf-8")
    )
    between = next(rule for rule in plos["rules"] if rule["id"] == "raster.resolution.range")
    between["constraint"]["value"] = [600, 300]
    with pytest.raises(ValueError, match="lower bound"):
        registry.validate_profile_data(plos)


def test_pinned_coordinates_are_reproducible_and_unpinned_queries_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", rp.VenueResolutionWarning)
        pinned = registry.resolve_profile("nature@2026.08.0")
    assert pinned.coordinate == "nature@2026.08.0"

    with pytest.warns(rp.VenueResolutionWarning, match="nature@2026.08.0"):
        assert registry.resolve_profile("nature").coordinate == "nature@2026.08.0"
    with pytest.raises(ValueError, match="Available coordinates.*nature@2026.08.0"):
        registry.resolve_profile("nature@2026.09.0")


def test_profile_schema_is_bundled_and_offline() -> None:
    schema = registry.profile_schema()
    assert schema["$schema"].endswith("2020-12/schema")
    assert schema["properties"]["schema_version"] == {"const": 2}
    schema["title"] = "mutated copy"
    assert registry.profile_schema()["title"] == "ResearchPlot venue profile"


def test_rule_constraints_applicability_and_compatibility_views() -> None:
    profile = registry.resolve_profile("nature@2026.08.0")
    width = profile.get_rule("figure.width.single")
    assert width is not None
    assert width.constraint.operator is ConstraintOperator.APPROX
    assert width.constraint.tolerance == pytest.approx(0.5)
    assert width.value == width.constraint.value == 89.0
    assert width.unit == width.constraint.unit == "mm"
    assert width.applies_to.widths == ("single",)
    assert width.applies_to.matches(role=FigureRole.MAIN, width="single")
    assert RulePhase.LIVE in width.phases
    assert RulePhase.FILE in width.phases


def test_acm_profile_is_bundle_only_and_does_not_invent_widths() -> None:
    profile = registry.resolve_profile("acm-acmart@2026.08.0")
    assert profile.default_width is None
    assert profile.width_options == ()
    with pytest.raises(ValueError, match="does not specify physical figure widths"):
        profile.width_mm()
    rule = profile.get_rule("metadata.alt_text.present")
    assert rule is not None
    assert rule.level is RuleLevel.REQUIRED
    assert rule.phases == (RulePhase.BUNDLE,)
    assert rule.probe == "metadata.alt_text.present"


def test_plos_profile_encodes_file_constraints_without_invented_defaults() -> None:
    profile = registry.resolve_profile("plos-biology@2026.08.0")
    assert profile.width_mm() == pytest.approx(132.0)
    assert profile.get_rule("figure.width.min").value == 66.8  # type: ignore[union-attr]
    assert profile.get_rule("figure.width.max").value == 190.5  # type: ignore[union-attr]
    dpi = profile.get_rule("raster.resolution.range")
    assert dpi is not None
    assert dpi.constraint.operator is ConstraintOperator.BETWEEN
    assert dpi.value == (300.0, 600.0)
    formats = profile.get_rule("export.formats.main")
    assert formats is not None and formats.value == ("tiff", "eps")


def test_font_rules_use_effective_families_and_real_embedding_observations() -> None:
    for coordinate in (
        "nature@2026.08.0",
        "acl-2026@2026.08.0",
        "cvpr-2026@2026.08.0",
        "icml-2026@2026.08.0",
        "neurips-2026@2026.08.0",
        "elsevier-generic@2026.08.0",
        "plos-biology@2026.08.0",
    ):
        family = registry.resolve_profile(coordinate).get_rule("font.family")
        assert family is not None
        assert family.probe == "font.families.effective"

    nature_embedding = registry.resolve_profile("nature@2026.08.0").get_rule(
        "font.pdf.embedding.required"
    )
    acl_embedding = registry.resolve_profile("acl-2026@2026.08.0").get_rule(
        "font.pdf.embedding.required"
    )
    cvpr_embedding = registry.resolve_profile("cvpr-2026@2026.08.0").get_rule(
        "font.pdf.embedding.recommended"
    )
    assert nature_embedding is not None and nature_embedding.probe == "pdf.unembedded_font_count"
    assert nature_embedding.level is RuleLevel.REQUIRED
    assert acl_embedding is not None and acl_embedding.probe == "pdf.unembedded_font_count"
    assert acl_embedding.level is RuleLevel.REQUIRED
    assert cvpr_embedding is not None and cvpr_embedding.probe == "pdf.unembedded_font_count"
    assert cvpr_embedding.level is RuleLevel.RECOMMENDED
    neurips_truetype = registry.resolve_profile("neurips-2026@2026.08.0").get_rule(
        "font.pdf.truetype_embedding.required"
    )
    assert neurips_truetype is not None
    assert neurips_truetype.probe == "pdf.unembedded_truetype_font_count"
    assert neurips_truetype.level is RuleLevel.REQUIRED


def test_load_profile_validates_local_json(tmp_path) -> None:
    source = files("researchplot.profiles").joinpath("nature.json")
    profile_path = tmp_path / "custom-name.json"
    profile_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    loaded = registry.load_profile(profile_path)
    assert loaded.coordinate == "nature@2026.08.0"
    assert loaded.digest == registry.resolve_profile("nature@2026.08.0").digest

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        registry.load_profile(invalid)

    non_finite = tmp_path / "non-finite.json"
    non_finite.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="not finite"):
        registry.load_profile(non_finite)

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * 1_000_001)
    with pytest.raises(ValueError, match="size limit"):
        registry.load_profile(oversized)


def test_installed_profile_pack_entry_point_is_discovered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.loads(
        files("researchplot.profiles").joinpath("nature.json").read_text(encoding="utf-8")
    )
    payload.update(
        {
            "id": "external-journal",
            "name": "External Journal",
            "aliases": ["external venue"],
        }
    )

    class FakeEntryPoint:
        name = "external"
        module = "example_profiles"
        attr = "profiles"

        def load(self) -> object:
            return payload

    class FakeEntryPoints:
        def select(self, *, group: str) -> tuple[FakeEntryPoint, ...]:
            assert group == "researchplot.profiles"
            return (FakeEntryPoint(),)

    monkeypatch.setattr(registry, "entry_points", lambda: FakeEntryPoints())
    registry.clear_profile_cache()
    try:
        installed = registry.list_profiles()
        assert len(installed) == 10
        assert registry.resolve_profile("external-journal@2026.08.0").name == "External Journal"
    finally:
        registry.clear_profile_cache()


def test_broken_profile_pack_does_not_hide_bundled_profiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenEntryPoint:
        name = "broken"
        module = "broken_profiles"
        attr = "profiles"

        def load(self) -> object:
            raise RuntimeError("broken on import")

    class FakeEntryPoints:
        def select(self, *, group: str) -> tuple[BrokenEntryPoint, ...]:
            assert group == "researchplot.profiles"
            return (BrokenEntryPoint(),)

    monkeypatch.setattr(registry, "entry_points", lambda: FakeEntryPoints())
    registry.clear_profile_cache()
    try:
        with pytest.warns(RuntimeWarning, match="Ignoring ResearchPlot profile pack"):
            installed = registry.list_profiles()
        assert {profile.id for profile in installed} == EXPECTED_PROFILE_IDS
    finally:
        registry.clear_profile_cache()
