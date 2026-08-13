from __future__ import annotations

import json
from dataclasses import replace
from importlib.resources import files

import pytest

import researchplot as rp
from researchplot.composition import compose_profile, resolve_composition
from researchplot.governance import validate_profile_governance
from researchplot.models import (
    AllExpression,
    ConstraintOperator,
    NotExpression,
    ProbeExpression,
    ProfileCoordinate,
    ProfileStatus,
    RuleConstraint,
    RulePhase,
)
from researchplot.probes import validate_probe_constraint
from researchplot.remote_registry import RegistryCapabilityError, RegistryClient
from researchplot.rule_expressions import evaluate_expression


def _v2_nature_payload() -> dict[str, object]:
    payload = json.loads(
        files("researchplot.profiles").joinpath("nature.json").read_text(encoding="utf-8")
    )
    payload["schema_version"] = 2
    for key in ("namespace", "status", "license", "maintainers", "extends", "governance"):
        payload.pop(key)
    for source in payload["sources"]:
        for key in ("kind", "publisher", "archive_url", "content_sha256"):
            source.pop(key)
    for rule in payload["rules"]:
        for key in ("expression", "supersedes", "rationale"):
            rule.pop(key)
    return payload


def test_schema_v2_translation_preserves_rules_and_marks_provenance() -> None:
    translated = rp.translate_v2_profile(_v2_nature_payload())
    assert translated["schema_version"] == 3
    assert translated["status"] == "translated"
    profile = rp.validate_profile_data(_v2_nature_payload())
    assert profile.schema_version == 3
    assert profile.status is ProfileStatus.TRANSLATED
    assert profile.width_mm("single") == pytest.approx(89.0)


def test_profile_coordinates_support_namespace_and_content_pin() -> None:
    digest = "a" * 64
    coordinate = ProfileCoordinate.parse(f"lab.example/custom-journal@2026.08.1#sha256:{digest}")
    assert coordinate.namespace == "lab.example"
    assert coordinate.digest == digest
    assert str(coordinate) == f"lab.example/custom-journal@2026.08.1#sha256:{digest}"
    nature = rp.resolve_profile("nature@2026.08.0")
    assert rp.resolve_profile(nature.pinned_coordinate) == nature
    with pytest.raises(ValueError, match="Digest mismatch"):
        rp.resolve_profile(f"nature@2026.08.0#sha256:{'0' * 64}")


def test_units_and_three_valued_expressions() -> None:
    width = ProbeExpression(
        "artifact.width_mm",
        RuleConstraint(ConstraintOperator.APPROX, 89.0, "mm", 0.5),
    )
    title = ProbeExpression("figure.has_title", RuleConstraint(ConstraintOperator.PROHIBITED, None))
    expression = AllExpression((width, NotExpression(NotExpression(title))))
    assert evaluate_expression(expression, {"artifact.width_mm": rp.Quantity(8.9, "cm")}) is None
    assert (
        evaluate_expression(
            expression,
            {"artifact.width_mm": rp.Quantity(3.5039, "in"), "figure.has_title": False},
        )
        is True
    )
    assert rp.convert_value(1, "in", "mm") == pytest.approx(25.4)
    assert rp.convert_value(72, "pt", "mm") == pytest.approx(25.4)
    with pytest.raises(ValueError, match="Cannot convert"):
        rp.convert_value(1, "px", "mm")


def test_expression_patterns_quantifiers_and_aggregates_are_bounded() -> None:
    pattern = rp.RuleConstraint(rp.ConstraintOperator.PATTERN, r"[a-z][a-z0-9-]+")
    exists = rp.RuleConstraint(rp.ConstraintOperator.EXISTS, True)
    assert rp.evaluate_constraint("figure-1", pattern)
    assert rp.evaluate_constraint(False, exists)

    all_names = rp.QuantifierExpression("all", "font.families.effective", pattern)
    any_name = rp.QuantifierExpression(
        "any",
        "font.families.effective",
        rp.RuleConstraint(rp.ConstraintOperator.EQ, "dejavu-sans"),
    )
    count = rp.AggregateExpression(
        "count",
        "font.families.effective",
        rp.RuleConstraint(rp.ConstraintOperator.GTE, 2),
    )
    values = {"font.families.effective": ("dejavu-sans", "arial")}
    assert rp.evaluate_expression(all_names, values) is True
    assert rp.evaluate_expression(any_name, values) is True
    assert rp.evaluate_expression(count, values) is True

    with pytest.raises(ValueError, match="256 pattern"):
        rp.evaluate_constraint("text", rp.RuleConstraint(rp.ConstraintOperator.PATTERN, "x" * 257))


def test_probe_catalog_rejects_bad_units_and_phases() -> None:
    with pytest.raises(ValueError, match="requires length"):
        validate_probe_constraint(
            "artifact.width_mm",
            RuleConstraint(ConstraintOperator.EQ, 89.0, "dpi"),
            (RulePhase.FILE,),
            label="width",
        )
    with pytest.raises(ValueError, match="cannot evaluate"):
        validate_probe_constraint(
            "metadata.alt_text.present",
            RuleConstraint(ConstraintOperator.EQ, True),
            (RulePhase.LIVE,),
            label="alt",
        )


def test_composition_requires_explicit_supersedes_and_is_deterministic() -> None:
    base = rp.resolve_profile("nature@2026.08.0")
    inherited = base.get_rule("figure.width.single")
    assert inherited is not None
    child = replace(
        base,
        id="nature-derived",
        aliases=(),
        extends=(base.coordinate,),
        rules=(replace(inherited, constraint=replace(inherited.constraint, value=90.0)),),
        digest="b" * 64,
        document_digest="b" * 64,
    )
    with pytest.raises(ValueError, match="silently overrides"):
        compose_profile(base, child)
    explicit = replace(child, rules=(replace(child.rules[0], supersedes=(inherited.id,)),))
    composed = compose_profile(base, explicit)
    assert composed.width_mm("single") == pytest.approx(90.0)
    assert len(composed.digest) == 64
    assert resolve_composition((base, explicit))[1].digest == composed.digest


def test_governance_requires_verified_maintainers() -> None:
    profile = rp.resolve_profile("acs-generic@2026.08.0")
    report = validate_profile_governance(replace(profile, maintainers=()))
    assert not report.passed
    assert any(issue.code == "governance.maintainers" for issue in report.errors)


def test_profile_lock_enforces_content_digest(tmp_path) -> None:
    profile = rp.resolve_profile("colm-2026@2026.08.0")
    path = rp.write_profile_lock(profile, tmp_path / "researchplot.lock.json")
    lock = rp.load_profile_lock(path)
    assert rp.resolve_locked_profile(lock) == profile
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["digest"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(rp.ProfileLockError, match="digest"):
        rp.resolve_locked_profile(path)


def test_remote_registry_is_opt_in_and_never_falls_back_unsigned(tmp_path) -> None:
    client = RegistryClient(
        "https://profiles.example.invalid/",
        trusted_root=tmp_path / "missing-root.json",
        cache_dir=tmp_path / "cache",
    )
    diagnostic = client.diagnostic()
    assert not diagnostic.available
    assert diagnostic.code in {"tuf-missing", "trusted-root-missing"}
    with pytest.raises(RegistryCapabilityError):
        client.refresh()


def test_remote_registry_passes_explicit_trust_bootstrap(tmp_path, monkeypatch) -> None:
    pytest.importorskip("tuf")
    trusted = tmp_path / "root.json"
    trusted.write_bytes(b'{"signed":"fixture"}')
    captured: dict[str, object] = {}

    class FakeUpdater:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    import tuf.ngclient

    monkeypatch.setattr(tuf.ngclient, "Updater", FakeUpdater)
    client = RegistryClient(
        "https://profiles.example.invalid/",
        trusted_root=trusted,
        cache_dir=tmp_path / "cache",
    )

    client._updater()

    assert captured["bootstrap"] == trusted.read_bytes()
    assert captured["metadata_base_url"] == "https://profiles.example.invalid/metadata/"


def test_every_bundled_profile_conforms_to_bundled_json_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = rp.profile_schema()
    root = files("researchplot.profiles")
    for resource in root.iterdir():
        if resource.name.endswith(".json"):
            jsonschema.validate(json.loads(resource.read_text(encoding="utf-8")), schema)


def test_additional_launch_profiles_are_verified_and_source_backed() -> None:
    expected_sources = {
        "aaai-2026": "aaai.org",
        "aistats-2026": "aistats.org",
        "cell-graphical-abstract": "cell.com",
        "eccv-2026": "eccv.ecva.net",
        "iclr-2026": "iclr.cc",
        "jacs": "acs.org",
        "jmlr": "jmlr.org",
        "physical-review-letters": "aps.org",
        "tmlr": "jmlr.org",
    }
    for profile_id, official_domain in expected_sources.items():
        profile = rp.resolve_profile(f"{profile_id}@2026.08.0")
        assert profile.status is ProfileStatus.VERIFIED
        assert profile.rules
        assert profile.sources
        assert any(official_domain in source.url for source in profile.sources)


def test_jacs_explicitly_composes_acs_and_scopes_toc_rules() -> None:
    profile = rp.resolve_profile("jacs@2026.08.0")
    assert profile.extends == ("acs-generic@2026.08.0",)
    assert profile.width_mm("single") == pytest.approx(82.55)
    assert "acs" not in profile.aliases
    toc = profile.get_rule("figure.width.toc_maximum")
    assert toc is not None
    assert toc.constraint.operator is ConstraintOperator.LTE
    assert toc.applies_to.roles == (rp.FigureRole.GRAPHICAL_ABSTRACT,)


def test_partial_profiles_do_not_invent_geometry() -> None:
    for coordinate in ("aaai-2026@2026.08.0", "physical-review-letters@2026.08.0"):
        profile = rp.resolve_profile(coordinate)
        assert profile.default_width is None
        assert profile.width_options == ()
        with pytest.raises(ValueError, match="does not specify physical figure widths"):
            profile.width_mm()


def test_cell_graphical_abstract_profile_is_role_limited() -> None:
    profile = rp.resolve_profile("cell-graphical-abstract@2026.08.0")
    assert profile.width_mm() == pytest.approx(139.7)
    width = profile.get_rule("figure.width.square")
    assert width is not None
    assert width.applies_to.roles == (rp.FigureRole.GRAPHICAL_ABSTRACT,)
    assert "ordinary Cell Press manuscript figures" in " ".join(profile.caveats)
