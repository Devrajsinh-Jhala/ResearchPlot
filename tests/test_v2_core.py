from __future__ import annotations

from pathlib import Path

import pytest
from matplotlib import pyplot as plt

import researchplot as rp


def _write_v3_config(path: Path, *, extra: str = "") -> None:
    path.write_text(
        f"""schema_version = 3
profile = "nature@2026.08.0"
policy = "complete"
{extra}

[[figures]]
id = "figure-1"
role = "main"
width = "single"
content = "line-art"
caption = "Measured response."
alt_text = "A curve rises and then falls."
source_data = ["data/figure-1.csv"]

[[figures.deliverables]]
id = "pdf"
format = "pdf"
path = "figures/figure-1.pdf"
required = true
preferred = true
""",
        encoding="utf-8",
    )


def _acm_project(path: Path, *, alt_text: str | None, attest: bool) -> rp.Project:
    figure = rp.FigureSpec(
        id="figure-1",
        content="line-art",
        caption="Measured response.",
        alt_text=alt_text,
        attestations={"metadata.alt_text.distinct_from_caption": "The description adds the trend."}
        if attest
        else {},
        deliverables=(rp.DeliverableSpec("pdf", "pdf", path=path),),
    )
    return rp.Project(
        rp.ProjectSpec(
            profile="acm-acmart@2026.08.0",
            figures=(figure,),
        )
    )


def test_strict_v3_config_loads_relative_paths_and_round_trips(tmp_path: Path) -> None:
    config_path = tmp_path / "researchplot.toml"
    _write_v3_config(config_path)

    spec = rp.ProjectSpec.load(config_path)

    assert spec.schema_version == 3
    assert spec.profile == "nature@2026.08.0"
    assert spec.figures[0].source_data == ((tmp_path / "data/figure-1.csv").resolve(),)
    deliverable = spec.figures[0].deliverables[0]
    assert deliverable.path == (tmp_path / "figures/figure-1.pdf").resolve()
    assert spec.to_dict()["figures"][0]["deliverables"][0]["path"] == (  # type: ignore[index]
        "figures/figure-1.pdf"
    )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ('profile = "nature@2026.08.0"\n', "schema_version"),
        (
            'schema_version = 3\nprofile = "nature@2026.08.0"\nunknown = true\n',
            "unknown fields",
        ),
        ('schema_version = 3\nprofile = "nature"\n', "non-empty array"),
    ],
)
def test_strict_v3_config_rejects_incomplete_or_unknown_content(
    tmp_path: Path, payload: str, message: str
) -> None:
    path = tmp_path / "researchplot.toml"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        rp.ProjectSpec.load(path)


@pytest.mark.parametrize("unsafe", ["../figure.pdf", "nested/../../figure.pdf"])
def test_project_paths_reject_parent_traversal(tmp_path: Path, unsafe: str) -> None:
    path = tmp_path / "researchplot.toml"
    path.write_text(
        f'''schema_version = 3
profile = "nature@2026.08.0"

[[figures]]
id = "figure-1"

[[figures.deliverables]]
id = "pdf"
format = "pdf"
path = "{unsafe}"
''',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="traversal"):
        rp.ProjectSpec.load(path)


def test_project_paths_reject_absolute_paths(tmp_path: Path) -> None:
    path = tmp_path / "researchplot.toml"
    absolute = (tmp_path / "figure.pdf").as_posix()
    path.write_text(
        f'''schema_version = 3
profile = "nature@2026.08.0"

[[figures]]
id = "figure-1"

[[figures.deliverables]]
id = "pdf"
format = "pdf"
path = "{absolute}"
''',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="relative"):
        rp.ProjectSpec.load(path)


def test_v2_report_contract_and_environment_provenance() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    project = rp.Project(
        rp.ProjectSpec(
            profile="nature@2026.08.0",
            figures=(
                rp.FigureSpec(
                    "figure-1",
                    (rp.DeliverableSpec("pdf", "pdf"),),
                    width="single",
                ),
            ),
        )
    )

    report = project.check()
    payload = report.to_dict()

    jsonschema.validate(payload, rp.validation_report_schema())
    assert payload["schema_version"] == 2
    assert len(payload["plan_digest"]) == 64
    assert payload["verdict"] == "indeterminate"
    assert payload["environment_provenance"]["python_version"]
    assert {item["phase"] for item in payload["phase_coverage"]} == {
        "live",
        "file",
        "bundle",
        "manuscript",
    }


def test_typed_observation_rejects_phase_and_unit_misuse() -> None:
    observation = rp.Observation(
        "artifact.width_mm",
        3.5,
        phase="file",
        unit="in",
        producer="fixture",
        supported_phases=(rp.EvidencePhase.FILE,),
        supported_formats=("PDF", "pdf"),
    )
    assert observation.unit == "in"
    assert observation.supported_formats == ("pdf",)
    assert observation.to_dict()["confidence"] == "deterministic"

    with pytest.raises(ValueError, match="does not support phase"):
        rp.Observation(
            "artifact.width_mm",
            89.0,
            phase="live",
            unit="mm",
            supported_phases=(rp.EvidencePhase.FILE,),
        )


def test_typed_attestations_and_waivers_preserve_true_verdict(tmp_path: Path) -> None:
    profile = rp.resolve_profile("acm-acmart@2026.08.0")
    attestation = rp.ManualAttestation(
        "metadata.alt_text.distinct_from_caption",
        reviewer="Devraj Jhala",
        date="2026-08-03",
        rationale="The description states the trend rather than repeating the caption.",
        evidence=("review/figure-1.md",),
    )
    waiver = rp.Waiver(
        "metadata.alt_text.present",
        profile_digest=profile.digest,
        reviewer="Editorial review board",
        reason="The required symbol font has written editor approval.",
        expires_on="2099-01-01",
    )
    figure = rp.FigureSpec(
        "figure-1",
        (rp.DeliverableSpec("pdf", "pdf", path=tmp_path / "figure.pdf"),),
        caption="Measured response.",
        alt_text="The response rises monotonically with input.",
        attestations={attestation.rule_id: attestation},
        waivers={waiver.rule_id: waiver},
    )
    project = rp.Project(rp.ProjectSpec(profile=profile.coordinate, figures=(figure,)))

    assert figure.attestation_statements[attestation.rule_id] == attestation.rationale
    assert figure.active_waiver_rule_ids == (waiver.rule_id,)
    assert project.check().verdict is rp.Verdict.COMPLIANT

    violated = rp.FigureSpec(
        "figure-2",
        (rp.DeliverableSpec("pdf", "pdf"),),
        caption="Measured response.",
        waivers={waiver.rule_id: waiver},
    )
    violated_report = rp.Project(
        rp.ProjectSpec(profile=profile.coordinate, figures=(violated,))
    ).check()
    assert violated_report.verdict is rp.Verdict.NON_COMPLIANT

    mismatched = rp.Waiver(
        "metadata.alt_text.present",
        profile_digest="0" * 64,
        reviewer="Reviewer",
        reason="Test mismatch.",
        expires_on="2099-01-01",
    )
    with pytest.raises(ValueError, match="different profile digest"):
        rp.Project(
            rp.ProjectSpec(
                profile=profile.coordinate,
                figures=(
                    rp.FigureSpec(
                        "figure-1",
                        (rp.DeliverableSpec("pdf", "pdf"),),
                        waivers={mismatched.rule_id: mismatched},
                    ),
                ),
            )
        )


def test_schema_v3_parses_structured_attestation_and_waiver(tmp_path: Path) -> None:
    profile = rp.resolve_profile("acm-acmart@2026.08.0")
    config = tmp_path / "researchplot.toml"
    config.write_text(
        f'''schema_version = 3
profile = "{profile.coordinate}"

[[figures]]
id = "figure-1"
attestations = {{ "metadata.alt_text.distinct_from_caption" = {{ reviewer = "Reviewer", date = "2026-08-03", rationale = "The prose was reviewed.", evidence = ["review.md"] }} }}
waivers = {{ "metadata.alt_text.present" = {{ profile_digest = "{profile.digest}", reviewer = "Editor", reason = "Approved alternate workflow.", expires_on = "2099-01-01" }} }}

[[figures.deliverables]]
id = "pdf"
format = "pdf"
path = "figure.pdf"
''',
        encoding="utf-8",
    )

    spec = rp.ProjectSpec.load(config)

    assert isinstance(
        spec.figures[0].attestations["metadata.alt_text.distinct_from_caption"],
        rp.ManualAttestation,
    )
    assert isinstance(spec.figures[0].waivers["metadata.alt_text.present"], rp.Waiver)


def test_project_spec_requires_a_pinned_profile_and_required_deliverable() -> None:
    optional = rp.DeliverableSpec("preview", "png", required=False)
    with pytest.raises(ValueError, match="at least one required"):
        rp.FigureSpec("figure-1", (optional,))
    required = rp.DeliverableSpec("pdf", "pdf")
    with pytest.raises(ValueError, match="exact"):
        rp.ProjectSpec(profile="nature", figures=(rp.FigureSpec("figure-1", (required,)),))


def test_deliverable_format_and_filename_must_agree(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="declares 'pdf'"):
        rp.DeliverableSpec("pdf", "pdf", path=tmp_path / "figure.png")


def test_export_plan_separates_allowed_formats_from_default_selection() -> None:
    selected = rp.target("nature@2026.08.0", width="single", content="line-art")

    default = selected.plan_export()
    explicit = rp.plan_export(selected, formats=("eps", "svg"), preferred="svg")

    assert default.allowed_formats == (
        rp.OutputFormat.PDF,
        rp.OutputFormat.EPS,
        rp.OutputFormat.SVG,
    )
    assert default.selected_formats == (rp.OutputFormat.PDF,)
    assert explicit.selected_formats == (rp.OutputFormat.EPS, rp.OutputFormat.SVG)
    assert explicit.preferred_format is rp.OutputFormat.SVG
    with pytest.raises(ValueError, match="not allowed"):
        selected.plan_export(formats=("png",))
    with pytest.raises(ValueError, match="at least one"):
        selected.plan_export(formats=())
    with pytest.raises(ValueError, match="explicitly requested"):
        selected.plan_export(formats=("pdf",), preferred="svg")


def test_export_plan_compiles_raster_settings() -> None:
    selected = rp.target("plos-biology@2026.08.0", content="line-art")
    plan = selected.plan_export(formats=("tiff",))
    setting = plan.setting_for("tif")

    assert setting.minimum_dpi == pytest.approx(300)
    assert setting.allowed_color_modes == ("RGB", "L")
    assert setting.compression == "lzw"


def test_project_check_evaluates_bundle_rules_and_manual_attestations(tmp_path: Path) -> None:
    artifact = tmp_path / "figure.pdf"
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    fig.savefig(artifact)
    plt.close(fig)

    compliant = _acm_project(
        artifact,
        alt_text="A diagonal line rises from lower left to upper right.",
        attest=True,
    ).check()
    unresolved = _acm_project(
        artifact,
        alt_text="A diagonal line rises from lower left to upper right.",
        attest=False,
    ).check()
    failed = _acm_project(artifact, alt_text=None, attest=True).check()

    assert compliant.verdict is rp.Verdict.COMPLIANT
    assert unresolved.verdict is rp.Verdict.INDETERMINATE
    assert failed.verdict is rp.Verdict.NON_COMPLIANT


def test_file_only_check_cannot_hide_required_live_coverage(tmp_path: Path) -> None:
    selected = rp.target("nature@2026.08.0", width="single", content="line-art")
    with selected.style() as style:
        fig, ax = style.subplots()
        ax.plot([0, 1], [0, 1])
        ax.set(xlabel="Input", ylabel="Response")
    artifact = selected.export(fig, tmp_path / "figure.pdf").paths[0]
    project = rp.Project(
        rp.ProjectSpec(
            profile="nature@2026.08.0",
            figures=(
                rp.FigureSpec(
                    "figure-1",
                    (rp.DeliverableSpec("pdf", "pdf", path=artifact),),
                    width="single",
                    content="line-art",
                ),
            ),
        )
    )

    assessment = project.check()
    complete = project.check(live_figures={"figure-1": fig})
    plt.close(fig)

    assert assessment.verdict is rp.Verdict.INDETERMINATE
    assert complete.verdict is rp.Verdict.COMPLIANT
    missing_ids = {item.requirement.rule_id for item in assessment.unresolved}
    assert {"font.family", "font.size.min", "font.size.max", "line.width.min"} <= missing_ids


def test_coverage_policy_preserves_tri_state_semantics(tmp_path: Path) -> None:
    project = _acm_project(
        tmp_path / "missing.pdf",
        alt_text="A trend description.",
        attest=False,
    )
    assessment = project.compliance_plan.assess(())
    assert assessment.verdict is rp.Verdict.INDETERMINATE
    assert assessment.blocks("complete")
    assert not assessment.blocks("violations")
    with pytest.raises(rp.PlanPolicyError):
        rp.enforce_assessment(assessment, "complete")


def test_v1_project_config_has_an_explicit_deprecation_bridge(tmp_path: Path) -> None:
    artifact = tmp_path / "figure.pdf"
    artifact.write_bytes(b"placeholder")
    config_path = tmp_path / "researchplot.toml"
    config_path.write_text(
        f'''profile = "nature@2026.08.0"

[[figures]]
path = "{artifact.name}"
width = "single"
content = "line-art"
''',
        encoding="utf-8",
    )
    legacy = rp.ProjectConfig.load(config_path)

    with pytest.warns(DeprecationWarning, match="schema_version = 3"):
        spec = rp.project_spec_from_v1(legacy)

    assert spec.schema_version == 3
    assert spec.profile == "nature@2026.08.0"
    assert spec.figures[0].deliverables[0].path == artifact.resolve()


def test_rich_project_models_are_frozen_and_round_trip(tmp_path: Path) -> None:
    panel_data = tmp_path / "panel-a.csv"
    attachment = tmp_path / "protocol.txt"
    table = tmp_path / "figure-table.csv"
    figure = rp.FigureSpec(
        "figure-1",
        (rp.DeliverableSpec("main", "pdf", path=tmp_path / "figure.pdf"),),
        number="S1",
        caption="Supplementary response.",
        alt_text="Two panels compare measured responses.",
        long_description="Panel A rises, whereas panel B remains flat.",
        panels=(
            rp.PanelSpec(
                "panel-a",
                label="A",
                alt_text="A rising curve.",
                source_data=(panel_data,),
            ),
        ),
        data_table=table,
        attachments=(attachment,),
        waivers={"font.family": "The editor approved the required symbol font."},
        metadata={"nested": {"values": [1, 2]}},
    )
    manuscript = rp.ManuscriptSpec(
        tmp_path / "paper.pdf",
        "pdf",
        matching_hints=(rp.ManuscriptMatchHint("figure-1", pages=(4,), number="S1"),),
    )
    spec = rp.ProjectSpec(
        profile="nature@2026.08.0",
        figures=(figure,),
        manuscript=manuscript,
        config_path=tmp_path / "researchplot.toml",
    )

    payload = spec.to_dict()
    serialized = payload["figures"][0]  # type: ignore[index]
    assert serialized["number"] == "S1"
    assert serialized["data_table"] == "figure-table.csv"
    assert serialized["panels"][0]["source_data"] == ["panel-a.csv"]
    assert payload["manuscript"]["matching_hints"][0]["pages"] == [4]  # type: ignore[index]
    with pytest.raises(TypeError):
        figure.metadata["new"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="unknown figures"):
        rp.ProjectSpec(
            profile="nature@2026.08.0",
            figures=(figure,),
            manuscript=rp.ManuscriptSpec(
                tmp_path / "paper.pdf",
                "pdf",
                matching_hints=(rp.ManuscriptMatchHint("figure-2", pages=(1,)),),
            ),
        )


def test_project_schema_is_bundled_valid_and_returns_copies() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = rp.project_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(
        rp.ProjectSpec(
            profile="nature@2026.08.0",
            figures=(
                rp.FigureSpec(
                    "figure-1",
                    (rp.DeliverableSpec("main", "pdf"),),
                    width="single",
                ),
            ),
        ).to_dict()
    )
    assert schema["properties"]["schema_version"] == {"const": 3}  # type: ignore[index]
    schema["title"] = "mutated"
    assert rp.project_schema()["title"] == "ResearchPlot project specification"


def test_frozen_executable_plan_verifies_lock_before_checking_artifacts(
    tmp_path: Path,
) -> None:
    profile = rp.resolve_profile("nature@2026.08.0")
    lock_path = rp.write_profile_lock(profile, tmp_path / "researchplot.lock.json")
    project = rp.Project(
        rp.ProjectSpec(
            profile="nature@2026.08.0",
            figures=(
                rp.FigureSpec(
                    "figure-1",
                    (
                        rp.DeliverableSpec(
                            "main",
                            "pdf",
                            path=tmp_path / "missing.pdf",
                        ),
                    ),
                    width="single",
                ),
            ),
            lock_path=lock_path,
        )
    )
    plan = project.plan(frozen=True)
    lock_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="lock"):
        plan.check()
    with pytest.raises(ValueError, match="requires project.lock"):
        rp.Project(
            rp.ProjectSpec(
                profile="nature@2026.08.0",
                figures=project.spec.figures,
            )
        ).plan(frozen=True)


def test_figure_target_style_check_export_and_project_bundle(tmp_path: Path) -> None:
    artifact = tmp_path / "figure.pdf"
    project = rp.Project(
        rp.ProjectSpec(
            profile="nature@2026.08.0",
            policy="violations",
            figures=(
                rp.FigureSpec(
                    "figure-1",
                    (rp.DeliverableSpec("main", "pdf", path=artifact, preferred=True),),
                    width="single",
                    content="line-art",
                ),
            ),
        )
    )
    figure = project.figure("figure-1")
    with figure.style(deliverable="main") as style:
        canvas, ax = style.subplots()
        ax.plot([0, 1], [0, 1])
        ax.set(xlabel="Input", ylabel="Response")

    assessment = figure.check(fig=canvas, include_artifacts=False)
    result = figure.export(canvas)
    bundle = project.bundle(tmp_path / "submission")
    plt.close(canvas)

    assert assessment.verdict is rp.Verdict.INDETERMINATE
    assert result.paths == (artifact,)
    assert bundle.manifest_path.is_file()
    assert (bundle.path / artifact.name).is_file()


def test_required_manuscript_is_an_explicit_capability_gap(tmp_path: Path) -> None:
    project = rp.Project(
        rp.ProjectSpec(
            profile="nature@2026.08.0",
            figures=(
                rp.FigureSpec(
                    "figure-1",
                    (rp.DeliverableSpec("main", "pdf"),),
                    width="single",
                ),
            ),
            manuscript=rp.ManuscriptSpec(tmp_path / "paper.pdf", "pdf", required=True),
        )
    )

    assessment = project.compliance_plan.assess(())
    assert assessment.verdict is rp.Verdict.INDETERMINATE
    assert assessment.capability_gaps
