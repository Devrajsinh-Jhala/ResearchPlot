from __future__ import annotations

from pathlib import Path

import pytest
from matplotlib import pyplot as plt

import researchplot as rp
from researchplot.cli import main


def _project(
    path: Path,
    *,
    profile: str = "iclr-2026@2026.08.0",
    policy: str = "complete",
    **figure_kwargs: object,
) -> rp.Project:
    return rp.Project(
        rp.ProjectSpec(
            profile=profile,
            policy=policy,
            figures=(
                rp.FigureSpec(
                    "figure-1",
                    (rp.DeliverableSpec("main", "pdf", path=path, preferred=True),),
                    **figure_kwargs,  # type: ignore[arg-type]
                ),
            ),
        )
    )


def _nature_figure():
    target = rp.target("nature@2026.08.0", width="single", content="line-art")
    with target.style() as style:
        figure, axes = style.subplots()
        axes.plot([0, 1], [0, 1])
        axes.set(xlabel="Input", ylabel="Response")
    return target, figure


def test_missing_required_deliverable_is_unresolved_without_venue_file_rules(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path / "missing.pdf")

    report = project.check()

    assert report.verdict is rp.Verdict.INDETERMINATE
    assert any(
        item.requirement.rule_id == "project.deliverable.present"
        and item.requirement.deliverable_id == "main"
        for item in report.unresolved
    )
    with pytest.raises(rp.PlanPolicyError) as error:
        project.bundle(tmp_path / "submission")
    assert error.value.assessment.verdict is rp.Verdict.INDETERMINATE
    assert not (tmp_path / "submission").exists()


@pytest.mark.parametrize("reference", ["source_data", "attachments", "data_table", "panels"])
def test_missing_referenced_evidence_is_unresolved_without_metadata_rules(
    tmp_path: Path, reference: str
) -> None:
    artifact = tmp_path / "figure.pdf"
    fig, _ = plt.subplots()
    try:
        fig.savefig(artifact)
    finally:
        plt.close(fig)
    missing = tmp_path / "missing.csv"
    options: dict[str, object] = {
        "source_data": (missing,),
        "attachments": (missing,),
        "data_table": missing,
        "panels": (rp.PanelSpec("panel-a", source_data=(missing,)),),
    }
    project = _project(artifact, **{reference: options[reference]})

    report = project.check()

    assert report.verdict is rp.Verdict.INDETERMINATE
    assert any(item.requirement.rule_id == "project.evidence.present" for item in report.unresolved)


def test_existing_required_deliverable_satisfies_project_availability(tmp_path: Path) -> None:
    artifact = tmp_path / "figure.pdf"
    fig, _ = plt.subplots()
    try:
        fig.savefig(artifact)
    finally:
        plt.close(fig)
    project = _project(artifact)

    assert project.check().verdict is rp.Verdict.COMPLIANT
    bundle = project.bundle(tmp_path / "submission")
    assert bundle.passed
    assert all(path.is_file() for item in bundle.items for path in item.paths)


def test_existing_source_data_satisfies_the_project_and_is_bundled(tmp_path: Path) -> None:
    artifact = tmp_path / "figure.pdf"
    data = tmp_path / "source.csv"
    data.write_text("input,response\n0,0\n1,1\n", encoding="utf-8")
    fig, _ = plt.subplots()
    try:
        fig.savefig(artifact)
    finally:
        plt.close(fig)
    project = _project(artifact, source_data=(data,))

    assert project.check().verdict is rp.Verdict.COMPLIANT
    bundle = project.bundle(tmp_path / "submission")

    assert rp.verify_manifest(bundle.path).valid
    assert (bundle.path / "source-data/figure.csv").read_bytes() == data.read_bytes()


def test_missing_optional_deliverable_does_not_block_an_existing_required_one(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "figure.pdf"
    fig, _ = plt.subplots()
    try:
        fig.savefig(artifact)
    finally:
        plt.close(fig)
    project = rp.Project(
        rp.ProjectSpec(
            profile="iclr-2026@2026.08.0",
            figures=(
                rp.FigureSpec(
                    "figure-1",
                    (
                        rp.DeliverableSpec("main", "pdf", path=artifact, preferred=True),
                        rp.DeliverableSpec(
                            "alternative", "pdf", path=tmp_path / "missing.pdf", required=False
                        ),
                    ),
                ),
            ),
        )
    )

    assert project.check().verdict is rp.Verdict.COMPLIANT
    assert project.bundle(tmp_path / "submission").passed


def test_complete_bundle_cannot_hide_missing_live_coverage(tmp_path: Path) -> None:
    target, fig = _nature_figure()
    try:
        artifact = target.export(fig, tmp_path / "figure.pdf").paths[0]
        project = _project(
            artifact,
            profile=target.coordinate,
            width="single",
            content="line-art",
        )
        assert project.check().verdict is rp.Verdict.INDETERMINATE

        with pytest.raises(rp.PlanPolicyError) as error:
            project.bundle(tmp_path / "submission")

        assert error.value.assessment.verdict is rp.Verdict.INDETERMINATE
        assert not (tmp_path / "submission").exists()
        assert not list(tmp_path.glob(".researchplot-project-bundle-*"))
    finally:
        plt.close(fig)


def test_complete_live_bundle_checks_the_newly_exported_artifact(tmp_path: Path) -> None:
    target, fig = _nature_figure()
    try:
        project = _project(
            tmp_path / "not-exported-yet.pdf",
            profile=target.coordinate,
            width="single",
            content="line-art",
        )

        bundle = project.bundle(tmp_path / "submission", live_figures={"figure-1": fig})

        assert bundle.passed
        assert bundle.manifest_path.is_file()
        assert all(path.is_file() for item in bundle.items for path in item.paths)
        assert rp.verify_manifest(bundle.path).valid
        assert not list(tmp_path.glob(".researchplot-project-bundle-*"))
    finally:
        plt.close(fig)


def test_bundle_cannot_omit_another_required_representation(tmp_path: Path) -> None:
    fig, _ = plt.subplots()
    artifact = tmp_path / "figure.pdf"
    try:
        fig.savefig(artifact)
    finally:
        plt.close(fig)
    project = rp.Project(
        rp.ProjectSpec(
            profile="iclr-2026@2026.08.0",
            figures=(
                rp.FigureSpec(
                    "figure-1",
                    (
                        rp.DeliverableSpec("main", "pdf", path=artifact, preferred=True),
                        rp.DeliverableSpec("second", "pdf", path=artifact.with_name("second.pdf")),
                    ),
                ),
            ),
        )
    )
    artifact.with_name("second.pdf").write_bytes(artifact.read_bytes())

    with pytest.raises(rp.PlanPolicyError) as error:
        project.bundle(tmp_path / "submission")

    assert error.value.assessment.verdict is rp.Verdict.INDETERMINATE
    assert not (tmp_path / "submission").exists()
    assert not list(tmp_path.glob(".researchplot-project-bundle-*"))


def _write_config(config: Path, *, width: str = "single") -> None:
    config.write_text(
        f'''schema_version = 3
profile = "nature@2026.08.0"
policy = "complete"

[[figures]]
id = "figure-1"
width = "{width}"
content = "line-art"

[[figures.deliverables]]
id = "main"
format = "pdf"
path = "figure.pdf"
preferred = true
''',
        encoding="utf-8",
    )


@pytest.mark.parametrize(("width", "expected_exit"), [("single", 3), ("double", 1)])
def test_cli_bundle_preserves_coverage_policy_exit_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], width: str, expected_exit: int
) -> None:
    target, fig = _nature_figure()
    try:
        target.export(fig, tmp_path / "figure.pdf")
    finally:
        plt.close(fig)
    config = tmp_path / "researchplot.toml"
    _write_config(config, width=width)

    code = main(
        ["bundle", "build", "--config", str(config), "--output", str(tmp_path / "submission")]
    )

    assert code == expected_exit
    assert "complete policy blocked" in capsys.readouterr().err
    assert not (tmp_path / "submission").exists()
