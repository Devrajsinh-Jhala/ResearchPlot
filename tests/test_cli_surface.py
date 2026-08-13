from __future__ import annotations

import json
from pathlib import Path

import pytest
from matplotlib import pyplot as plt
from PIL import Image

import researchplot as rp
import researchplot.cli as cli
from researchplot.cli import main


def _nature_pdf(path: Path) -> Path:
    selected = rp.target("nature@2026.08.0", width="single", content="line-art")
    with selected.style() as style:
        figure, axes = style.subplots()
        axes.plot([0, 1], [0, 1])
        axes.set(xlabel="Input", ylabel="Response")
    result = selected.export(figure, path, policy="violations")
    plt.close(figure)
    return result.paths[0]


def test_profile_cli_text_json_diff_validate_lock_and_explain(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["profile", "list", "--kind", "journal", "--year", "1900"]) == 0
    assert capsys.readouterr().out.strip() == "No matching profiles."

    assert main(["profile", "search", "nature"]) == 0
    assert "nature@2026.08.0" in capsys.readouterr().out
    assert main(["profile", "show", "nature@2026.08.0"]) == 0
    shown = capsys.readouterr().out
    assert "Nature" in shown and "Source:" in shown

    assert (
        main(
            [
                "profile",
                "diff",
                "nature@2026.08.0",
                "ieee-journal@2026.08.0",
            ]
        )
        == 0
    )
    assert "Added:" in capsys.readouterr().out

    source = Path(rp.__file__).parent / "profiles" / "nature.json"
    assert main(["profile", "validate", str(source), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True

    lock = tmp_path / "profile.lock.json"
    assert main(["profile", "lock", "nature@2026.08.0", "--output", str(lock)]) == 0
    assert str(lock) in capsys.readouterr().out
    assert main(["profile", "verify", "--lock", str(lock)]) == 0
    assert "Verified nature@2026.08.0" in capsys.readouterr().out

    assert main(["explain", "figure.width.single", "--profile", "nature"]) == 0
    assert "Source:" in capsys.readouterr().out
    assert main(["explain", "missing.rule", "--profile", "nature"]) == 2
    assert "has no rule" in capsys.readouterr().err


def test_cli_report_modes_directory_expansion_and_argument_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artifact_dir = tmp_path / "figures"
    artifact_dir.mkdir()
    artifact = _nature_pdf(artifact_dir / "figure.pdf")
    (artifact_dir / "ignored.txt").write_text("ignored", encoding="utf-8")

    text_report = tmp_path / "audit.txt"
    assert (
        main(
            [
                "audit",
                str(artifact_dir),
                "--profile",
                "nature@2026.08.0",
                "--width",
                "single",
                "--content",
                "line-art",
                "--output",
                str(text_report),
            ]
        )
        == 0
    )
    assert str(artifact) in text_report.read_text(encoding="utf-8")

    assert (
        main(
            [
                "audit",
                str(artifact),
                "--profile",
                "nature",
                "--json",
                "--sarif",
            ]
        )
        == 2
    )
    assert "Choose only one" in capsys.readouterr().err
    assert main(["check", str(artifact)]) == 2
    assert "--profile is required" in capsys.readouterr().err
    assert main(["check", "--profile", "nature"]) == 2
    assert "at least one figure" in capsys.readouterr().err

    assert (
        main(
            [
                "audit",
                str(artifact),
                "--profile",
                "nature",
                "--format",
                "html",
            ]
        )
        == 2
    )
    assert "HTML reports require" in capsys.readouterr().err


def test_project_report_modes_doctor_fix_and_manuscript_text(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artifact = _nature_pdf(tmp_path / "figure.pdf")
    config = tmp_path / "researchplot.toml"
    lock = tmp_path / "researchplot.lock.json"
    assert (
        main(
            [
                "init",
                "--profile",
                "nature@2026.08.0",
                "--figure",
                str(artifact),
                "--width",
                "single",
                "--content",
                "line-art",
                "--output",
                str(config),
                "--lock",
                str(lock),
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert main(["check", "--config", str(config)]) == 3
    assert "Verdict: indeterminate" in capsys.readouterr().out
    sarif = tmp_path / "project.sarif"
    assert (
        main(
            [
                "check",
                "--config",
                str(config),
                "--format",
                "sarif",
                "--output",
                str(sarif),
            ]
        )
        == 3
    )
    assert json.loads(sarif.read_text(encoding="utf-8"))["version"] == "2.1.0"
    html = tmp_path / "project.html"
    assert main(["check", "--config", str(config), "--html", str(html)]) == 3
    assert "<!doctype html>" in html.read_text(encoding="utf-8")

    assert main(["doctor", "--profile", "nature@2026.08.0"]) == 0
    assert "ResearchPlot doctor" in capsys.readouterr().out

    raster = tmp_path / "plain.png"
    Image.new("RGB", (20, 20), "white").save(raster)
    markdown = tmp_path / "fix.md"
    assert main(["fix", str(raster), "--plan", "--output", str(markdown)]) in {0, 1, 3}
    assert "remediation" in markdown.read_text(encoding="utf-8").casefold()

    assert main(["manuscript", "check", str(artifact)]) == 3
    manuscript_output = capsys.readouterr().out
    assert "Manuscript:" in manuscript_output and "Verdict: indeterminate" in manuscript_output
    assert main(["manuscript", "check"]) == 2
    assert "Provide a manuscript" in capsys.readouterr().err


def test_cli_tar_archive_round_trip_and_archive_output_guard(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artifact = _nature_pdf(tmp_path / "figure.pdf")
    config = tmp_path / "researchplot.toml"
    lock = tmp_path / "researchplot.lock.json"
    assert (
        main(
            [
                "init",
                "--profile",
                "nature@2026.08.0",
                "--figure",
                str(artifact),
                "--width",
                "single",
                "--content",
                "line-art",
                "--output",
                str(config),
                "--lock",
                str(lock),
            ]
        )
        == 0
    )
    capsys.readouterr()
    directory = tmp_path / "submission"
    assert (
        main(
            [
                "bundle",
                "build",
                "--config",
                str(config),
                "--output",
                str(directory),
                "--policy",
                "violations",
            ]
        )
        == 0
    )
    capsys.readouterr()
    archive = tmp_path / "submission.tar"
    assert main(["bundle", "archive", str(directory), str(archive)]) == 0
    capsys.readouterr()
    assert main(["bundle", "verify", str(archive), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True

    assert (
        main(
            [
                "bundle",
                "build",
                "--config",
                str(config),
                "--output",
                str(tmp_path / "invalid.tar"),
            ]
        )
        == 2
    )
    assert "ZIP or TAR" in capsys.readouterr().err


def test_rich_project_toml_serializer_round_trips_author_metadata(tmp_path: Path) -> None:
    profile = rp.resolve_profile("nature@2026.08.0")
    attestation = rp.ManualAttestation(
        "manual.review",
        reviewer="Reviewer",
        date="2026-08-13",
        rationale="Reviewed against the proof.",
        evidence=("reviews/proof.md",),
    )
    waiver = rp.Waiver(
        "recommendation.rule",
        profile_digest=profile.digest,
        reviewer="Editor",
        reason="Approved for this submission.",
        expires_on="2099-01-01",
    )
    figure = rp.FigureSpec(
        "figure-1",
        (
            rp.DeliverableSpec(
                "main",
                "pdf",
                path=tmp_path / "figures" / "figure-1.pdf",
                preferred=True,
            ),
        ),
        width="single",
        number="S1",
        caption="Response by input.",
        alt_text="A line rises from left to right.",
        long_description="The response doubles between successive inputs.",
        key_trends=("Response increases monotonically.",),
        panels=(
            rp.PanelSpec(
                "panel-a",
                label="A",
                order=1,
                description="Primary response.",
                source_data=(tmp_path / "data" / "panel-a.csv",),
            ),
        ),
        source_data=(tmp_path / "data" / "figure.csv",),
        data_table=tmp_path / "data" / "accessible.csv",
        attachments=(tmp_path / "code" / "plot.py",),
        attestations={attestation.rule_id: attestation},
        waivers={waiver.rule_id: waiver},
    )
    spec = rp.ProjectSpec(
        profile=profile.coordinate,
        figures=(figure,),
        manuscript=rp.ManuscriptSpec(
            tmp_path / "paper.pdf",
            "pdf",
            required=True,
            matching_hints=(
                rp.ManuscriptMatchHint(
                    "figure-1", pages=(2,), number="S1", caption="Response by input."
                ),
            ),
        ),
        lock_path=tmp_path / "researchplot.lock.json",
        config_path=tmp_path / "researchplot.toml",
    )
    rendered = cli._project_toml(spec, root=tmp_path)
    destination = tmp_path / "researchplot.toml"
    destination.write_text(rendered, encoding="utf-8")

    loaded = rp.ProjectSpec.load(destination)
    assert loaded.figures[0].key_trends == figure.key_trends
    assert loaded.figures[0].panels[0].order == 1
    assert isinstance(loaded.figures[0].attestations[attestation.rule_id], rp.ManualAttestation)
    assert isinstance(loaded.figures[0].waivers[waiver.rule_id], rp.Waiver)
    with pytest.raises(ValueError, match="Cannot serialize"):
        cli._toml_inline(object())
