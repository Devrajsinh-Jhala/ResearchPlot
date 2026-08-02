from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

import researchplot as rp
import researchplot.figure_inspector as figure_inspector
import researchplot.transactional_export as transactional_export
from researchplot.cli import main


def _nature_figure() -> tuple[rp.Target, Figure]:
    selected = rp.target("nature@2026.08.0", width="single", content="line-art")
    with selected.style() as style:
        fig, ax = style.subplots()
        ax.plot([0, 1, 2], [0, 1, 0], label="Measured")
        ax.set(xlabel="Input", ylabel="Response")
        ax.legend()
    return selected, fig


def test_transactional_multiformat_export_and_manifest(tmp_path: Path) -> None:
    selected, fig = _nature_figure()

    result = selected.export(fig, tmp_path / "figure")

    assert {path.suffix for path in result.paths} == {".pdf", ".eps", ".svg"}
    assert result.report.verdict is rp.Verdict.COMPLIANT
    assert result.manifest_path.is_file()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["profile"] == "nature@2026.08.0"
    assert len(manifest["profile_digest"]) == 64
    assert manifest["sources"][0]["locator"]
    for artifact in manifest["artifacts"]:
        path = tmp_path / artifact["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
    file_findings = [finding for finding in result.report.findings if finding.phase == "file"]
    assert {finding.artifact for finding in file_findings} >= {
        "figure.pdf",
        "figure.eps",
        "figure.svg",
    }
    plt.close(fig)


def test_post_audit_failure_leaves_no_partial_outputs(tmp_path: Path) -> None:
    selected, fig = _nature_figure()

    with pytest.raises(rp.CompliancePolicyError):
        selected.export(fig, tmp_path / "blocked", formats=("pdf", "png"))

    assert not list(tmp_path.glob("blocked*"))
    plt.close(fig)


def test_plos_export_applies_required_tiff_dpi_mode_and_compression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(figure_inspector, "_effective_family", lambda _text: "Arial")
    selected = rp.target("plos-biology@2026.08.0", content="line-art")
    with selected.style() as style:
        fig, ax = style.subplots()
        ax.plot([0, 1], [0, 1])
        ax.set(xlabel="Input", ylabel="Response")

    result = selected.export(fig, tmp_path / "figure")

    assert {path.suffix for path in result.paths} == {".tiff", ".eps"}
    findings = {finding.rule_id: finding for finding in result.report.findings}
    assert findings["raster.resolution.range"].observed == pytest.approx(300)
    assert findings["raster.color_mode.allowed"].observed == "RGB"
    assert findings["raster.compression.required"].observed == "lzw"
    assert result.report.verdict is rp.Verdict.COMPLIANT
    plt.close(fig)


def test_export_refuses_overwrite_unless_explicit(tmp_path: Path) -> None:
    selected, fig = _nature_figure()
    destination = tmp_path / "figure.pdf"
    first = selected.export(fig, destination)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        selected.export(fig, destination)

    second = selected.export(fig, destination, overwrite=True)
    assert first.paths == second.paths
    assert destination.is_file()
    plt.close(fig)


def test_export_never_replaces_an_existing_directory(tmp_path: Path) -> None:
    selected, fig = _nature_figure()
    destination = tmp_path / "figure.pdf"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("preserve me", encoding="utf-8")

    with pytest.raises(IsADirectoryError, match="only replaces existing regular files"):
        selected.export(fig, destination, overwrite=True)

    assert destination.is_dir()
    assert marker.read_text(encoding="utf-8") == "preserve me"
    plt.close(fig)


def test_export_never_replaces_an_existing_manifest_directory(tmp_path: Path) -> None:
    selected, fig = _nature_figure()
    destination = tmp_path / "figure.pdf"
    manifest = tmp_path / "figure.researchplot.json"
    manifest.mkdir()
    marker = manifest / "keep.txt"
    marker.write_text("preserve me", encoding="utf-8")

    with pytest.raises(IsADirectoryError, match="only replaces existing regular files"):
        selected.export(fig, destination, overwrite=True)

    assert not destination.exists()
    assert manifest.is_dir()
    assert marker.read_text(encoding="utf-8") == "preserve me"
    plt.close(fig)


def test_no_overwrite_rejects_a_destination_created_during_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected, fig = _nature_figure()
    destination = tmp_path / "figure.pdf"
    original_commit = transactional_export._commit_staged

    def inject_collision(
        staged: tuple[Path, ...], final: tuple[Path, ...], *, overwrite: bool
    ) -> None:
        final[0].write_text("created concurrently", encoding="utf-8")
        original_commit(staged, final, overwrite=overwrite)

    monkeypatch.setattr(transactional_export, "_commit_staged", inject_collision)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        selected.export(fig, destination)

    assert destination.read_text(encoding="utf-8") == "created concurrently"
    assert not (tmp_path / "figure.researchplot.json").exists()
    plt.close(fig)


def test_acm_widthless_bundle_and_manual_attestation(tmp_path: Path) -> None:
    source = tmp_path / "input.pdf"
    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])
    fig.savefig(source)
    plt.close(fig)

    submission = rp.Submission("acm-acmart@2026.08.0", output_dir=tmp_path / "submission")
    submission.add(
        "figure1.pdf",
        source,
        alt_text="A line falls from left to right.",
        caption="Model response.",
        attestations={"metadata.alt_text.distinct_from_caption": "Reviewed by the authors."},
    )

    result = submission.build()

    assert result.passed
    assert result.manifest_path.is_file()
    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert payload["profile"] == "acm-acmart@2026.08.0"
    assert payload["researchplot_version"]
    assert payload["sources"]
    assert payload["figures"][0]["report"]["verdict"] == "compliant"


def test_acm_bundle_rejects_whitespace_only_alt_text(tmp_path: Path) -> None:
    source = tmp_path / "input.pdf"
    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])
    fig.savefig(source)
    plt.close(fig)
    submission = rp.Submission("acm-acmart@2026.08.0", output_dir=tmp_path / "submission")
    submission.add(
        "figure1.pdf",
        source,
        alt_text="   ",
        caption="Model response.",
        attestations={"metadata.alt_text.distinct_from_caption": "Reviewed."},
    )

    with pytest.raises(rp.CompliancePolicyError) as error:
        submission.build()

    assert error.value.report.verdict is rp.Verdict.NON_COMPLIANT


def test_submission_rejects_derived_output_collisions(tmp_path: Path) -> None:
    source = tmp_path / "input.pdf"
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    fig.savefig(source)
    plt.close(fig)
    submission = rp.Submission("acm-acmart@2026.08.0", output_dir=tmp_path / "bundle")
    metadata = {
        "alt_text": "A line rises from left to right.",
        "caption": "Trend.",
        "attestations": {"metadata.alt_text.distinct_from_caption": "Reviewed."},
    }
    submission.add("figure", source, **metadata)
    submission.add("figure.pdf", source, **metadata)

    with pytest.raises(ValueError, match="Bundle path.*conflicts"):
        submission.build()

    assert not (tmp_path / "bundle").exists()


def test_submission_rejects_casefolded_source_data_collisions(tmp_path: Path) -> None:
    source = tmp_path / "input.pdf"
    source_data = tmp_path / "input.csv"
    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])
    fig.savefig(source)
    plt.close(fig)
    source_data.write_text("x,y\n0,1\n", encoding="utf-8")
    submission = rp.Submission("acm-acmart@2026.08.0", output_dir=tmp_path / "bundle")
    metadata = {
        "alt_text": "A line falls from left to right.",
        "caption": "Trend.",
        "source_data": source_data,
        "attestations": {"metadata.alt_text.distinct_from_caption": "Reviewed."},
    }
    submission.add("Figure.pdf", source, **metadata)
    submission.add("figure.eps", source, **metadata)

    with pytest.raises(ValueError, match="source data.*conflicts"):
        submission.build()


@pytest.mark.parametrize("name", ["CON.pdf", "name:stream.pdf", "trailing."])
def test_submission_rejects_nonportable_names(name: str) -> None:
    fig = Figure()
    submission = rp.Submission("acm-acmart@2026.08.0")

    with pytest.raises(ValueError, match="reserved|not portable|trailing"):
        submission.add(name, fig)


def test_submission_rejects_non_string_descriptive_metadata() -> None:
    submission = rp.Submission("acm-acmart@2026.08.0")

    with pytest.raises(TypeError, match="alt_text"):
        submission.add("figure.pdf", Figure(), alt_text=123)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="caption"):
        submission.add("figure.pdf", Figure(), caption=123)  # type: ignore[arg-type]


def test_submission_manifest_uses_detected_content_format(tmp_path: Path) -> None:
    source = tmp_path / "actual.png"
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    fig.savefig(source)
    plt.close(fig)
    submission = rp.Submission("acm-acmart@2026.08.0", output_dir=tmp_path / "bundle")
    submission.add(
        "mislabelled.pdf",
        source,
        alt_text="A line rises from left to right.",
        caption="Trend.",
        attestations={"metadata.alt_text.distinct_from_caption": "Reviewed."},
    )

    result = submission.build()

    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    artifact = next(item for item in payload["artifacts"] if item["path"] == "mislabelled.pdf")
    assert artifact["format"] == "png"


def test_submission_manifest_labels_extensionless_source_data(tmp_path: Path) -> None:
    source = tmp_path / "input.pdf"
    source_data = tmp_path / "dataset"
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    fig.savefig(source)
    plt.close(fig)
    source_data.write_text("0 1\n1 2\n", encoding="utf-8")
    submission = rp.Submission("acm-acmart@2026.08.0", output_dir=tmp_path / "bundle")
    submission.add(
        "figure.pdf",
        source,
        alt_text="A line rises from left to right.",
        caption="Trend.",
        source_data=source_data,
        attestations={"metadata.alt_text.distinct_from_caption": "Reviewed."},
    )

    result = submission.build()

    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    artifact = next(item for item in payload["artifacts"] if item["path"] == "source-data/figure")
    assert artifact["format"] == "data"


def test_project_config_resolves_relative_paths_and_bundle_copies_source_data(
    tmp_path: Path,
) -> None:
    selected, fig = _nature_figure()
    artifact = selected.export(fig, tmp_path / "figure.pdf").paths[0]
    plt.close(fig)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    source_data = data_dir / "figure.csv"
    source_data.write_text("x,y\n0,0\n1,1\n", encoding="utf-8")
    config_path = tmp_path / "researchplot.toml"
    config_path.write_text(
        """
profile = "nature@2026.08.0"
policy = "violations"

[[figures]]
path = "figure.pdf"
role = "main"
width = "single"
content = "line-art"
alt_text = "A line rises and falls."
caption = "Measured response."
source_data = "data/figure.csv"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = rp.ProjectConfig.load(config_path)
    assert config.figures[0].path == artifact.resolve()
    assert config.figures[0].source_data == source_data.resolve()
    submission = rp.Submission(config.profile, output_dir=tmp_path / "bundle", policy=config.policy)
    figure_config = config.figures[0]
    submission.add(
        figure_config.path.name,
        figure_config.path,
        role=figure_config.role,
        width=figure_config.width,
        content=figure_config.content,
        alt_text=figure_config.alt_text,
        caption=figure_config.caption,
        source_data=figure_config.source_data,
    )
    result = submission.build()

    copied_data = result.path / "source-data" / "figure.csv"
    assert copied_data.read_text(encoding="utf-8") == source_data.read_text(encoding="utf-8")
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    paths = {artifact_record["path"] for artifact_record in manifest["artifacts"]}
    assert paths == {"figure.pdf", "source-data/figure.csv"}


def test_project_config_uses_the_public_data_visualization_default(tmp_path: Path) -> None:
    artifact = tmp_path / "figure.pdf"
    artifact.write_bytes(b"placeholder")
    config_path = tmp_path / "researchplot.toml"
    config_path.write_text(
        """profile = "acm-acmart@2026.08.0"

[[figures]]
path = "figure.pdf"
""",
        encoding="utf-8",
    )

    config = rp.ProjectConfig.load(config_path)

    assert config.figures[0].content is rp.ContentKind.DATA_VISUALIZATION


def test_cli_profile_check_json_sarif_and_exit_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    selected, fig = _nature_figure()
    artifact = selected.export(fig, tmp_path / "figure.pdf").paths[0]
    plt.close(fig)

    assert main(["profile", "list", "--json"]) == 0
    catalog = json.loads(capsys.readouterr().out)
    assert len(catalog) == 9

    assert (
        main(
            [
                "check",
                str(artifact),
                "--profile",
                "nature@2026.08.0",
                "--width",
                "single",
                "--content",
                "line-art",
                "--format",
                "json",
            ]
        )
        == 0
    )
    report_payload = json.loads(capsys.readouterr().out)
    assert report_payload[0]["report"]["verdict"] == "compliant"

    sarif_path = tmp_path / "report.sarif"
    assert (
        main(
            [
                "check",
                str(artifact),
                "--profile",
                "nature@2026.08.0",
                "--width",
                "double",
                "--content",
                "line-art",
                "--format",
                "sarif",
                "--output",
                str(sarif_path),
            ]
        )
        == 1
    )
    sarif = json.loads(sarif_path.read_text(encoding="utf-8"))
    assert sarif["version"] == "2.1.0"
    assert sarif["runs"][0]["results"]
    for descriptor in sarif["runs"][0]["tool"]["driver"]["rules"]:
        assert "helpUri" not in descriptor or isinstance(descriptor["helpUri"], str)

    assert main(["check", str(artifact), "--profile", "not-a-profile"]) == 2
    assert "Unknown venue" in capsys.readouterr().err


def test_cli_bundle_uses_compliance_exit_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "input.pdf"
    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])
    fig.savefig(source)
    plt.close(fig)

    relaxed = tmp_path / "relaxed.toml"
    relaxed.write_text(
        f'''profile = "acm-acmart@2026.08.0"
policy = "off"

[[figures]]
path = "{source.as_posix()}"
''',
        encoding="utf-8",
    )
    relaxed_output = tmp_path / "relaxed-bundle"
    assert (
        main(
            [
                "bundle",
                "build",
                "--config",
                str(relaxed),
                "--output",
                str(relaxed_output),
            ]
        )
        == 1
    )
    assert relaxed_output.is_dir()
    capsys.readouterr()

    unresolved = tmp_path / "unresolved.toml"
    unresolved.write_text(
        f'''profile = "acm-acmart@2026.08.0"
policy = "complete"

[[figures]]
path = "{source.as_posix()}"
alt_text = "A line falls from left to right."
caption = "Trend."
''',
        encoding="utf-8",
    )
    unresolved_output = tmp_path / "unresolved-bundle"
    assert (
        main(
            [
                "bundle",
                "build",
                "--config",
                str(unresolved),
                "--output",
                str(unresolved_output),
            ]
        )
        == 3
    )
    assert not unresolved_output.exists()
    assert "indeterminate" in capsys.readouterr().err


def test_profile_lock_is_deterministic(tmp_path: Path) -> None:
    profile = rp.resolve_profile("nature@2026.08.0")
    path = tmp_path / "researchplot.lock.json"

    rp.write_profile_lock(profile, path)
    first = path.read_bytes()
    rp.write_profile_lock(profile, path)

    assert path.read_bytes() == first
    payload = json.loads(first)
    assert payload["profile"] == profile.coordinate
    assert payload["digest"] == profile.digest
