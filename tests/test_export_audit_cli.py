from __future__ import annotations

import json
from pathlib import Path

import pytest
from matplotlib import pyplot as plt

import researchplot as rp
from researchplot.cli import main


def _simple_figure(venue: str = "ieee", width: str | None = None):
    context = rp.use(venue, width=width)
    with context:
        fig, ax = context.subplots()
        ax.plot([0, 1], [0, 1])
    return context, fig


def test_strict_export_blocks_required_failures(tmp_path: Path) -> None:
    fig, _ = plt.subplots(figsize=(2, 2))
    with pytest.raises(rp.ComplianceError) as error:
        rp.export_figure(fig, tmp_path / "bad.pdf", venue="nature", width="single")
    assert error.value.report.failures
    assert not (tmp_path / "bad.pdf").exists()


def test_vector_export_and_pdf_eps_audits(tmp_path: Path) -> None:
    context, fig = _simple_figure()
    outputs = context.export(fig, tmp_path / "figure", formats=["pdf", "eps"])
    assert {path.suffix for path in outputs} == {".pdf", ".eps"}
    for output in outputs:
        report = rp.audit_file(output, venue="ieee", width="single", artwork="vector")
        assert report.passed
        assert any(check.rule_id == "figure.width.single" for check in report.checks)


@pytest.mark.parametrize("suffix", ["png", "jpeg", "tiff"])
def test_raster_audits_dimensions_and_dpi(tmp_path: Path, suffix: str) -> None:
    context, fig = _simple_figure()
    output = tmp_path / f"raster.{suffix}"
    fig.savefig(output, dpi=300, bbox_inches=None)
    report = rp.audit_file(output, venue="ieee", width="single", artwork="halftone")
    assert report.passed
    dpi_check = next(check for check in report.checks if check.rule_id == "export.min_dpi.halftone")
    assert dpi_check.status is rp.CheckStatus.PASS


def test_svg_audit_reports_unavailable_font_check_as_skipped(tmp_path: Path) -> None:
    context, fig = _simple_figure("nature")
    output = context.export(fig, tmp_path / "nature.svg", formats=None)[0]
    report = rp.audit_file(output, venue="nature", width="single", artwork="vector")
    font = next(check for check in report.checks if check.rule_id == "font.family")
    assert font.status is rp.CheckStatus.SKIP


def test_export_rejects_disallowed_format_and_low_dpi(tmp_path: Path) -> None:
    context, fig = _simple_figure("elsevier-generic")
    with pytest.warns(UserWarning, match="not listed"):
        paths = context.export(fig, tmp_path / "advisory.png", artwork="vector")
    assert paths[0].exists()
    with pytest.raises(rp.ComplianceError):
        context.export(fig, tmp_path / "bad.tiff", artwork="halftone", dpi=72)
    with pytest.warns(UserWarning, match="below"):
        paths = context.export(
            fig,
            tmp_path / "non-strict.tiff",
            artwork="halftone",
            dpi=72,
            strict=False,
        )
    assert paths[0].exists()

    ieee, ieee_fig = _simple_figure("ieee")
    with pytest.raises(rp.ComplianceError):
        ieee.export(ieee_fig, tmp_path / "required.png", artwork="vector")


def test_audit_missing_and_unsupported_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        rp.audit_file(tmp_path / "missing.pdf", venue="ieee", width="single")
    path = tmp_path / "figure.txt"
    path.write_text("not a figure", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        rp.audit_file(path, venue="ieee", width="single")


def test_cli_json_and_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["venues", "list", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert len(listed) == 7

    assert main(["doctor", "--venue", "nature", "--json"]) == 0
    doctor = json.loads(capsys.readouterr().out)
    assert doctor["profile_id"] == "nature"

    context, fig = _simple_figure()
    pdf = context.export(fig, tmp_path / "cli.pdf")[0]
    assert (
        main(
            [
                "audit",
                str(pdf),
                "--venue",
                "ieee",
                "--width",
                "single",
                "--artwork",
                "vector",
                "--json",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["passed"] is True

    assert main(["venues", "info", "not-a-venue", "--json"]) == 2
    error = json.loads(capsys.readouterr().err)
    assert "Unknown venue" in error["error"]


def test_cli_unreadable_file_is_invalid_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a pdf")
    assert (
        main(
            [
                "audit",
                str(broken),
                "--venue",
                "ieee",
                "--width",
                "single",
                "--artwork",
                "vector",
                "--json",
            ]
        )
        == 2
    )
    assert json.loads(capsys.readouterr().err)["exit_code"] == 2
