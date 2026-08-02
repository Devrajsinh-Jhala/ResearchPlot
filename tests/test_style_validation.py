from __future__ import annotations

import matplotlib as mpl
import pytest
from matplotlib import pyplot as plt

import researchplot as rp


@pytest.mark.parametrize(
    "profile_id",
    [
        "ieee-journal",
        "nature",
        "elsevier-generic",
        "neurips-2026",
        "icml-2026",
        "cvpr-2026",
        "acl-2026",
    ],
)
def test_context_creates_exact_default_width(profile_id: str) -> None:
    profile = rp.resolve_venue(profile_id)
    with rp.use(profile) as style:
        fig, _ = style.subplots()
    assert fig.get_size_inches()[0] * 25.4 == pytest.approx(profile.width_mm(), abs=1e-8)


def test_context_restores_rcparams_after_success_and_exception() -> None:
    original = mpl.rcParams["font.size"]
    with rp.use("nature"):
        assert mpl.rcParams["font.size"] == 7
    assert mpl.rcParams["font.size"] == original

    with pytest.raises(RuntimeError, match="boom"):
        with rp.use("nature"):
            assert mpl.rcParams["font.size"] == 7
            raise RuntimeError("boom")
    assert mpl.rcParams["font.size"] == original


def test_overrides_take_precedence_and_are_validated() -> None:
    with rp.use("ieee", overrides={"axes.grid": True, "font.size": 11}):
        assert mpl.rcParams["axes.grid"] is True
        assert mpl.rcParams["font.size"] == 11
    with pytest.raises(ValueError, match="Unknown Matplotlib rcParam"):
        rp.use("ieee", overrides={"not.a.real.param": 1})


def test_required_dimension_failure_and_report_helpers() -> None:
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.plot([0, 1], [0, 1])
    report = rp.validate_figure(fig, venue="nature", width="single")
    assert not report.passed
    assert any(check.rule_id == "figure.width.single" for check in report.failures)
    assert report.to_dict()["passed"] is False
    assert "ResearchPlot validation" in str(report)


def test_required_recommended_and_inferred_semantics() -> None:
    with rp.use("nature") as style:
        fig, ax = style.subplots()
        ax.plot([0, 1], [0, 1], linewidth=0.1)
        ax.set_title("Move me to the caption")
        report = style.validate(fig)
    status = {check.rule_id: check.status for check in report.checks}
    assert status["line.width.min"] is rp.CheckStatus.FAIL
    assert status["figure.title.prohibited"] is rp.CheckStatus.WARN

    with rp.use("ieee") as style:
        fig, ax = style.subplots()
        ax.plot([0, 1], [0, 1], linewidth=0.1)
        inferred = style.validate(fig)
    line_check = next(check for check in inferred.checks if check.rule_id == "line.width.min")
    assert line_check.status is rp.CheckStatus.INFO
    assert inferred.passed


def test_invalid_width_and_artwork_are_intentional_errors() -> None:
    with pytest.raises(ValueError, match="Available widths"):
        rp.use("nature", width="triple")
    fig, _ = plt.subplots()
    with pytest.raises(ValueError, match="Unknown artwork"):
        rp.validate_figure(fig, venue="nature", width="single", artwork="photo-ish")


def test_unobservable_artist_checks_are_skipped_not_passed() -> None:
    with rp.use("nature") as style:
        fig = style.figure()
        report = style.validate(fig)
    statuses = {check.rule_id: check.status for check in report.checks}
    assert statuses["font.family"] is rp.CheckStatus.SKIP
    assert statuses["font.size.min"] is rp.CheckStatus.SKIP
    assert statuses["font.size.max"] is rp.CheckStatus.SKIP
    assert statuses["line.width.min"] is rp.CheckStatus.SKIP
    assert statuses["export.formats.vector"] is rp.CheckStatus.SKIP
