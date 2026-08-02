from __future__ import annotations

import matplotlib as mpl
import pytest
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

import researchplot as rp
from researchplot import figure_inspector
from researchplot import style as style_module


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
        "plos-biology",
    ],
)
def test_target_style_creates_exact_default_width(profile_id: str) -> None:
    profile = rp.resolve_profile(f"{profile_id}@2026.08.0")
    selected = rp.target(profile)
    with selected.style() as style:
        fig, _ = style.subplots()
    assert fig.get_size_inches()[0] * 25.4 == pytest.approx(profile.width_mm(), abs=1e-8)
    plt.close(fig)


def test_context_restores_rcparams_after_success_and_exception() -> None:
    selected = rp.target("nature@2026.08.0")
    original = mpl.rcParams["font.size"]
    with selected.style():
        assert mpl.rcParams["font.size"] == 7
    assert mpl.rcParams["font.size"] == original

    with pytest.raises(RuntimeError, match="boom"):
        with selected.style():
            assert mpl.rcParams["font.size"] == 7
            raise RuntimeError("boom")
    assert mpl.rcParams["font.size"] == original


def test_overrides_take_precedence_and_are_validated() -> None:
    selected = rp.target("ieee-journal@2026.08.0")
    with selected.style(overrides={"axes.grid": True, "font.size": 11}):
        assert mpl.rcParams["axes.grid"] is True
        assert mpl.rcParams["font.size"] == 11
    with pytest.raises(ValueError, match="Unknown Matplotlib rcParam"):
        selected.style(overrides={"not.a.real.param": 1})


def test_missing_required_font_uses_an_honest_installed_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback_path = style_module.font_manager.findfont("DejaVu Sans")
    monkeypatch.setattr(style_module.font_manager.fontManager, "ttflist", [])
    monkeypatch.setattr(
        style_module.font_manager,
        "findfont",
        lambda *_args, **_kwargs: fallback_path,
    )

    family = style_module._font_family(rp.resolve_profile("plos-biology@2026.08.0"))

    assert family == "DejaVu Sans"
    assert family not in {"Arial", "Times", "Symbol"}


def test_declaring_an_unavailable_exact_font_does_not_bypass_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback_path = figure_inspector.font_manager.findfont("DejaVu Sans")
    monkeypatch.setattr(
        figure_inspector.font_manager,
        "findfont",
        lambda *_args, **_kwargs: fallback_path,
    )
    selected = rp.target("plos-biology@2026.08.0")
    with selected.style(overrides={"font.family": "Arial"}) as style:
        fig, ax = style.subplots()
        ax.set_xlabel("Input")
        report = selected.validate(fig)

    finding = next(item for item in report.findings if item.rule_id == "font.family")
    assert finding.outcome is rp.Outcome.FAIL
    assert finding.observed == ("DejaVu Sans",)
    plt.close(fig)


def test_required_dimension_failure_and_report_helpers() -> None:
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.plot([0, 1], [0, 1])
    report = rp.target("nature@2026.08.0", width="single").validate(fig)
    assert report.verdict is rp.Verdict.NON_COMPLIANT
    assert any(finding.rule_id == "figure.width.single" for finding in report.failures)
    assert report.to_dict()["verdict"] == "non_compliant"
    assert "ResearchPlot:" in str(report)
    plt.close(fig)


def test_required_recommended_and_inferred_semantics() -> None:
    selected = rp.target("nature@2026.08.0")
    with selected.style() as style:
        fig, ax = style.subplots()
        ax.plot([0, 1], [0, 1], linewidth=0.1)
        ax.set_title("Move me to the caption")
        report = selected.validate(fig)
    outcomes = {finding.rule_id: finding.outcome for finding in report.findings}
    assert outcomes["line.width.min"] is rp.Outcome.FAIL
    assert outcomes["figure.title.prohibited"] is rp.Outcome.FAIL
    assert any(item.rule_id == "figure.title.prohibited" for item in report.warnings)
    plt.close(fig)


def test_required_line_width_includes_collection_based_lines() -> None:
    selected = rp.target("nature@2026.08.0")
    with selected.style() as style:
        fig, ax = style.subplots()
        ax.add_collection(LineCollection([[(0, 0), (1, 1)]], linewidths=[0.01]))
        ax.set(xlim=(0, 1), ylim=(0, 1))
        report = selected.validate(fig)

    finding = next(item for item in report.findings if item.rule_id == "line.width.min")
    assert finding.outcome is rp.Outcome.FAIL
    assert finding.observed == pytest.approx(0.01)
    plt.close(fig)

    inferred_target = rp.target("ieee-journal@2026.08.0")
    with inferred_target.style() as style:
        fig, ax = style.subplots()
        ax.plot([0, 1], [0, 1], linewidth=0.1)
        inferred = inferred_target.validate(fig)
    line_finding = next(
        finding for finding in inferred.findings if finding.rule_id == "line.width.min"
    )
    assert line_finding.level is rp.RuleLevel.INFERRED
    assert line_finding.outcome is rp.Outcome.FAIL
    assert inferred.verdict is rp.Verdict.COMPLIANT
    plt.close(fig)


def test_invalid_width_and_widthless_styling_are_intentional_errors() -> None:
    with pytest.raises(ValueError, match="Available widths"):
        rp.target("nature@2026.08.0", width="triple")
    with pytest.raises(ValueError, match="non-empty"):
        rp.target("nature@2026.08.0", width="")
    with pytest.raises(TypeError, match="string or null"):
        rp.target("nature@2026.08.0", width=0)  # type: ignore[arg-type]
    bundle_only = rp.target("acm-acmart@2026.08.0")
    assert bundle_only.width is None
    with pytest.raises(ValueError, match="does not define a physical figure width"):
        bundle_only.style()


def test_unobservable_artist_checks_are_skipped_not_passed() -> None:
    selected = rp.target("nature@2026.08.0")
    with selected.style() as style:
        fig = style.figure()
        report = selected.validate(fig)
    outcomes = {finding.rule_id: finding.outcome for finding in report.findings}
    assert outcomes["font.family"] is rp.Outcome.SKIP
    assert outcomes["font.size.min"] is rp.Outcome.SKIP
    assert outcomes["font.size.max"] is rp.Outcome.SKIP
    assert outcomes["line.width.min"] is rp.Outcome.SKIP
    assert report.verdict is rp.Verdict.INDETERMINATE
    plt.close(fig)
