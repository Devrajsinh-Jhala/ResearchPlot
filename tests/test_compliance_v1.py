from __future__ import annotations

from dataclasses import replace

import pytest

from researchplot.compliance import (
    Finding,
    Outcome,
    Policy,
    Report,
    RuleEngine,
    TargetContext,
    Verdict,
)
from researchplot.models import (
    ConstraintOperator,
    ContentKind,
    FigureRole,
    OutputFormat,
    RuleApplicability,
    RuleConstraint,
    RuleLevel,
    RulePhase,
    SourceRef,
    VenueKind,
    VenueProfile,
    VenueRule,
    VerificationMode,
)
from researchplot.observations import Observation, ObservationSet


def _profile(*rules: VenueRule) -> VenueProfile:
    source = SourceRef(
        "official",
        "Official guide",
        "https://example.org/guide",
        "Figures > Size",
        "2026-08-02",
        "2026-08-02",
    )
    width = VenueRule(
        "figure.width.single",
        "figure.width_mm",
        RuleConstraint(ConstraintOperator.APPROX, 89.0, "mm", 0.5),
        RuleApplicability(widths=("single",)),
        VerificationMode.AUTOMATED,
        RuleLevel.REQUIRED,
        ("official",),
        "Single-column width.",
        (RulePhase.LIVE, RulePhase.FILE),
    )
    return VenueProfile(
        id="example",
        name="Example",
        kind=VenueKind.JOURNAL,
        year=None,
        aliases=(),
        scope="Tests",
        default_width="single",
        verified_on="2026-08-02",
        sources=(source,),
        rules=(width, *rules),
        revision="2026.08.0",
        effective_date="2026-08-02",
        digest="abc",
    )


def _rule(
    rule_id: str,
    probe: str,
    operator: ConstraintOperator,
    value: object,
    *,
    level: RuleLevel = RuleLevel.REQUIRED,
    verification: VerificationMode = VerificationMode.AUTOMATED,
    phases: tuple[RulePhase, ...] = (RulePhase.LIVE,),
    applies_to: RuleApplicability | None = None,
) -> VenueRule:
    return VenueRule(
        rule_id,
        probe,
        RuleConstraint(operator, value),  # type: ignore[arg-type]
        applies_to or RuleApplicability(),
        verification,
        level,
        ("official",),
        f"Rule {rule_id}.",
        phases,
    )


def test_required_skip_is_indeterminate_not_passed() -> None:
    profile = _profile()
    report = RuleEngine().evaluate(
        profile,
        ObservationSet((Observation("figure.width_mm", available=False),)),
        TargetContext("main", "single", "data_visualization"),
        phase="live",
    )
    assert report.verdict is Verdict.INDETERMINATE
    assert not report.passed
    assert report.blocks(Policy.COMPLETE)
    assert not report.blocks(Policy.VIOLATIONS)


def test_failure_severity_is_separate_from_outcome() -> None:
    recommendation = _rule(
        "accessibility.contrast",
        "accessibility.text_contrast.min",
        ConstraintOperator.GTE,
        4.5,
        level=RuleLevel.RECOMMENDED,
    )
    profile = _profile(recommendation)
    observations = ObservationSet(
        (
            Observation("figure.width_mm", 89.0),
            Observation("accessibility.text_contrast.min", 2.0),
        )
    )
    report = RuleEngine().evaluate(
        profile,
        observations,
        TargetContext("main", "single", "data_visualization"),
        phase="live",
    )
    finding = next(item for item in report.findings if item.rule_id == recommendation.id)
    assert finding.outcome is Outcome.FAIL
    assert finding.level is RuleLevel.RECOMMENDED
    assert report.verdict is Verdict.COMPLIANT
    assert report.warnings == (finding,)


def test_applicability_phase_and_manual_attestation() -> None:
    alt_text = _rule(
        "metadata.alt_text",
        "metadata.alt_text.present",
        ConstraintOperator.EQ,
        True,
        verification=VerificationMode.MANUAL,
        phases=(RulePhase.BUNDLE,),
        applies_to=RuleApplicability(roles=(FigureRole.MAIN,)),
    )
    file_format = _rule(
        "artifact.format",
        "artifact.format",
        ConstraintOperator.IN,
        ("pdf", "eps"),
        phases=(RulePhase.FILE,),
        applies_to=RuleApplicability(
            content_kinds=(ContentKind.LINE_ART,),
            output_formats=(OutputFormat.PDF, OutputFormat.EPS),
        ),
    )
    profile = _profile(alt_text, file_format)
    target = TargetContext("main", "single", "line_art", "pdf")

    live = RuleEngine().evaluate(
        profile,
        ObservationSet((Observation("figure.width_mm", 89.0),)),
        target,
        phase="live",
    )
    assert {item.rule_id for item in live.findings} == {"figure.width.single"}

    bundle = RuleEngine().evaluate(
        profile,
        ObservationSet((Observation("metadata.alt_text.present", True, phase="bundle"),)),
        target,
        phase="bundle",
        attestations={alt_text.id: "Alt text supplied in the manuscript."},
    )
    assert bundle.verdict is Verdict.COMPLIANT
    assert bundle.findings[0].outcome is Outcome.PASS

    blank_attestation = RuleEngine().evaluate(
        profile,
        ObservationSet((Observation("metadata.alt_text.present", True, phase="bundle"),)),
        target,
        phase="bundle",
        attestations={alt_text.id: "   "},
    )
    assert blank_attestation.verdict is Verdict.INDETERMINATE
    assert blank_attestation.findings[0].outcome is Outcome.SKIP


def test_unsupported_verification_always_skips_available_observations() -> None:
    unsupported = _rule(
        "visual.human_review",
        "visual.human_review",
        ConstraintOperator.EQ,
        True,
        verification=VerificationMode.UNSUPPORTED,
    )
    profile = _profile(unsupported)
    report = RuleEngine().evaluate(
        profile,
        ObservationSet(
            (
                Observation("figure.width_mm", 89.0),
                Observation("visual.human_review", True),
            )
        ),
        TargetContext("main", "single", "data_visualization"),
        phase="live",
    )

    finding = next(item for item in report.findings if item.rule_id == unsupported.id)
    assert finding.outcome is Outcome.SKIP
    assert report.verdict is Verdict.INDETERMINATE


def test_invalid_phase_is_rejected_instead_of_passing_empty_report() -> None:
    with pytest.raises(ValueError, match="typo"):
        RuleEngine().evaluate(
            _profile(),
            ObservationSet((Observation("figure.width_mm", 89.0),)),
            TargetContext("main", "single", "data_visualization"),
            phase="typo",
        )


def test_observation_from_another_phase_cannot_satisfy_a_rule() -> None:
    file_rule = _rule(
        "artifact.format",
        "artifact.format",
        ConstraintOperator.EQ,
        "pdf",
        phases=(RulePhase.FILE,),
    )
    report = RuleEngine().evaluate(
        _profile(file_rule),
        ObservationSet((Observation("artifact.format", "pdf", phase="live"),)),
        TargetContext("main", "single", "line_art", "pdf"),
        phase="file",
    )

    finding = next(item for item in report.findings if item.rule_id == file_rule.id)
    assert finding.outcome is Outcome.SKIP
    assert "during 'live', not 'file'" in finding.message


def test_combining_file_reports_preserves_same_rule_for_each_artifact() -> None:
    context = TargetContext("main", "single", "line_art", "pdf")
    first = Report(
        "example@2026.08.0",
        context,
        (
            Finding(
                "figure.width.single",
                Outcome.PASS,
                RuleLevel.REQUIRED,
                "file",
                "First artifact width.",
                artifact="figure-a.pdf",
            ),
        ),
    )
    second = replace(
        first,
        findings=(
            replace(
                first.findings[0],
                message="Second artifact width.",
                artifact="figure-b.pdf",
            ),
        ),
    )

    combined = Report.combine((first, second))

    assert [finding.artifact for finding in combined.findings] == [
        "figure-a.pdf",
        "figure-b.pdf",
    ]


def test_combining_conflicting_evidence_keeps_the_strongest_outcome() -> None:
    context = TargetContext("main", "single", "line_art", "pdf")
    failed = Finding(
        "font.pdf.embedding.required",
        Outcome.FAIL,
        RuleLevel.REQUIRED,
        "file",
        "An unembedded font was found.",
        artifact="figure.pdf",
    )
    passed = replace(failed, outcome=Outcome.PASS, message="All fonts were embedded.")
    first = Report("example@2026.08.0", context, (failed,))
    second = Report("example@2026.08.0", context, (passed,))

    combined = Report.combine((first, second))
    reverse = Report.combine((second, first))

    assert combined.findings == (failed,)
    assert reverse.findings == (failed,)
    assert combined.verdict is Verdict.NON_COMPLIANT


def test_report_dict_is_deterministic() -> None:
    profile = _profile()
    observations = ObservationSet((Observation("figure.width_mm", 89.0),))
    context = TargetContext("main", "single", "data_visualization")
    first = RuleEngine().evaluate(profile, observations, context, phase="live")
    second = replace(first)
    assert first.to_dict() == second.to_dict()
    assert "generated_at" not in first.to_dict()


def test_duplicate_observations_are_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        ObservationSet((Observation("a", 1), Observation("a", 2)))
