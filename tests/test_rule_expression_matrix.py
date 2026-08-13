from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from researchplot.api_types import EvidenceConfidence
from researchplot.compliance import Outcome, RuleEngine, TargetContext, Verdict
from researchplot.models import (
    AggregateExpression,
    AllExpression,
    AnyExpression,
    ConstraintOperator,
    NotExpression,
    ProbeExpression,
    QuantifierExpression,
    RuleApplicability,
    RuleConstraint,
    RuleLevel,
    RulePhase,
    VenueRule,
    VerificationMode,
)
from researchplot.observations import Observation, ObservationSet
from researchplot.rule_expressions import (
    evaluate_constraint,
    evaluate_expression,
    expression_probes,
)
from researchplot.units import Quantity


@pytest.mark.parametrize(
    ("operator", "expected", "observed", "result"),
    [
        (ConstraintOperator.EQ, "pdf", "pdf", True),
        (ConstraintOperator.NE, "eps", "pdf", True),
        (ConstraintOperator.GT, 4, 5, True),
        (ConstraintOperator.GTE, 5, 5, True),
        (ConstraintOperator.LT, 6, 5, True),
        (ConstraintOperator.LTE, 5, 5, True),
        (ConstraintOperator.IN, ("pdf", "eps"), "pdf", True),
        (ConstraintOperator.NOT_IN, ("png", "jpg"), "pdf", True),
        (ConstraintOperator.BETWEEN, (4.0, 8.0), 6.0, True),
        (ConstraintOperator.SUBSET, ("CMYK", "RGB"), ("RGB",), True),
        (ConstraintOperator.CONTAINS, "embedded", ("subset", "embedded"), True),
        (ConstraintOperator.NOT_CONTAINS, "script", ("font", "image"), True),
        (ConstraintOperator.APPROX, 89.0, 89.4, True),
        (ConstraintOperator.EXISTS, True, "value", True),
        (ConstraintOperator.EXISTS, False, None, True),
        (ConstraintOperator.PATTERN, r"fig-[0-9]+", "fig-12", True),
        (ConstraintOperator.REQUIRED, True, "caption", True),
        (ConstraintOperator.PROHIBITED, True, False, True),
        (ConstraintOperator.PROHIBITED, "JavaScript", "none", True),
    ],
)
def test_constraint_operator_matrix(
    operator: ConstraintOperator, expected: object, observed: object, result: bool
) -> None:
    tolerance = 0.5 if operator is ConstraintOperator.APPROX else None
    constraint = RuleConstraint(operator, expected, tolerance=tolerance)  # type: ignore[arg-type]
    assert evaluate_constraint(observed, constraint) is result


def test_constraint_units_and_invalid_collection_shapes() -> None:
    millimetres = RuleConstraint(ConstraintOperator.APPROX, 25.4, "mm", 0.01)
    assert evaluate_constraint(Quantity(1.0, "in"), millimetres)

    with pytest.raises(TypeError, match="two bounds"):
        evaluate_constraint(2, RuleConstraint(ConstraintOperator.BETWEEN, (1,)))
    with pytest.raises(TypeError, match="numeric bounds"):
        evaluate_constraint(2, RuleConstraint(ConstraintOperator.BETWEEN, ("a", "b")))
    with pytest.raises(TypeError, match="numeric observation"):
        evaluate_constraint("2", RuleConstraint(ConstraintOperator.BETWEEN, (1, 3)))
    with pytest.raises(TypeError, match="collection values"):
        evaluate_constraint("pdf", RuleConstraint(ConstraintOperator.SUBSET, ("pdf",)))
    with pytest.raises(TypeError, match="numeric observation"):
        evaluate_constraint("89", RuleConstraint(ConstraintOperator.APPROX, 89.0))
    with pytest.raises(TypeError, match="numeric expected"):
        evaluate_constraint(89, RuleConstraint(ConstraintOperator.APPROX, "89"))
    with pytest.raises(TypeError, match="boolean expected"):
        evaluate_constraint("x", RuleConstraint(ConstraintOperator.EXISTS, "yes"))


@pytest.mark.parametrize(
    ("constraint", "observed", "message"),
    [
        (RuleConstraint(ConstraintOperator.PATTERN, 12), "fig-1", "string values"),
        (RuleConstraint(ConstraintOperator.PATTERN, "x" * 257), "x", "limited"),
        (RuleConstraint(ConstraintOperator.PATTERN, "x"), "x" * 4097, "limited"),
        (RuleConstraint(ConstraintOperator.PATTERN, "["), "x", "Invalid pattern"),
    ],
)
def test_pattern_constraints_are_bounded_and_safe(
    constraint: RuleConstraint, observed: object, message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        evaluate_constraint(observed, constraint)


def test_three_valued_composition_and_probe_collection() -> None:
    width = ProbeExpression(
        "artifact.width_mm",
        RuleConstraint(ConstraintOperator.APPROX, 89.0, "mm", 0.5),
    )
    title = ProbeExpression("figure.has_title", RuleConstraint(ConstraintOperator.PROHIBITED, True))
    duplicate_width = ProbeExpression(
        "artifact.width_mm", RuleConstraint(ConstraintOperator.GT, 0.0, "mm")
    )
    combined = AllExpression((width, AnyExpression((title, duplicate_width))))

    assert expression_probes(combined) == ("artifact.width_mm", "figure.has_title")
    assert evaluate_expression(combined, {}) is None
    assert evaluate_expression(combined, {"artifact.width_mm": Quantity(1, "in")}) is False
    assert (
        evaluate_expression(
            combined,
            {
                "artifact.width_mm": Quantity(3.5039, "in"),
                "figure.has_title": False,
            },
        )
        is True
    )
    assert evaluate_expression(NotExpression(title), {}) is None
    assert evaluate_expression(AnyExpression((title, width)), {}) is None
    assert (
        evaluate_expression(
            AnyExpression((title, width)),
            {"figure.has_title": True, "artifact.width_mm": Quantity(89, "mm")},
        )
        is True
    )


def test_quantifiers_and_aggregates_reject_unknown_evidence_shapes() -> None:
    positive = RuleConstraint(ConstraintOperator.GT, 0)
    all_positive = QuantifierExpression("all", "series.values", positive)
    any_positive = QuantifierExpression("any", "series.values", positive)
    minimum = AggregateExpression("minimum", "series.values", positive)
    maximum = AggregateExpression("maximum", "series.values", positive)
    count = AggregateExpression("count", "series.values", RuleConstraint(ConstraintOperator.GTE, 2))

    assert evaluate_expression(all_positive, {}) is None
    assert evaluate_expression(all_positive, {"series.values": ()}) is False
    assert evaluate_expression(all_positive, {"series.values": (1, 2)}) is True
    assert evaluate_expression(any_positive, {"series.values": (-1, 2)}) is True
    assert evaluate_expression(minimum, {"series.values": ()}) is None
    assert evaluate_expression(minimum, {"series.values": (1, 2)}) is True
    assert evaluate_expression(maximum, {"series.values": (-2, -1)}) is False
    assert evaluate_expression(count, {"series.values": (1, 2)}) is True

    for expression in (all_positive, count):
        with pytest.raises(TypeError, match="collection observation"):
            evaluate_expression(expression, {"series.values": Quantity(1, "mm")})
    with pytest.raises(TypeError, match="numeric observations"):
        evaluate_expression(minimum, {"series.values": (1, "two")})


@dataclass(frozen=True)
class _UnknownExpression:
    expressions: tuple[()] = ()


def test_unknown_expression_types_fail_closed() -> None:
    with pytest.raises(TypeError, match="Unsupported expression type"):
        evaluate_expression(_UnknownExpression(), {})  # type: ignore[arg-type]


def test_rule_engine_expression_coverage_and_manual_attestation_metadata() -> None:
    profile = __import__("researchplot").resolve_profile("nature@2026.08.0")
    source_rule = profile.rules[0]
    expression = AllExpression(
        (
            ProbeExpression(
                "artifact.width_mm",
                RuleConstraint(ConstraintOperator.APPROX, 89.0, "mm", 0.5),
            ),
            ProbeExpression(
                "artifact.active_content",
                RuleConstraint(ConstraintOperator.PROHIBITED, True),
            ),
        )
    )
    expression_rule = replace(
        source_rule,
        id="artifact.expression.safe",
        probe="artifact.width_mm",
        constraint=RuleConstraint(ConstraintOperator.EQ, True),
        expression=expression,
        phases=(RulePhase.FILE,),
    )
    manual_rule = VenueRule(
        "metadata.review.manual",
        "metadata.review.present",
        RuleConstraint(ConstraintOperator.EQ, True),
        RuleApplicability(),
        VerificationMode.MANUAL,
        RuleLevel.REQUIRED,
        source_rule.source_ids,
        "A human reviewed the figure.",
        (RulePhase.FILE,),
    )
    selected = replace(profile, rules=(expression_rule, manual_rule))
    context = TargetContext("main", "single", "data_visualization", "pdf")
    observations = ObservationSet(
        (
            Observation("artifact.width_mm", 89.0, unit="mm", phase="file"),
            Observation(
                "artifact.active_content",
                False,
                phase="file",
                confidence=EvidenceConfidence.HEURISTIC,
            ),
        )
    )
    unresolved = RuleEngine().evaluate(selected, observations, context, phase="file")
    assert unresolved.verdict is Verdict.INDETERMINATE
    assert all(finding.outcome is Outcome.SKIP for finding in unresolved.findings)
    assert "heuristic probes" in unresolved.findings[0].message

    attestations = {
        manual_rule.id: {
            "reviewer": "Reviewer One",
            "date": "2026-08-03",
            "rationale": "Checked against the submission proof.",
            "evidence": "proof.pdf#page=2",
        }
    }
    complete = RuleEngine().evaluate(
        selected,
        ObservationSet(
            (
                Observation("artifact.width_mm", 89.0, unit="mm", phase="file"),
                Observation("artifact.active_content", False, phase="file"),
            )
        ),
        context,
        phase="file",
        attestations=attestations,
    )
    assert complete.verdict is Verdict.COMPLIANT
    assert complete.findings[1].observed == attestations[manual_rule.id]
