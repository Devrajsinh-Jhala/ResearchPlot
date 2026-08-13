"""Typed, bounded evaluation for schema-v3 declarative rule expressions."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import TypeAlias

from .models import (
    AggregateExpression,
    AllExpression,
    AnyExpression,
    ConstraintOperator,
    NotExpression,
    ProbeExpression,
    QuantifierExpression,
    RuleConstraint,
    RuleExpression,
)
from .units import Quantity, convert_value

Observation: TypeAlias = object | Quantity
ExpressionResult: TypeAlias = bool | None


def _coerce_observed(value: Observation, constraint: RuleConstraint) -> object:
    if isinstance(value, Quantity):
        if constraint.unit is None:
            return value.value
        return convert_value(value.value, value.unit, constraint.unit)
    return value


def evaluate_constraint(value: Observation, constraint: RuleConstraint) -> bool:
    """Evaluate one constraint against an observed value.

    Unit-bearing observations are converted to the rule unit before comparison.
    Missing observations are handled by :func:`evaluate_expression`, not here.
    """

    observed = _coerce_observed(value, constraint)
    expected = constraint.value
    operator = constraint.operator
    if operator is ConstraintOperator.EQ:
        return observed == expected
    if operator is ConstraintOperator.NE:
        return observed != expected
    if operator is ConstraintOperator.GT:
        return bool(observed > expected)  # type: ignore[operator]
    if operator is ConstraintOperator.GTE:
        return bool(observed >= expected)  # type: ignore[operator]
    if operator is ConstraintOperator.LT:
        return bool(observed < expected)  # type: ignore[operator]
    if operator is ConstraintOperator.LTE:
        return bool(observed <= expected)  # type: ignore[operator]
    if operator is ConstraintOperator.IN:
        return observed in expected  # type: ignore[operator]
    if operator is ConstraintOperator.NOT_IN:
        return observed not in expected  # type: ignore[operator]
    if operator is ConstraintOperator.BETWEEN:
        if not isinstance(expected, tuple) or len(expected) != 2:
            raise TypeError("A between constraint requires two bounds.")
        lower, upper = expected
        if not isinstance(lower, (int, float)) or not isinstance(upper, (int, float)):
            raise TypeError("A between constraint requires numeric bounds.")
        if not isinstance(observed, (int, float)) or isinstance(observed, bool):
            raise TypeError("A between constraint requires a numeric observation.")
        return lower <= observed <= upper
    if operator is ConstraintOperator.SUBSET:
        if not isinstance(expected, tuple) or not isinstance(
            observed, (tuple, list, set, frozenset)
        ):
            raise TypeError("A subset constraint requires collection values.")
        return set(observed).issubset(set(expected))
    if operator is ConstraintOperator.CONTAINS:
        return expected in observed  # type: ignore[operator]
    if operator is ConstraintOperator.NOT_CONTAINS:
        return expected not in observed  # type: ignore[operator]
    if operator is ConstraintOperator.APPROX:
        tolerance = constraint.tolerance if constraint.tolerance is not None else 0.0
        if not isinstance(observed, (int, float)) or isinstance(observed, bool):
            raise TypeError("An approx constraint requires a numeric observation.")
        if not isinstance(expected, (int, float)) or isinstance(expected, bool):
            raise TypeError("An approx constraint requires a numeric expected value.")
        return abs(float(observed) - float(expected)) <= tolerance
    if operator is ConstraintOperator.EXISTS:
        if not isinstance(expected, bool):
            raise TypeError("An exists constraint requires a boolean expected value.")
        return (observed is not None) is expected
    if operator is ConstraintOperator.PATTERN:
        if not isinstance(expected, str) or not isinstance(observed, str):
            raise TypeError("A pattern constraint requires string values.")
        if len(expected) > 256 or len(observed) > 4096:
            raise ValueError("Pattern checks are limited to 256 pattern and 4096 input characters.")
        try:
            return re.fullmatch(expected, observed) is not None
        except re.error as exc:
            raise ValueError(f"Invalid pattern: {exc}") from exc
    if operator is ConstraintOperator.REQUIRED:
        return bool(observed)
    if operator is ConstraintOperator.PROHIBITED:
        return not bool(observed) if expected is True else observed != expected
    raise AssertionError(f"Unhandled constraint operator: {operator}")


def evaluate_expression(
    expression: RuleExpression,
    observations: Mapping[str, Observation],
) -> ExpressionResult:
    """Evaluate an expression using three-valued logic.

    ``None`` means one or more required probes were not observed. An ``all``
    expression still fails when any known child fails; an ``any`` expression
    still succeeds when any known child succeeds.
    """

    if isinstance(expression, ProbeExpression):
        if expression.probe not in observations:
            return None
        return evaluate_constraint(observations[expression.probe], expression.constraint)
    if isinstance(expression, QuantifierExpression):
        if expression.probe not in observations:
            return None
        observed = observations[expression.probe]
        if isinstance(observed, Quantity) or not isinstance(
            observed, (tuple, list, set, frozenset)
        ):
            raise TypeError("A quantifier requires a collection observation.")
        values = tuple(observed)
        if not values:
            return False
        results = tuple(evaluate_constraint(item, expression.constraint) for item in values)
        return all(results) if expression.quantifier == "all" else any(results)
    if isinstance(expression, AggregateExpression):
        if expression.probe not in observations:
            return None
        observed = observations[expression.probe]
        if isinstance(observed, Quantity) or not isinstance(
            observed, (tuple, list, set, frozenset)
        ):
            raise TypeError("An aggregate requires a collection observation.")
        values = tuple(observed)
        if expression.aggregate == "count":
            aggregate: object = len(values)
        else:
            if not values:
                return None
            if any(not isinstance(item, (int, float)) or isinstance(item, bool) for item in values):
                raise TypeError("Minimum and maximum aggregates require numeric observations.")
            aggregate = min(values) if expression.aggregate == "minimum" else max(values)
        return evaluate_constraint(aggregate, expression.constraint)
    if isinstance(expression, NotExpression):
        result = evaluate_expression(expression.expression, observations)
        return None if result is None else not result
    composed_results: tuple[ExpressionResult, ...] = tuple(
        evaluate_expression(item, observations) for item in expression.expressions
    )
    if isinstance(expression, AllExpression):
        if False in composed_results:
            return False
        return None if None in composed_results else True
    if isinstance(expression, AnyExpression):
        if True in composed_results:
            return True
        return None if None in composed_results else False
    raise TypeError(f"Unsupported expression type: {type(expression).__name__}")


def expression_probes(expression: RuleExpression) -> tuple[str, ...]:
    """Return probe IDs in deterministic encounter order."""

    if isinstance(expression, ProbeExpression):
        return (expression.probe,)
    if isinstance(expression, (QuantifierExpression, AggregateExpression)):
        return (expression.probe,)
    if isinstance(expression, NotExpression):
        return expression_probes(expression.expression)
    result: list[str] = []
    for child in expression.expressions:
        for probe in expression_probes(child):
            if probe not in result:
                result.append(probe)
    return tuple(result)
