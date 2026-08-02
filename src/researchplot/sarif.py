"""SARIF 2.1.0 serialization for CI and GitHub code scanning."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .compliance import Outcome, Report
from .models import RuleLevel


def reports_to_sarif(items: Iterable[tuple[str | Path, Report]]) -> dict[str, object]:
    """Convert artifact reports into a deterministic SARIF document."""

    values = tuple(items)
    rules: dict[str, dict[str, object]] = {}
    results: list[dict[str, object]] = []
    for path, report in values:
        artifact = Path(path).as_posix()
        for finding in report.findings:
            if finding.outcome is Outcome.PASS:
                continue
            rule_descriptor: dict[str, object] = {
                "id": finding.rule_id,
                "name": finding.rule_id.replace(".", "_"),
                "shortDescription": {"text": finding.message.split(".", 1)[0]},
                "properties": {
                    "ruleLevel": finding.level.value,
                    "verification": finding.verification,
                    "sourceUrls": list(finding.source_urls),
                },
            }
            if finding.source_urls:
                rule_descriptor["helpUri"] = finding.source_urls[0]
            rules.setdefault(
                finding.rule_id,
                rule_descriptor,
            )
            if finding.outcome is Outcome.SKIP:
                level = "warning" if finding.level is RuleLevel.REQUIRED else "note"
            elif finding.level is RuleLevel.REQUIRED:
                level = "error"
            elif finding.level is RuleLevel.RECOMMENDED:
                level = "warning"
            else:
                level = "note"
            result: dict[str, object] = {
                "ruleId": finding.rule_id,
                "level": level,
                "message": {"text": finding.message},
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {"uri": artifact},
                        }
                    }
                ],
                "properties": {
                    "outcome": finding.outcome.value,
                    "profile": report.profile,
                    "observed": finding.observed,
                    "expected": finding.expected,
                    "suggestion": finding.suggestion,
                },
            }
            results.append(result)
    return {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "ResearchPlot",
                        "informationUri": "https://devrajsinh-jhala.github.io/ResearchPlot/",
                        "rules": [rules[key] for key in sorted(rules)],
                    }
                },
                "results": results,
            }
        ],
    }
