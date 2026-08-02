from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

from researchplot import export_manifest_schema, report_schema, submission_manifest_schema
from researchplot.compliance import Finding, Outcome, Report, TargetContext
from researchplot.models import RuleLevel

ROOT = Path(__file__).parents[1]
SCHEMA_DIRECTORY = ROOT / "schemas"


def _load_schema(name: str) -> dict[str, object]:
    return json.loads((SCHEMA_DIRECTORY / name).read_text(encoding="utf-8"))


def _report_payload() -> dict[str, object]:
    report = Report(
        profile="nature@2026.08.0",
        target=TargetContext(
            role="main",
            width="single",
            content="line_art",
            output_format="pdf",
        ),
        findings=(
            Finding(
                rule_id="figure.width.single",
                outcome=Outcome.PASS,
                level=RuleLevel.REQUIRED,
                phase="file",
                verification="automated",
                message="The artifact has the required single-column width.",
                observed=89.0,
                expected=89.0,
                source_urls=("https://example.com/official-guide",),
            ),
        ),
    )
    return report.to_dict()


def _validators() -> tuple[Draft202012Validator, Draft202012Validator, Draft202012Validator]:
    report_schema = _load_schema("report.schema.json")
    export_schema = _load_schema("export-manifest.schema.json")
    manifest_schema = _load_schema("submission-manifest.schema.json")
    Draft202012Validator.check_schema(report_schema)
    Draft202012Validator.check_schema(export_schema)
    Draft202012Validator.check_schema(manifest_schema)
    return (
        Draft202012Validator(report_schema),
        Draft202012Validator(export_schema),
        Draft202012Validator(manifest_schema),
    )


def test_contract_schemas_are_valid_draft_2020_12() -> None:
    report_schema = _load_schema("report.schema.json")
    export_schema = _load_schema("export-manifest.schema.json")
    manifest_schema = _load_schema("submission-manifest.schema.json")

    Draft202012Validator.check_schema(report_schema)
    Draft202012Validator.check_schema(export_schema)
    Draft202012Validator.check_schema(manifest_schema)
    assert report_schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert export_schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert manifest_schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"

    def references(value: object) -> list[str]:
        if isinstance(value, dict):
            found = [str(value["$ref"])] if "$ref" in value else []
            for child in value.values():
                found.extend(references(child))
            return found
        if isinstance(value, list):
            return [item for child in value for item in references(child)]
        return []

    assert all(reference.startswith("#/") for reference in references(export_schema))
    assert all(reference.startswith("#/") for reference in references(manifest_schema))


def test_contract_schemas_are_bundled_and_return_independent_copies() -> None:
    report = report_schema()
    export = export_manifest_schema()
    manifest = submission_manifest_schema()
    assert report["$id"] == _load_schema("report.schema.json")["$id"]
    assert export["$id"] == _load_schema("export-manifest.schema.json")["$id"]
    assert manifest["$id"] == _load_schema("submission-manifest.schema.json")["$id"]
    report["title"] = "changed"
    assert report_schema()["title"] == "ResearchPlot compliance report"


def test_report_to_dict_matches_public_report_schema() -> None:
    report_validator, _, _ = _validators()

    report_validator.validate(_report_payload())


def test_single_export_manifest_matches_public_export_schema() -> None:
    _, export_validator, _ = _validators()
    report = _report_payload()
    export_manifest = {
        "schema_version": 1,
        "researchplot_version": "1.0.0",
        "profile": "nature@2026.08.0",
        "profile_digest": "a" * 64,
        "sources": [
            {
                "id": "guide",
                "title": "Official guide",
                "url": "https://example.com/official-guide",
                "locator": "Figure requirements",
                "retrieved_on": "2026-08-02",
                "verified_on": "2026-08-02",
            }
        ],
        "caveats": [],
        "target": report["target"],
        "artifacts": [
            {
                "path": "figure1.pdf",
                "sha256": "b" * 64,
                "bytes": 4096,
                "format": "pdf",
            }
        ],
        "metadata": {"experiment": "baseline"},
        "report": report,
    }

    export_validator.validate(export_manifest)


def test_bundle_manifest_matches_public_manifest_schema() -> None:
    _, _, manifest_validator = _validators()
    report = _report_payload()
    manifest = {
        "schema_version": 1,
        "researchplot_version": "1.0.0",
        "profile": "nature@2026.08.0",
        "profile_digest": "a" * 64,
        "sources": [
            {
                "id": "guide",
                "title": "Official guide",
                "url": "https://example.com/official-guide",
                "locator": "Figure requirements",
                "retrieved_on": "2026-08-02",
                "verified_on": "2026-08-02",
            }
        ],
        "caveats": [],
        "artifacts": [
            {
                "path": "figure1.pdf",
                "sha256": "b" * 64,
                "bytes": 4096,
                "format": "pdf",
            }
        ],
        "figures": [
            {
                "name": "figure1",
                "paths": ["figure1.pdf"],
                "metadata": {
                    "alt_text": "A line rises as the measured input increases.",
                    "caption": "Measured response by input.",
                    "source_data": "data/figure1.csv",
                },
                "report": report,
            }
        ],
    }

    manifest_validator.validate(manifest)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [("verdict", "probably"), ("schema_version", 2)],
)
def test_report_schema_rejects_contract_drift(field: str, invalid_value: object) -> None:
    report_validator, _, _ = _validators()
    payload = _report_payload()
    payload[field] = invalid_value

    with pytest.raises(ValidationError):
        report_validator.validate(payload)


def test_manifest_schema_rejects_non_sha256_digest() -> None:
    _, _, manifest_validator = _validators()
    manifest = {
        "schema_version": 1,
        "researchplot_version": "1.0.0",
        "profile": "nature@2026.08.0",
        "profile_digest": "not-a-digest",
        "sources": [
            {
                "id": "guide",
                "title": "Official guide",
                "url": "https://example.com/official-guide",
                "locator": "Figure requirements",
                "retrieved_on": "2026-08-02",
                "verified_on": "2026-08-02",
            }
        ],
        "caveats": [],
        "artifacts": [{"path": "figure1.pdf", "sha256": "b" * 64, "bytes": 1, "format": "pdf"}],
        "figures": [
            {
                "name": "figure1",
                "paths": ["figure1.pdf"],
                "metadata": {"alt_text": None, "caption": None, "source_data": None},
                "report": _report_payload(),
            }
        ],
    }

    with pytest.raises(ValidationError):
        manifest_validator.validate(manifest)
