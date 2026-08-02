"""Generate self-contained export and bundle manifest schemas from report v1."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "schemas"


def _source_properties() -> dict[str, object]:
    return {
        "type": "array",
        "minItems": 1,
        "items": {"$ref": "#/$defs/source"},
    }


def _artifact_definition() -> dict[str, object]:
    return {
        "type": "object",
        "required": ["path", "sha256", "bytes", "format"],
        "properties": {
            "path": {"type": "string", "minLength": 1},
            "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "bytes": {"type": "integer", "minimum": 0},
            "format": {"type": "string", "pattern": "^[a-z0-9]+$"},
        },
        "additionalProperties": False,
    }


def _metadata_definition() -> dict[str, object]:
    nullable_string = {"type": ["string", "null"]}
    return {
        "type": "object",
        "required": ["alt_text", "caption", "source_data"],
        "properties": {
            "alt_text": copy.deepcopy(nullable_string),
            "caption": copy.deepcopy(nullable_string),
            "source_data": copy.deepcopy(nullable_string),
        },
        "additionalProperties": False,
    }


def _report_contract() -> tuple[dict[str, object], dict[str, object]]:
    report = json.loads((SCHEMAS / "report.schema.json").read_text(encoding="utf-8"))
    definition = {
        key: copy.deepcopy(report[key])
        for key in ("type", "required", "properties", "additionalProperties")
    }
    return definition, copy.deepcopy(report["$defs"])


def _common_properties() -> dict[str, object]:
    return {
        "schema_version": {"const": 1},
        "researchplot_version": {"type": "string", "minLength": 1},
        "profile": {"type": "string", "minLength": 1},
        "profile_digest": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "sources": _source_properties(),
        "caveats": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "uniqueItems": True,
        },
        "artifacts": {
            "type": "array",
            "minItems": 1,
            "items": {"$ref": "#/$defs/artifact"},
        },
    }


def _schema_header(identifier: str, title: str, description: str) -> dict[str, object]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": (
            "https://raw.githubusercontent.com/Devrajsinh-Jhala/ResearchPlot/"
            f"v1.0.0/schemas/{identifier}"
        ),
        "title": title,
        "description": description,
        "type": "object",
    }


def generate() -> dict[str, dict[str, object]]:
    report_definition, report_defs = _report_contract()
    shared_defs = {
        **report_defs,
        "report": report_definition,
        "artifact": _artifact_definition(),
    }

    export = _schema_header(
        "export-manifest.schema.json",
        "ResearchPlot single-export manifest",
        "Evidence manifest written beside artifacts by Target.export().",
    )
    export_properties = _common_properties()
    export_properties.update(
        {
            "target": {"$ref": "#/$defs/target"},
            "metadata": {"type": "object", "additionalProperties": True},
            "report": {"$ref": "#/$defs/report"},
        }
    )
    export.update(
        {
            "required": [
                "schema_version",
                "researchplot_version",
                "profile",
                "profile_digest",
                "sources",
                "caveats",
                "target",
                "artifacts",
                "metadata",
                "report",
            ],
            "properties": export_properties,
            "additionalProperties": False,
            "$defs": copy.deepcopy(shared_defs),
        }
    )

    bundle = _schema_header(
        "submission-manifest.schema.json",
        "ResearchPlot submission bundle manifest",
        "Evidence manifest written as researchplot-manifest.json by Submission.build().",
    )
    bundle_properties = _common_properties()
    bundle_properties["figures"] = {
        "type": "array",
        "minItems": 1,
        "items": {"$ref": "#/$defs/figure"},
    }
    bundle_defs = copy.deepcopy(shared_defs)
    bundle_defs["metadata"] = _metadata_definition()
    bundle_defs["figure"] = {
        "type": "object",
        "required": ["name", "paths", "metadata", "report"],
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "paths": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1},
            },
            "metadata": {"$ref": "#/$defs/metadata"},
            "report": {"$ref": "#/$defs/report"},
        },
        "additionalProperties": False,
    }
    bundle.update(
        {
            "required": [
                "schema_version",
                "researchplot_version",
                "profile",
                "profile_digest",
                "sources",
                "caveats",
                "artifacts",
                "figures",
            ],
            "properties": bundle_properties,
            "additionalProperties": False,
            "$defs": bundle_defs,
        }
    )
    return {
        "export-manifest.schema.json": export,
        "submission-manifest.schema.json": bundle,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    stale: list[str] = []
    for name, payload in generate().items():
        expected = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        path = SCHEMAS / name
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != expected:
                stale.append(name)
        else:
            path.write_text(expected, encoding="utf-8")
    if stale:
        parser.error("generated contract schemas are stale: " + ", ".join(stale))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
