"""Access to the machine-readable v1 report and manifest contracts."""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any, cast


@lru_cache(maxsize=3)
def _load_contract(name: str) -> dict[str, Any]:
    resource = files("researchplot").joinpath("schemas", name)
    try:
        try:
            text = resource.read_text(encoding="utf-8")
        except FileNotFoundError:
            source_tree = Path(__file__).resolve().parents[2] / "schemas" / name
            text = source_tree.read_text(encoding="utf-8")
        payload = json.loads(text)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"ResearchPlot was installed without a valid {name} contract.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Bundled contract {name} is not a JSON object.")
    return cast(dict[str, Any], payload)


def report_schema() -> dict[str, Any]:
    """Return an independent copy of the compliance-report JSON Schema."""

    return copy.deepcopy(_load_contract("report.schema.json"))


def export_manifest_schema() -> dict[str, Any]:
    """Return an independent copy of the single-export manifest JSON Schema."""

    return copy.deepcopy(_load_contract("export-manifest.schema.json"))


def submission_manifest_schema() -> dict[str, Any]:
    """Return an independent copy of the multi-figure bundle manifest JSON Schema."""

    return copy.deepcopy(_load_contract("submission-manifest.schema.json"))
