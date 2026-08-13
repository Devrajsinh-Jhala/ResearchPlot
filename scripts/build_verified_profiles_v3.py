"""Generate the first post-v1 verified profile expansion from official sources."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TODAY = "2026-08-03"
REVISION = "2026.08.0"


def source(
    id: str, title: str, url: str, locator: str, kind: str = "official_guideline"
) -> dict[str, Any]:
    return {
        "id": id,
        "title": title,
        "url": url,
        "locator": locator,
        "retrieved_on": TODAY,
        "verified_on": TODAY,
        "kind": kind,
        "publisher": title,
        "archive_url": None,
        "content_sha256": None,
    }


def rule(
    id: str,
    probe: str,
    operator: str,
    value: object,
    *,
    source_id: str,
    description: str,
    level: str = "required",
    unit: str | None = None,
    tolerance: float | None = None,
    phases: list[str] | None = None,
    formats: list[str] | None = None,
    content: list[str] | None = None,
    verification: str = "automated",
    rationale: str | None = None,
) -> dict[str, Any]:
    return {
        "id": id,
        "level": level,
        "probe": probe,
        "constraint": {"operator": operator, "value": value, "unit": unit, "tolerance": tolerance},
        "applies_to": {
            "roles": [],
            "content_kinds": content or [],
            "output_formats": formats or [],
            "widths": [],
        },
        "verification": verification,
        "phases": phases or ["live"],
        "source_ids": [source_id],
        "description": description,
        "expression": None,
        "supersedes": [],
        "rationale": rationale,
    }


def profile(
    id: str,
    name: str,
    kind: str,
    year: int | None,
    aliases: list[str],
    scope: str,
    default_width: str | None,
    sources: list[dict[str, Any]],
    rules: list[dict[str, Any]],
    caveats: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "id": id,
        "revision": REVISION,
        "effective_date": TODAY,
        "name": name,
        "kind": kind,
        "year": year,
        "aliases": aliases,
        "scope": scope,
        "default_width": default_width,
        "verified_on": TODAY,
        "caveats": caveats,
        "sources": sources,
        "rules": rules,
        "namespace": "researchplot",
        "status": "verified",
        "license": "MIT",
        "maintainers": ["Devrajsinh Jhala"],
        "extends": [],
        "governance": {
            "policy": "researchplot-governance-v1",
            "reviewers": ["Devrajsinh Jhala"],
            "reviewed_on": TODAY,
            "change_note": "Initial source-backed schema-v3 profile.",
        },
    }


ACS = "acs-artwork"
COLM = "colm-template"
AAS = "aas-graphics"
USENIX = "usenix-template"
OSDI = "osdi-requirements"

PROFILES = [
    profile(
        "acs-generic",
        "American Chemical Society generic artwork",
        "publisher",
        None,
        ["acs", "american chemical society"],
        "Generic ACS journal artwork guidance.",
        "single",
        [
            source(
                ACS,
                "ACS Graphics Preparation",
                "https://pubs.acs.org/page/4authors/submission/graphics_prep.html",
                "Sizing, typography, lines, and resolution",
            )
        ],
        [
            rule(
                "figure.width.single",
                "artifact.width_mm",
                "approx",
                82.55,
                source_id=ACS,
                description="Single-column graphics are 3.25 inches wide.",
                unit="mm",
                tolerance=0.5,
                phases=["live", "file"],
            ),
            rule(
                "figure.width.double",
                "artifact.width_mm",
                "approx",
                177.8,
                source_id=ACS,
                description="Double-column graphics are 7 inches wide.",
                unit="mm",
                tolerance=0.5,
                phases=["live", "file"],
            ),
            rule(
                "figure.height.maximum",
                "artifact.height_mm",
                "lte",
                241.3,
                source_id=ACS,
                description="Graphics must not exceed 9.5 inches in height.",
                unit="mm",
                phases=["live", "file"],
            ),
            rule(
                "font.size.minimum",
                "font.size.min_pt",
                "gte",
                5,
                source_id=ACS,
                description="Lettering must be at least 5 point.",
                unit="pt",
            ),
            rule(
                "font.family.allowed",
                "font.families.effective",
                "subset",
                ["Arial", "Helvetica"],
                source_id=ACS,
                description="Use Arial or Helvetica lettering.",
                level="recommended",
            ),
            rule(
                "line.width.minimum",
                "line.width.min_pt",
                "gte",
                1,
                source_id=ACS,
                description="Lines should be at least 1 point.",
                unit="pt",
            ),
            rule(
                "export.min_dpi.recommended",
                "artifact.dpi",
                "gte",
                600,
                source_id=ACS,
                description="ACS recommends at least 600 DPI for best results.",
                level="recommended",
                unit="dpi",
                phases=["file"],
                formats=["png", "jpeg", "tiff"],
            ),
        ],
        ["Individual ACS journals and article types may override this generic artwork guidance."],
    ),
    profile(
        "colm-2026",
        "Conference on Language Modeling 2026",
        "conference",
        2026,
        ["colm", "colm 2026"],
        "COLM 2026 proceedings figures.",
        "full",
        [
            source(
                COLM,
                "Official COLM 2026 template",
                "https://github.com/COLM-org/Template/releases/tag/2026",
                "colm2026_conference.sty and example paper",
                "official_template",
            ),
            source(
                "colm-instructions",
                "COLM submission instructions",
                "https://colmweb.org/submission-instructions.html",
                "Official template requirement",
                "official_policy",
            ),
        ],
        [
            rule(
                "figure.width.full",
                "artifact.width_mm",
                "approx",
                139.7,
                source_id=COLM,
                description="Full text width is 5.5 inches.",
                unit="mm",
                tolerance=0.5,
                phases=["live", "file"],
            ),
            rule(
                "figure.height.maximum",
                "artifact.height_mm",
                "lte",
                228.6,
                source_id=COLM,
                description="The template text block is 9 inches high.",
                unit="mm",
                phases=["live", "file"],
                level="inferred",
                rationale="Derived from the official template text-height declaration.",
            ),
            rule(
                "font.size.minimum",
                "font.size.min_pt",
                "gte",
                9,
                source_id=COLM,
                description="Figure text should not be smaller than the template small size.",
                unit="pt",
            ),
            rule(
                "font.family.template",
                "font.families.effective",
                "subset",
                ["Palatino", "Palatino Linotype", "Book Antiqua", "serif"],
                source_id=COLM,
                description="Match the Palatino-family template typography.",
                level="recommended",
            ),
            rule(
                "export.formats.vector",
                "artifact.format",
                "in",
                ["pdf", "svg"],
                source_id=COLM,
                description="Use vector PDF or SVG for line art when possible.",
                level="recommended",
                phases=["file"],
                content=["data_visualization", "line_art"],
            ),
            rule(
                "accessibility.non_color_distinctions",
                "color.non_color_distinctions",
                "eq",
                True,
                source_id=COLM,
                description="Figures should remain comprehensible in black and white.",
                level="recommended",
            ),
        ],
        [
            "The 5.5-inch rule is full text width; narrower layouts remain author choices unless the template states otherwise."
        ],
    ),
    profile(
        "aas-journals",
        "American Astronomical Society journals",
        "publisher",
        None,
        ["aas", "aas journals", "astrophysical journal"],
        "Static graphics submitted to AAS journals.",
        None,
        [
            source(
                AAS,
                "AAS Graphics Guide",
                "https://journals.aas.org/graphics-guide/",
                "File formats, resolution, fonts, lines, and accessibility",
            )
        ],
        [
            rule(
                "export.formats.allowed",
                "artifact.format",
                "in",
                ["pdf", "eps", "png", "jpeg", "tiff"],
                source_id=AAS,
                description="AAS accepts PDF, EPS, PNG, JPEG, and TIFF static graphics.",
                phases=["file"],
            ),
            rule(
                "export.formats.vector",
                "artifact.format",
                "in",
                ["pdf", "eps"],
                source_id=AAS,
                description="PDF or EPS is preferred for vector graphics.",
                level="recommended",
                phases=["file"],
                content=["data_visualization", "line_art"],
            ),
            rule(
                "export.min_dpi.raster",
                "artifact.dpi",
                "gte",
                300,
                source_id=AAS,
                description="Raster graphics require at least 300 DPI.",
                unit="dpi",
                phases=["file"],
                formats=["png", "jpeg", "tiff"],
            ),
            rule(
                "font.size.minimum",
                "font.size.min_pt",
                "gte",
                6,
                source_id=AAS,
                description="Text must be at least 6 point at publication size.",
                unit="pt",
            ),
            rule(
                "font.family.allowed",
                "font.families.effective",
                "subset",
                ["Times", "Times New Roman", "Helvetica", "Symbol"],
                source_id=AAS,
                description="Use common Times, Helvetica, or Symbol fonts.",
                level="recommended",
            ),
            rule(
                "line.width.minimum",
                "line.width.min_pt",
                "gte",
                0.5,
                source_id=AAS,
                description="Lines must be at least 0.5 point.",
                unit="pt",
            ),
            rule(
                "artifact.page_count.single",
                "artifact.page_count",
                "eq",
                1,
                source_id=AAS,
                description="Each graphics file must contain one page.",
                unit="page",
                phases=["file"],
            ),
            rule(
                "accessibility.non_color_distinctions",
                "color.non_color_distinctions",
                "eq",
                True,
                source_id=AAS,
                description="Do not use color as the only visual delimiter.",
                level="recommended",
            ),
        ],
        [
            "Interactive figures and data-behind-the-figure deliverables have additional requirements not automated by this profile."
        ],
    ),
    profile(
        "usenix-osdi-2026",
        "USENIX OSDI 2026",
        "conference",
        2026,
        ["usenix", "osdi", "osdi 2026", "usenix 2026"],
        "OSDI 2026 paper figures under the official USENIX template.",
        "single",
        [
            source(
                USENIX,
                "USENIX paper templates",
                "https://www.usenix.org/conferences/author-resources/paper-templates",
                "7 x 9 inch text block, two columns, typography",
                "official_template",
            ),
            source(
                OSDI,
                "OSDI 2026 requirements for authors",
                "https://www.usenix.org/conference/osdi26/requirements-authors",
                "PDF and black-and-white figure requirements",
                "official_policy",
            ),
        ],
        [
            rule(
                "figure.width.full",
                "artifact.width_mm",
                "approx",
                177.8,
                source_id=USENIX,
                description="The template text block is 7 inches wide.",
                unit="mm",
                tolerance=0.5,
                phases=["live", "file"],
            ),
            rule(
                "figure.width.single",
                "artifact.width_mm",
                "approx",
                84.709,
                source_id=USENIX,
                description="Single-column width is derived from the 7-inch block and 0.33-inch gutter.",
                level="inferred",
                unit="mm",
                tolerance=0.5,
                phases=["live", "file"],
                rationale="(7 - 0.33) / 2 inches, converted to millimetres.",
            ),
            rule(
                "figure.height.maximum",
                "artifact.height_mm",
                "lte",
                228.6,
                source_id=USENIX,
                description="The template text block is 9 inches high.",
                unit="mm",
                phases=["live", "file"],
            ),
            rule(
                "font.size.body",
                "font.size.min_pt",
                "gte",
                10,
                source_id=USENIX,
                description="The template body typography is 10 point Times.",
                level="inferred",
                unit="pt",
                rationale="This is template body type; captions may have distinct documented sizing.",
            ),
            rule(
                "export.formats.pdf",
                "artifact.format",
                "in",
                ["pdf"],
                source_id=OSDI,
                description="OSDI submissions must be PDF.",
                phases=["file"],
            ),
            rule(
                "accessibility.non_color_distinctions",
                "color.non_color_distinctions",
                "eq",
                True,
                source_id=OSDI,
                description="Graphs and figures must be readable in black and white.",
            ),
        ],
        ["USENIX conference requirements vary; this profile is specifically pinned to OSDI 2026."],
    ),
]


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "src" / "researchplot" / "profiles"
    for item in PROFILES:
        (root / f"{item['id']}.json").write_text(
            json.dumps(item, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
