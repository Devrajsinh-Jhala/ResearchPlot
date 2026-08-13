"""Generate additional launch profiles from directly inspected official sources."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from build_verified_profiles_v3 import profile, rule, source


def applies(
    item: dict[str, Any],
    *,
    roles: list[str] | None = None,
    widths: list[str] | None = None,
) -> dict[str, Any]:
    item["applies_to"]["roles"] = roles or []
    item["applies_to"]["widths"] = widths or []
    return item


CELL = "cell-graphical-abstract"
JACS = "jacs-requirements"
ACS_TOC = "acs-toc-guidelines"
PRL = "prl-author-guidelines"
AAAI = "aaai-2026-submission"
ICLR = "iclr-2026-template"
AISTATS = "aistats-2026-template"
ECCV = "eccv-2026-template"
JMLR = "jmlr-format"
TMLR = "tmlr-template"


cell = profile(
    "cell-graphical-abstract",
    "Cell Press graphical abstract",
    "publisher",
    None,
    ["cell press graphical abstract", "cell graphical abstract"],
    "Cell Press graphical abstracts only; not main manuscript figures.",
    "square",
    [
        source(
            CELL,
            "Cell Press Graphical Abstract Guidelines",
            "https://crosstalk.cell.com/hubfs/Files/GA_guide.pdf",
            "Technical requirements: size, resolution, type, and preferred formats",
        )
    ],
    [
        applies(
            rule(
                "figure.width.square",
                "artifact.width_mm",
                "approx",
                139.7,
                source_id=CELL,
                description="Graphical abstracts should be 5.5 inches square.",
                unit="mm",
                tolerance=0.5,
                phases=["live", "file"],
            ),
            roles=["graphical_abstract"],
            widths=["square"],
        ),
        applies(
            rule(
                "figure.height.square",
                "artifact.height_mm",
                "approx",
                139.7,
                source_id=CELL,
                description="Graphical abstracts should be 5.5 inches square.",
                unit="mm",
                tolerance=0.5,
                phases=["live", "file"],
            ),
            roles=["graphical_abstract"],
            widths=["square"],
        ),
        applies(
            rule(
                "export.min_dpi.graphical_abstract",
                "artifact.dpi",
                "gte",
                300,
                source_id=CELL,
                description="The submitted graphical abstract should be 300 DPI.",
                unit="dpi",
                phases=["file"],
                formats=["png", "jpeg", "tiff"],
            ),
            roles=["graphical_abstract"],
        ),
        applies(
            rule(
                "font.size.minimum",
                "font.size.min_pt",
                "gte",
                12,
                source_id=CELL,
                description="Graphical-abstract text should be at least 12 point.",
                unit="pt",
            ),
            roles=["graphical_abstract"],
        ),
        applies(
            rule(
                "font.size.maximum",
                "font.size.max_pt",
                "lte",
                16,
                source_id=CELL,
                description="Graphical-abstract text should not exceed 16 point.",
                unit="pt",
            ),
            roles=["graphical_abstract"],
        ),
        applies(
            rule(
                "font.family.arial",
                "font.families.effective",
                "subset",
                ["Arial"],
                source_id=CELL,
                description="Cell Press specifies Arial for graphical abstracts.",
            ),
            roles=["graphical_abstract"],
        ),
        applies(
            rule(
                "export.formats.preferred",
                "artifact.format",
                "in",
                ["tiff", "pdf", "jpeg"],
                source_id=CELL,
                description="TIFF, PDF, and JPEG are the preferred graphical-abstract formats.",
                level="recommended",
                phases=["file"],
            ),
            roles=["graphical_abstract"],
        ),
    ],
    [
        "This profile applies only to the separate Cell Press graphical-abstract deliverable.",
        "The guide does not establish dimensions for ordinary Cell Press manuscript figures.",
    ],
)

jacs = profile(
    "jacs",
    "Journal of the American Chemical Society",
    "journal",
    None,
    ["journal of the american chemical society"],
    "JACS manuscript artwork plus the required TOC graphic.",
    "single",
    [
        source(
            JACS,
            "JACS submission requirements",
            "https://pubs.acs.org/page/jacsat/submission/recentchanges.html",
            "Articles and Communications require a TOC graphic",
            "official_policy",
        ),
        source(
            ACS_TOC,
            "ACS Guidelines for Table of Contents/Abstract Graphics",
            "https://pubsapp.acs.org/paragonplus/submission/toc_abstract_graphics_guidelines.pdf",
            "Specifications, updated February 28, 2024",
        ),
    ],
    [
        applies(
            rule(
                "figure.width.toc_maximum",
                "artifact.width_mm",
                "lte",
                82.55,
                source_id=ACS_TOC,
                description="A TOC graphic must be no wider than 3.25 inches.",
                unit="mm",
                phases=["live", "file"],
            ),
            roles=["graphical_abstract"],
        ),
        applies(
            rule(
                "figure.height.toc_maximum",
                "artifact.height_mm",
                "lte",
                44.45,
                source_id=ACS_TOC,
                description="A TOC graphic must be no higher than 1.75 inches.",
                unit="mm",
                phases=["live", "file"],
            ),
            roles=["graphical_abstract"],
        ),
        applies(
            rule(
                "font.size.toc_minimum",
                "font.size.min_pt",
                "gte",
                6,
                source_id=ACS_TOC,
                description="TOC graphic text must be at least 6 point.",
                unit="pt",
            ),
            roles=["graphical_abstract"],
        ),
        applies(
            rule(
                "font.family.toc",
                "font.families.effective",
                "subset",
                ["Helvetica", "Arial"],
                source_id=ACS_TOC,
                description="A sans-serif face such as Helvetica is recommended for TOC graphics.",
                level="recommended",
            ),
            roles=["graphical_abstract"],
        ),
        applies(
            rule(
                "export.formats.toc",
                "artifact.format",
                "in",
                ["tiff", "eps"],
                source_id=ACS_TOC,
                description="TOC graphics must be supplied as TIFF or EPS.",
                phases=["file"],
            ),
            roles=["graphical_abstract"],
        ),
        applies(
            rule(
                "export.min_dpi.toc",
                "artifact.dpi",
                "gte",
                300,
                source_id=ACS_TOC,
                description="Color TIFF TOC graphics require at least 300 DPI.",
                unit="dpi",
                phases=["file"],
                formats=["tiff"],
            ),
            roles=["graphical_abstract"],
        ),
    ],
    [
        "The profile can validate a supplied TOC graphic but cannot establish that the manuscript contains one.",
        "Black-and-white TIFF TOC graphics require 1200 DPI; that color-mode-specific rule remains manual.",
    ],
)
jacs["extends"] = ["acs-generic@2026.08.0"]

prl = profile(
    "physical-review-letters",
    "Physical Review Letters",
    "journal",
    None,
    ["prl", "physical review letters"],
    "PRL figure accessibility guidance.",
    None,
    [
        source(
            PRL,
            "Physical Review Letters publishing guidelines",
            "https://journals.aps.org/prl/authors",
            "Accessibility requirements for figures",
            "official_policy",
        )
    ],
    [
        rule(
            "accessibility.non_color_distinctions",
            "color.non_color_distinctions",
            "eq",
            True,
            source_id=PRL,
            description="Use words or symbols rather than color alone to convey meaning.",
        )
    ],
    [
        "The accessible PRL author page does not state exact figure widths; no physical geometry is claimed."
    ],
)

aaai = profile(
    "aaai-2026",
    "AAAI 2026",
    "conference",
    2026,
    ["aaai", "aaai 26", "aaai-26"],
    "AAAI-26 PDF font technology requirement.",
    None,
    [
        source(
            AAAI,
            "AAAI-26 Submission Instructions",
            "https://aaai.org/conference/aaai/aaai-26/submission-instructions/",
            "High-resolution PDF using Type 1 or TrueType fonts",
            "official_policy",
        )
    ],
    [
        rule(
            "pdf.fonts.no_type3",
            "pdf.type3_font_count",
            "eq",
            0,
            source_id=AAAI,
            description="AAAI-26 permits Type 1 or TrueType fonts, not Type 3 fonts.",
            phases=["file"],
            formats=["pdf"],
        )
    ],
    [
        "The accessible submission page establishes PDF font technology but not standalone figure dimensions.",
        "This profile therefore cannot create a venue-sized style.",
    ],
)

iclr = profile(
    "iclr-2026",
    "International Conference on Learning Representations 2026",
    "conference",
    2026,
    ["iclr", "iclr 2026"],
    "ICLR 2026 paper figures under the official template.",
    "full",
    [
        source(
            "iclr-author-guide",
            "ICLR 2026 Author Guide",
            "https://iclr.cc/Conferences/2026/AuthorGuide",
            "Mandatory official template",
            "official_policy",
        ),
        source(
            ICLR,
            "Official ICLR 2026 template",
            "https://github.com/ICLR/Master-Template/raw/master/iclr2026.zip",
            "Style text width/height and figure instructions",
            "official_template",
        ),
    ],
    [
        rule(
            "figure.width.full",
            "artifact.width_mm",
            "approx",
            139.7,
            source_id=ICLR,
            description="The official template text width is 5.5 inches.",
            level="inferred",
            unit="mm",
            tolerance=0.5,
            phases=["live", "file"],
            rationale="The full-width figure target is the template text width.",
        ),
        rule(
            "figure.height.maximum",
            "artifact.height_mm",
            "lte",
            228.6,
            source_id=ICLR,
            description="Figures must fit within the 9-inch text block.",
            level="inferred",
            unit="mm",
            phases=["live", "file"],
            rationale="The maximum usable height is derived from the template text height.",
        ),
        rule(
            "accessibility.non_color_distinctions",
            "color.non_color_distinctions",
            "eq",
            True,
            source_id=ICLR,
            description="Captions and body text should make sense in black and white or color.",
            level="recommended",
        ),
    ],
    [
        "The template is single-column; it does not define a narrower canonical figure width.",
        "No figure-specific minimum font size is stated.",
    ],
)

aistats = profile(
    "aistats-2026",
    "Artificial Intelligence and Statistics 2026",
    "conference",
    2026,
    ["aistats", "aistats 2026"],
    "AISTATS 2026 proceedings figures.",
    "single",
    [
        source(
            "aistats-call",
            "AISTATS 2026 Call for Papers",
            "https://virtual.aistats.org/Conferences/2026/CallForPapers",
            "Mandatory official paper pack",
            "official_policy",
        ),
        source(
            AISTATS,
            "AISTATS 2026 paper pack",
            "https://aistats.org/aistats2026/AISTATS2026PaperPack.zip",
            "3.25/6.75-inch columns and 9.25-inch text height",
            "official_template",
        ),
    ],
    [
        rule(
            "figure.width.single",
            "artifact.width_mm",
            "approx",
            82.55,
            source_id=AISTATS,
            description="A single column is 3.25 inches wide.",
            unit="mm",
            tolerance=0.5,
            phases=["live", "file"],
        ),
        rule(
            "figure.width.double",
            "artifact.width_mm",
            "approx",
            171.45,
            source_id=AISTATS,
            description="The two-column text block is 6.75 inches wide.",
            unit="mm",
            tolerance=0.5,
            phases=["live", "file"],
        ),
        rule(
            "figure.height.maximum",
            "artifact.height_mm",
            "lte",
            234.95,
            source_id=AISTATS,
            description="Figures must fit within the 9.25-inch text block.",
            level="inferred",
            unit="mm",
            phases=["live", "file"],
            rationale="Maximum usable height is derived from the template text height.",
        ),
    ],
    ["The template requires legible artwork but does not state a numeric figure-font minimum."],
)

eccv = profile(
    "eccv-2026",
    "European Conference on Computer Vision 2026",
    "conference",
    2026,
    ["eccv", "eccv 2026"],
    "ECCV 2026 paper figures under the official author kit.",
    "full",
    [
        source(
            "eccv-policy",
            "ECCV 2026 Submission Policies",
            "https://eccv.ecva.net/Conferences/2026/SubmissionPolicies",
            "Mandatory, changed 2026 author kit",
            "official_policy",
        ),
        source(
            ECCV,
            "ECCV 2026 official-linked paper template",
            "https://github.com/paolo-favaro/paper-template",
            "122 x 193 mm text block and 6-point figure minimum",
            "official_template",
        ),
    ],
    [
        rule(
            "figure.width.full",
            "artifact.width_mm",
            "approx",
            122.0,
            source_id=ECCV,
            description="The official template text block is 122 mm wide.",
            level="inferred",
            unit="mm",
            tolerance=0.5,
            phases=["live", "file"],
            rationale="The full-width target is the template text width.",
        ),
        rule(
            "figure.height.maximum",
            "artifact.height_mm",
            "lte",
            193.0,
            source_id=ECCV,
            description="Figures must fit within the 193 mm text block.",
            level="inferred",
            unit="mm",
            phases=["live", "file"],
        ),
        rule(
            "font.size.minimum",
            "font.size.min_pt",
            "gte",
            6,
            source_id=ECCV,
            description="Figure lettering must not be smaller than 6 point.",
            unit="pt",
        ),
    ],
    ["ECCV 2026 uses a single-column LNCS-derived layout; no second canonical width is stated."],
)

jmlr = profile(
    "jmlr",
    "Journal of Machine Learning Research",
    "journal",
    None,
    ["journal of machine learning research"],
    "JMLR article figures under the official final format.",
    "full",
    [
        source(
            JMLR,
            "JMLR Instructions for Formatting Articles",
            "https://jmlr.org/format/format.html",
            "Single-column geometry and figure typography",
            "official_guideline",
        )
    ],
    [
        rule(
            "figure.width.full",
            "artifact.width_mm",
            "approx",
            152.4,
            source_id=JMLR,
            description="The official single-column text width is 6 inches.",
            level="inferred",
            unit="mm",
            tolerance=0.5,
            phases=["live", "file"],
            rationale="The full-width target is the official style-file text width.",
        ),
        rule(
            "figure.height.maximum",
            "artifact.height_mm",
            "lte",
            215.9,
            source_id=JMLR,
            description="Figures must fit within the 8.5-inch text block.",
            level="inferred",
            unit="mm",
            phases=["live", "file"],
        ),
        rule(
            "font.size.minimum",
            "font.size.min_pt",
            "gte",
            9,
            source_id=JMLR,
            description="Captions, labels, and illustration text must be at least 9 point.",
            unit="pt",
        ),
    ],
    ["The width is the full text block; JMLR does not define alternative named figure widths."],
)

tmlr = profile(
    "tmlr",
    "Transactions on Machine Learning Research",
    "journal",
    None,
    ["transactions on machine learning research"],
    "TMLR paper figures under the mandatory official template.",
    "full",
    [
        source(
            "tmlr-author-guide",
            "TMLR guidelines for authors",
            "https://jmlr.org/tmlr/author-guide.html",
            "Mandatory PDF and LaTeX template",
            "official_policy",
        ),
        source(
            TMLR,
            "Official TMLR style file and template",
            "https://github.com/JmlrOrg/tmlr-style-file",
            "6.5 x 9 inch text block and figure guidance",
            "official_template",
        ),
    ],
    [
        rule(
            "figure.width.full",
            "artifact.width_mm",
            "approx",
            165.1,
            source_id=TMLR,
            description="The mandatory template text width is 6.5 inches.",
            level="inferred",
            unit="mm",
            tolerance=0.5,
            phases=["live", "file"],
            rationale="The full-width target is the template text width.",
        ),
        rule(
            "figure.height.maximum",
            "artifact.height_mm",
            "lte",
            228.6,
            source_id=TMLR,
            description="Figures must fit within the 9-inch text block.",
            level="inferred",
            unit="mm",
            phases=["live", "file"],
        ),
        rule(
            "accessibility.non_color_distinctions",
            "color.non_color_distinctions",
            "eq",
            True,
            source_id=TMLR,
            description="Captions and body text should make sense in black and white or color.",
            level="recommended",
        ),
    ],
    ["The template does not state a numeric minimum font size specifically for figures."],
)

PROFILES = [cell, jacs, prl, aaai, iclr, aistats, eccv, jmlr, tmlr]


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "src" / "researchplot" / "profiles"
    for item in PROFILES:
        (root / f"{item['id']}.json").write_text(
            json.dumps(item, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
