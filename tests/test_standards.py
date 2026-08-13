from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

import researchplot as rp


def _manifest() -> dict[str, object]:
    return {
        "profile": "nature@2026.08.0",
        "profile_digest": "a" * 64,
        "sources": [],
        "artifacts": [
            {"path": "figure-1.pdf", "format": "pdf", "bytes": 10, "sha256": "b" * 64},
            {"path": "figure-1.svg", "format": "svg", "bytes": 20, "sha256": "c" * 64},
            {"path": "data/figure-1.csv", "format": "csv", "bytes": 30, "sha256": "d" * 64},
        ],
        "figures": [
            {
                "name": "figure-1",
                "paths": ["figure-1.pdf", "figure-1.svg"],
                "metadata": {
                    "number": "S1",
                    "caption": "Measured response by condition.",
                    "alt_text": "Two curves rise at different rates.",
                    "long_description": "The solid curve rises faster than the dashed curve.",
                    "source_data": ["data/figure-1.csv"],
                },
            }
        ],
    }


def test_jats_preserves_authored_accessibility_text_and_alternatives() -> None:
    fragment = rp.submission_manifest_to_jats(_manifest())
    root = ET.fromstring(fragment)
    figure = root.find("fig")

    assert figure is not None
    assert figure.findtext("label") == "Figure S1"
    assert figure.findtext("caption/p") == "Measured response by condition."
    assert figure.findtext("long-desc") == ("The solid curve rises faster than the dashed curve.")
    graphics = figure.findall("alternatives/graphic")
    assert len(graphics) == 2
    assert {graphic.findtext("alt-text") for graphic in graphics} == {
        "Two curves rise at different rates."
    }


def test_ro_crate_links_figure_representations_and_source_data() -> None:
    crate = rp.submission_manifest_to_ro_crate(_manifest())
    graph = crate["@graph"]
    assert isinstance(graph, list)
    figure = next(item for item in graph if item.get("@id") == "#figure-figure-1")

    assert figure["@type"] == "ImageObject"
    assert figure["associatedMedia"] == [
        {"@id": "figure-1.pdf"},
        {"@id": "figure-1.svg"},
    ]
    assert figure["hasPart"] == [{"@id": "data/figure-1.csv"}]
    assert figure["abstract"] == "Two curves rise at different rates."


def test_ro_crate_rejects_broken_figure_relationships() -> None:
    manifest = _manifest()
    figure = manifest["figures"][0]  # type: ignore[index]
    figure["metadata"]["source_data"] = ["data/missing.csv"]  # type: ignore[index]

    with pytest.raises(ValueError, match="absent from the manifest"):
        rp.submission_manifest_to_ro_crate(manifest)
