from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from PIL import Image
from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    FloatObject,
    NameObject,
    TextStringObject,
)

from researchplot.artifact_security import (
    InspectionBudget,
    InspectionResourceError,
    ManifestVerificationError,
    create_deterministic_archive,
    inspect_artifact_isolated,
    verify_deterministic_archive,
    verify_manifest,
)
from researchplot.inspectors import inspect_artifact
from researchplot.manuscript import audit_manuscript_pdf
from researchplot.remediation import RemediationKind, plan_remediation
from researchplot.standards import (
    RO_CRATE_CONTEXT,
    submission_manifest_to_jats,
    submission_manifest_to_ro_crate,
)
from researchplot.visual_diagnostics import diagnose_visual, render_accessibility_previews


def _active_pdf(path: Path, *, pages: int = 1) -> None:
    writer = PdfWriter()
    for _ in range(pages):
        page = writer.add_blank_page(width=252, height=144)
        image = DecodedStreamObject()
        image.set_data(b"\x00\x00\x00")
        image.update(
            {
                NameObject("/Type"): NameObject("/XObject"),
                NameObject("/Subtype"): NameObject("/Image"),
                NameObject("/Width"): FloatObject(1),
                NameObject("/Height"): FloatObject(1),
                NameObject("/ColorSpace"): NameObject("/DeviceRGB"),
                NameObject("/BitsPerComponent"): FloatObject(8),
                NameObject("/SMask"): DictionaryObject(),
            }
        )
        image_ref = writer._add_object(image)
        page[NameObject("/Resources")] = DictionaryObject(
            {
                NameObject("/XObject"): DictionaryObject({NameObject("/Im1"): image_ref}),
                NameObject("/ExtGState"): DictionaryObject(
                    {NameObject("/GS1"): DictionaryObject({NameObject("/ca"): FloatObject(0.5)})}
                ),
            }
        )
    writer._root_object[NameObject("/OpenAction")] = DictionaryObject(
        {
            NameObject("/S"): NameObject("/JavaScript"),
            NameObject("/JS"): TextStringObject("app.alert('must never execute')"),
        }
    )
    with path.open("wb") as stream:
        writer.write(stream)


def _bundle(root: Path) -> Path:
    root.mkdir()
    artifact = root / "figure.png"
    artifact.write_bytes(b"static artifact")
    payload = artifact.read_bytes()
    manifest = {
        "schema_version": 1,
        "profile": "nature@2026",
        "profile_digest": "a" * 64,
        "sources": [],
        "artifacts": [
            {
                "path": artifact.name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "format": "png",
            }
        ],
        "figures": [
            {
                "name": "result",
                "paths": [artifact.name],
                "metadata": {
                    "caption": "Accuracy by condition < baseline.",
                    "alt_text": "Three labeled lines increase from left to right.",
                },
            }
        ],
    }
    manifest_path = root / "researchplot-manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return manifest_path


def test_pdf_inspector_passively_reports_active_content_xobjects_and_transparency(
    tmp_path: Path,
) -> None:
    path = tmp_path / "active.pdf"
    _active_pdf(path)

    inspection = inspect_artifact(path)

    assert inspection.get("pdf.active_content") is True
    assert "JavaScript" in inspection.get("pdf.active_content_types", ())
    assert inspection.get("pdf.image_xobject_count") == 1
    assert inspection.get("pdf.image_color_spaces") == ("/DeviceRGB",)
    assert inspection.get("pdf.has_transparency") is True
    assert set(inspection.get("pdf.transparency_reasons", ())) == {
        "image-soft-mask",
        "non-opaque-fill",
    }


def test_svg_active_content_is_observed_without_fetching(tmp_path: Path) -> None:
    path = tmp_path / "active.svg"
    path.write_text(
        """<svg xmlns="http://www.w3.org/2000/svg" width="10mm" height="5mm">
<script>throw new Error('must never execute')</script>
<foreignObject><div xmlns="http://www.w3.org/1999/xhtml">x</div></foreignObject>
<a href="javascript:alert(1)" onclick="alert(2)"><animate attributeName="x" /></a>
</svg>""",
        encoding="utf-8",
    )

    inspection = inspect_artifact(path)

    assert inspection.get("svg.active_content") is True
    assert inspection.get("svg.script_element_count") == 1
    assert inspection.get("svg.foreign_object_count") == 1
    assert inspection.get("svg.event_handler_count") == 1
    assert inspection.get("svg.javascript_link_count") == 1
    assert inspection.get("svg.animation_element_count") == 1


def test_raster_exif_orientation_and_multiframe_are_explicit(tmp_path: Path) -> None:
    oriented = tmp_path / "oriented.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (20, 10), "red").save(oriented, dpi=(100, 200), exif=exif)

    inspected = inspect_artifact(oriented)

    assert inspected.get("raster.exif_orientation") == 6
    assert inspected.get("raster.requires_orientation_transform") is True
    assert inspected.get("raster.display_pixel_width") == 10
    assert inspected.get("raster.display_pixel_height") == 20

    multipage = tmp_path / "pages.tiff"
    first = Image.new("RGB", (10, 8), "red")
    second = Image.new("L", (6, 4), 128)
    first.save(multipage, save_all=True, append_images=[second], dpi=(300, 300))
    multi = inspect_artifact(multipage)
    assert multi.get("raster.frame_count") == 2
    assert multi.get("raster.frame_sizes") == ((10, 8), (6, 4))
    assert multi.get("raster.frames_uniform") is False
    assert multi.get("artifact.width_mm") is None


def test_isolated_inspection_enforces_preparse_byte_budget(tmp_path: Path) -> None:
    path = tmp_path / "figure.png"
    Image.new("RGB", (2, 2)).save(path)

    with pytest.raises(InspectionResourceError, match="budget"):
        inspect_artifact_isolated(path, budget=InspectionBudget(max_bytes=1))

    execution = inspect_artifact_isolated(path, budget=InspectionBudget(timeout_seconds=20))
    assert execution.inspection.format == "png"
    assert execution.isolated is True
    assert "process-boundary" in execution.limits_applied


def test_manifest_verification_and_deterministic_archive_detect_tampering(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _bundle(bundle)
    verification = verify_manifest(bundle, strict=True)
    assert verification.valid is True
    assert verification.checked_artifacts == 1

    first = create_deterministic_archive(bundle, tmp_path / "first.zip")
    second = create_deterministic_archive(bundle, tmp_path / "second.zip")
    assert first.sha256 == second.sha256
    assert verify_deterministic_archive(first.path).valid is True

    (bundle / "figure.png").write_bytes(b"changed")
    invalid = verify_manifest(bundle)
    assert invalid.valid is False
    assert {issue.code for issue in invalid.issues} == {"size-mismatch", "digest-mismatch"}


def test_deterministic_tar_and_source_date_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "bundle"
    _bundle(bundle)
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")

    first = create_deterministic_archive(bundle, tmp_path / "first.tar")
    second = create_deterministic_archive(bundle, tmp_path / "second.tar")

    assert first.sha256 == second.sha256
    assert verify_deterministic_archive(first.path).valid is True
    with pytest.raises(ValueError, match=".zip or .tar"):
        create_deterministic_archive(bundle, tmp_path / "unsupported.tgz")


def test_manifest_rejects_windows_portable_collision_and_unsafe_paths(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    manifest_path = _bundle(bundle)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"].append(
        {"path": "FIGURE.PNG", "bytes": 0, "sha256": "0" * 64, "format": "png"}
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = verify_manifest(bundle)
    assert "duplicate-path" in {issue.code for issue in result.issues}

    manifest["artifacts"][1]["path"] = "../escape.png"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = verify_manifest(bundle)
    assert "unsafe-path" in {issue.code for issue in result.issues}

    with pytest.raises(ManifestVerificationError, match="verification failed"):
        create_deterministic_archive(bundle, tmp_path / "unsafe.zip")


def test_accessibility_previews_are_deterministic_and_diagnostics_flag_blank() -> None:
    image = Image.new("RGBA", (12, 8), (250, 250, 250, 128))
    first = render_accessibility_previews(image)
    second = render_accessibility_previews(image)

    assert [item.name for item in first.images] == [
        "original",
        "grayscale",
        "protanopia",
        "deuteranopia",
        "tritanopia",
    ]
    assert [item.sha256 for item in first.images] == [item.sha256 for item in second.images]
    report = diagnose_visual(image)
    assert {item.code for item in report.warnings} >= {
        "luminance-contrast",
        "luminance-entropy",
    }
    assert report.to_dict()["previews"][0]["width"] == 12  # type: ignore[index]


def test_manuscript_audit_reports_page_resources_and_security(tmp_path: Path) -> None:
    path = tmp_path / "manuscript.pdf"
    _active_pdf(path, pages=2)

    audit = audit_manuscript_pdf(path)

    assert audit.page_count == 2
    assert audit.page_sizes_uniform is True
    assert audit.has_active_content is True
    assert audit.pages[0].image_xobjects == 1
    assert audit.pages[0].has_transparency is True
    assert audit.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_jats_and_ro_crate_exports_preserve_metadata_and_escape_xml(tmp_path: Path) -> None:
    manifest_path = _bundle(tmp_path / "bundle")
    xml = submission_manifest_to_jats(manifest_path)
    root = ET.fromstring(xml)
    figure = root.find("fig")
    assert figure is not None
    assert figure.findtext("caption/p") == "Accuracy by condition < baseline."
    graphic = figure.find("graphic")
    assert graphic is not None
    assert graphic.findtext("alt-text") == "Three labeled lines increase from left to right."
    assert graphic.attrib["{http://www.w3.org/1999/xlink}href"] == "figure.png"

    crate = submission_manifest_to_ro_crate(
        manifest_path,
        name="Submission",
        creators=["Devraj Jhala"],
        license="https://spdx.org/licenses/MIT.html",
    )
    assert crate["@context"] == RO_CRATE_CONTEXT
    graph = crate["@graph"]
    assert isinstance(graph, list)
    descriptor = next(item for item in graph if item["@id"] == "ro-crate-metadata.json")
    assert descriptor["conformsTo"] == {"@id": "https://w3id.org/ro/crate/1.3"}
    artifact = next(item for item in graph if item["@id"] == "figure.png")
    assert artifact["sha256"] == hashlib.sha256(b"static artifact").hexdigest()
    json.dumps(crate, allow_nan=False)


def test_remediation_plan_is_frozen_deterministic_and_traceable(tmp_path: Path) -> None:
    path = tmp_path / "active.pdf"
    _active_pdf(path)
    inspection = inspect_artifact(path)

    first = plan_remediation(inspection)
    second = plan_remediation(inspection)

    assert first == second
    assert first.remediations[0].kind is RemediationKind.ACTIVE_CONTENT
    assert first.remediations[0].severity == "critical"
    kinds = {item.kind for item in first.remediations}
    assert RemediationKind.TRANSPARENCY in kinds
    assert "Artifact remediation plan" in first.to_markdown()
