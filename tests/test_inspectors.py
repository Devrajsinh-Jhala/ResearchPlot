from __future__ import annotations

import json
import zlib
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import cast

import pytest
from PIL import Image
from pypdf import PdfWriter
from pypdf.generic import (
    ArrayObject,
    DecodedStreamObject,
    DictionaryObject,
    FloatObject,
    NameObject,
    NumberObject,
)

from researchplot.inspectors import (
    ArtifactInspectionError,
    ArtifactParseError,
    Observation,
    UnsupportedArtifactError,
    inspect_artifact,
)


def _numbers(*values: float) -> ArrayObject:
    return ArrayObject([FloatObject(value) for value in values])


def _write_pdf_with_recursive_fonts(path: Path) -> None:
    writer = PdfWriter()
    first = writer.add_blank_page(width=72, height=144)
    second = writer.add_blank_page(width=144, height=72)
    second[NameObject("/CropBox")] = _numbers(0, 0, 140, 70)

    type3 = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type3"),
            NameObject("/Name"): NameObject("/Type3Test"),
            NameObject("/FontBBox"): _numbers(0, 0, 1, 1),
            NameObject("/FontMatrix"): _numbers(0.001, 0, 0, 0.001, 0, 0),
            NameObject("/CharProcs"): DictionaryObject(),
            NameObject("/Encoding"): NameObject("/WinAnsiEncoding"),
            NameObject("/FirstChar"): NumberObject(0),
            NameObject("/LastChar"): NumberObject(0),
            NameObject("/Widths"): _numbers(0),
        }
    )
    type3_ref = writer._add_object(type3)

    unembedded = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/TrueType"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    unembedded_ref = writer._add_object(unembedded)

    font_file = DecodedStreamObject()
    font_file.set_data(b"test font program")
    font_file_ref = writer._add_object(font_file)
    descriptor = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/FontDescriptor"),
            NameObject("/FontName"): NameObject("/EmbeddedTest"),
            NameObject("/Flags"): NumberObject(4),
            NameObject("/FontBBox"): _numbers(0, -200, 1000, 900),
            NameObject("/ItalicAngle"): NumberObject(0),
            NameObject("/Ascent"): NumberObject(800),
            NameObject("/Descent"): NumberObject(-200),
            NameObject("/CapHeight"): NumberObject(700),
            NameObject("/StemV"): NumberObject(80),
            NameObject("/FontFile2"): font_file_ref,
        }
    )
    descriptor_ref = writer._add_object(descriptor)
    embedded = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/TrueType"),
            NameObject("/BaseFont"): NameObject("/EmbeddedTest"),
            NameObject("/FirstChar"): NumberObject(0),
            NameObject("/LastChar"): NumberObject(0),
            NameObject("/Widths"): _numbers(500),
            NameObject("/FontDescriptor"): descriptor_ref,
        }
    )
    embedded_ref = writer._add_object(embedded)

    form = DecodedStreamObject()
    form.set_data(b"")
    form.update(
        {
            NameObject("/Type"): NameObject("/XObject"),
            NameObject("/Subtype"): NameObject("/Form"),
            NameObject("/BBox"): _numbers(0, 0, 10, 10),
            NameObject("/Resources"): DictionaryObject(
                {NameObject("/Font"): DictionaryObject({NameObject("/FEmbedded"): embedded_ref})}
            ),
        }
    )
    form_ref = writer._add_object(form)
    first[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {
                    NameObject("/FType3"): type3_ref,
                    NameObject("/FHelvetica"): unembedded_ref,
                }
            ),
            NameObject("/XObject"): DictionaryObject({NameObject("/NestedForm"): form_ref}),
        }
    )
    second[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/FHelvetica"): unembedded_ref})}
    )
    with path.open("wb") as stream:
        writer.write(stream)


def _write_pdf_with_deep_font_resources(path: Path, depth: int) -> None:
    writer = PdfWriter()
    page = writer.add_blank_page(width=72, height=72)
    nested_font_ref = None
    for index in range(depth):
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type3"),
                NameObject("/Name"): NameObject(f"/Nested{index}"),
            }
        )
        if nested_font_ref is not None:
            font[NameObject("/Resources")] = DictionaryObject(
                {NameObject("/Font"): DictionaryObject({NameObject("/Next"): nested_font_ref})}
            )
        nested_font_ref = writer._add_object(font)
    assert nested_font_ref is not None
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/Root"): nested_font_ref})}
    )
    with path.open("wb") as stream:
        writer.write(stream)


def test_pdf_inspector_covers_every_page_box_and_recursive_font(tmp_path: Path) -> None:
    path = tmp_path / "multipage.pdf"
    _write_pdf_with_recursive_fonts(path)

    result = inspect_artifact(path)

    assert result.format == "pdf"
    assert result.mime_type == "application/pdf"
    assert result.get("pdf.page_count") == 2
    assert result.get("artifact.page_count") == 2
    assert result.get("pdf.page_widths_mm") == pytest.approx((25.4, 140 * 25.4 / 72))
    assert result.get("pdf.page_heights_mm") == pytest.approx((50.8, 70 * 25.4 / 72))
    assert result.get("pdf.page.2.crop_box") == pytest.approx((0, 0, 140, 70))
    assert result.get("pdf.page.2.effective_box_type") == "crop_box"
    assert result.get("pdf.page.2.effective_box") == pytest.approx((0, 0, 140, 70))
    assert result.get("artifact.is_single_page") is False
    assert result.get("artifact.width_mm") is None
    assert "Submit one figure per PDF" in result.warnings[0]
    assert result.get("pdf.font_count") == 3
    assert result.get("pdf.font_resource_occurrences") == 4
    assert result.get("pdf.type3_font_count") == 1
    assert result.get("pdf.embedded_font_count") == 2
    assert result.get("pdf.unembedded_font_count") == 1
    assert result.get("pdf.unembedded_truetype_font_count") == 1
    font_names = cast(tuple[str, ...], result.get("pdf.font_names"))
    font_details = cast(tuple[tuple[str, str, bool, str], ...], result.get("pdf.font_details"))
    assert "/EmbeddedTest" in font_names
    assert any("/XObject" in detail[3] for detail in font_details)


def test_pdf_rotation_changes_displayed_dimensions(tmp_path: Path) -> None:
    path = tmp_path / "rotated.pdf"
    writer = PdfWriter()
    page = writer.add_blank_page(width=72, height=144)
    page.rotate(90)
    with path.open("wb") as stream:
        writer.write(stream)

    result = inspect_artifact(path)

    assert result.get("pdf.page.1.rotation") == 90
    assert result.get("artifact.width_mm") == pytest.approx(50.8)
    assert result.get("artifact.height_mm") == pytest.approx(25.4)


def test_pdf_uses_visible_crop_box_for_single_page_dimensions(tmp_path: Path) -> None:
    path = tmp_path / "cropped.pdf"
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    page[NameObject("/CropBox")] = _numbers(10, 20, 262, 164)
    with path.open("wb") as stream:
        writer.write(stream)

    result = inspect_artifact(path)

    assert result.get("pdf.page.1.effective_box_type") == "crop_box"
    assert result.get("artifact.is_single_page") is True
    assert result.get("artifact.width_mm") == pytest.approx(88.9)
    assert result.get("artifact.height_mm") == pytest.approx(50.8)


def test_pdf_font_resource_recursion_limit_cannot_be_reset(tmp_path: Path) -> None:
    path = tmp_path / "deep-font-resources.pdf"
    _write_pdf_with_deep_font_resources(path, depth=70)

    with pytest.raises(ArtifactParseError, match="resource nesting exceeds 64 levels"):
        inspect_artifact(path)


def test_empty_page_tree_and_malformed_pdf_are_actionable(tmp_path: Path) -> None:
    empty_pages = tmp_path / "empty-pages.pdf"
    writer = PdfWriter()
    with empty_pages.open("wb") as stream:
        writer.write(stream)
    with pytest.raises(ArtifactParseError, match="contains no pages"):
        inspect_artifact(empty_pages)

    malformed = tmp_path / "malformed.pdf"
    malformed.write_bytes(b"%PDF-1.7\nthis is not a PDF")
    with pytest.raises(ArtifactParseError, match="Could not parse PDF"):
        inspect_artifact(malformed)


def test_svg_inspector_reports_dimensions_text_fonts_and_external_links(tmp_path: Path) -> None:
    path = tmp_path / "figure.svg"
    path.write_text(
        """<?xml version="1.0"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
     width="89mm" viewBox="0 0 890 445">
  <style>.series { font-family: Source Sans 3; fill: url(https://example.test/p.svg#p); }</style>
  <text class="series" style="font-family: Arial, sans-serif">Result</text>
  <image href="data:image/png;base64,AAAA" />
  <image href="assets/panel.png" />
  <a xlink:href="#local"><path d="M 0 0" /></a>
</svg>
""",
        encoding="utf-8",
    )

    result = inspect_artifact(path)

    assert result.format == "svg"
    assert result.get("artifact.width_mm") == pytest.approx(89)
    assert result.get("artifact.height_mm") == pytest.approx(44.5)
    assert result.get("svg.view_box") == (0, 0, 890, 445)
    assert result.get("svg.text_element_count") == 1
    assert result.get("svg.has_editable_text") is True
    assert result.get("svg.font_declarations") == (
        "Arial, sans-serif",
        "Source Sans 3",
    )
    assert result.get("svg.external_links") == (
        "assets/panel.png",
        "https://example.test/p.svg#p",
    )


def test_svg_viewbox_alone_does_not_invent_physical_dimensions(tmp_path: Path) -> None:
    path = tmp_path / "viewport.svg"
    path.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 50"/>')

    result = inspect_artifact(path)

    assert result.get("artifact.width_mm") is None
    assert result.get("artifact.height_mm") is None
    assert "viewBox coordinates alone" in result.warnings[0]


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "payload, message",
    [
        (
            '<!DOCTYPE svg [<!ENTITY x "unsafe">]><svg xmlns="http://www.w3.org/2000/svg"/>',
            "DTD and entity declarations",
        ),
        ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 nope 10"/>', "viewBox"),
        ('<svg xmlns="http://www.w3.org/2000/svg"><g></svg>', "Could not parse SVG"),
    ],
)
def test_svg_rejects_unsafe_or_malformed_xml(tmp_path: Path, payload: str, message: str) -> None:
    path = tmp_path / "unsafe.svg"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ArtifactParseError, match=message):
        inspect_artifact(path)


def test_svg_accepts_matplotlib_svg_11_doctype(tmp_path: Path) -> None:
    path = tmp_path / "matplotlib.svg"
    path.write_text(
        '<?xml version="1.0" encoding="utf-8" standalone="no"?>\n'
        '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"\n'
        ' "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n'
        '<svg xmlns="http://www.w3.org/2000/svg" width="89mm" height="55mm" '
        'viewBox="0 0 89 55"><path d="M 0 0 L 1 1" /></svg>',
        encoding="utf-8",
    )

    result = inspect_artifact(path)

    assert result.format == "svg"
    assert result.metadata["artifact.width_mm"] == pytest.approx(89.0)


def test_svg_rejects_excessive_nesting(tmp_path: Path) -> None:
    path = tmp_path / "deep.svg"
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg">' + "<g>" * 256 + "</g>" * 256 + "</svg>",
        encoding="utf-8",
    )

    with pytest.raises(ArtifactParseError, match="nesting exceeds"):
        inspect_artifact(path)


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    ("file_format", "suffix", "save_options", "expected_compression"),
    [
        ("PNG", ".png", {"dpi": (300, 300), "icc_profile": b"test-profile"}, "deflate"),
        ("JPEG", ".jpg", {"dpi": (300, 300), "progressive": True}, "jpeg-progressive"),
        ("TIFF", ".tif", {"dpi": (300, 300), "compression": "tiff_lzw"}, "lzw"),
    ],
)
def test_raster_inspector_reports_pixels_dpi_color_and_compression(
    tmp_path: Path,
    file_format: str,
    suffix: str,
    save_options: dict[str, object],
    expected_compression: str,
) -> None:
    path = tmp_path / f"figure{suffix}"
    Image.new("RGB", (300, 150), color=(10, 20, 30)).save(path, format=file_format, **save_options)

    result = inspect_artifact(path)

    assert result.get("raster.pixel_width") == 300
    assert result.get("raster.pixel_height") == 150
    assert result.get("raster.mode") == "RGB"
    assert result.get("raster.bit_depth") == 8
    assert result.get("raster.dpi_x") == pytest.approx(300, rel=0.002)
    assert result.get("raster.dpi_y") == pytest.approx(300, rel=0.002)
    assert result.get("artifact.width_mm") == pytest.approx(25.4, rel=0.002)
    assert result.get("artifact.height_mm") == pytest.approx(12.7, rel=0.002)
    assert result.get("raster.compression") == expected_compression
    if file_format == "PNG":
        assert result.get("raster.has_icc_profile") is True
        assert result.get("raster.icc_profile_size") == len(b"test-profile")


def test_raster_without_dpi_is_indeterminate_not_invented(tmp_path: Path) -> None:
    path = tmp_path / "no-dpi.png"
    Image.new("L", (20, 10)).save(path)

    result = inspect_artifact(path)

    assert result.get("artifact.width_mm") is None
    assert result.get("raster.dpi_x") is None
    assert "DPI metadata is absent" in result.warnings[0]


def test_content_detection_does_not_trust_extension(tmp_path: Path) -> None:
    path = tmp_path / "actually-png.jpg"
    Image.new("RGB", (10, 10)).save(path, format="PNG")

    result = inspect_artifact(path)

    assert result.format == "png"
    assert result.get("artifact.extension_matches_content") is False
    assert any("suggests jpeg" in warning for warning in result.warnings)


def test_malformed_raster_is_actionable(tmp_path: Path) -> None:
    path = tmp_path / "broken.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\nnot-an-image")

    with pytest.raises(ArtifactParseError, match="Could not parse raster image"):
        inspect_artifact(path)


def test_raster_pixel_cap_is_independent_of_pillow_global(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "oversized-header.png"
    Image.new("RGB", (1, 1)).save(path)
    payload = bytearray(path.read_bytes())
    payload[16:20] = (20_000).to_bytes(4, "big")
    payload[20:24] = (20_000).to_bytes(4, "big")
    payload[29:33] = (zlib.crc32(payload[12:29]) & 0xFFFFFFFF).to_bytes(4, "big")
    path.write_bytes(payload)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", None)

    with pytest.raises(ArtifactParseError, match="exceeds the safety limit"):
        inspect_artifact(path)


def test_eps_inspector_prefers_hires_bounding_box(tmp_path: Path) -> None:
    path = tmp_path / "figure.eps"
    path.write_text(
        """%!PS-Adobe-3.0 EPSF-3.0
%%BoundingBox: 0 0 72 144
%%HiResBoundingBox: 0.25 0.5 71.75 143.5
%%EndComments
showpage
%%EOF
""",
        encoding="latin-1",
    )

    result = inspect_artifact(path)

    assert result.format == "eps"
    assert result.get("eps.bounding_box") == (0, 0, 72, 144)
    assert result.get("eps.hires_bounding_box") == (0.25, 0.5, 71.75, 143.5)
    assert result.get("artifact.width_mm") == pytest.approx(71.5 * 25.4 / 72)
    assert result.get("artifact.height_mm") == pytest.approx(143 * 25.4 / 72)


def test_eps_atend_box_is_found_in_trailer(tmp_path: Path) -> None:
    path = tmp_path / "trailer.eps"
    path.write_text(
        """%!PS-Adobe-3.0 EPSF-3.0
%%BoundingBox: (atend)
%%EndComments
showpage
%%Trailer
%%BoundingBox: -10 -20 62 124
%%EOF
""",
        encoding="latin-1",
    )

    result = inspect_artifact(path)

    assert result.get("eps.bounding_box") == (-10, -20, 62, 124)
    assert result.get("artifact.width_mm") == pytest.approx(25.4)
    assert result.get("artifact.height_mm") == pytest.approx(50.8)


def test_eps_without_a_concrete_box_returns_warning(tmp_path: Path) -> None:
    path = tmp_path / "unknown-size.eps"
    path.write_text(
        "%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: (atend)\nshowpage\n%%EOF\n",
        encoding="latin-1",
    )

    result = inspect_artifact(path)

    assert result.get("artifact.width_mm") is None
    assert "neither BoundingBox nor HiResBoundingBox" in result.warnings[0]


def test_missing_empty_and_unsupported_artifacts_have_specific_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        inspect_artifact(tmp_path / "missing.pdf")

    empty = tmp_path / "empty.pdf"
    empty.touch()
    with pytest.raises(ArtifactParseError, match="empty"):
        inspect_artifact(empty)

    unknown = tmp_path / "figure.dat"
    unknown.write_bytes(b"not a supported artifact")
    with pytest.raises(UnsupportedArtifactError, match="Expected PDF, SVG, PNG"):
        inspect_artifact(unknown)


def test_inspection_models_are_immutable_and_json_serializable(tmp_path: Path) -> None:
    path = tmp_path / "small.png"
    Image.new("RGB", (2, 2)).save(path)
    result = inspect_artifact(path)

    with pytest.raises(FrozenInstanceError):
        result.format = "jpeg"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.observations[0].key = "changed"  # type: ignore[misc]
    assert isinstance(result.observations[0], Observation)
    assert json.loads(json.dumps(result.to_dict()))["format"] == "png"
    assert isinstance(ArtifactInspectionError("message"), ValueError)
