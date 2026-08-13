from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from pypdf import PdfWriter
from pypdf.generic import (
    ArrayObject,
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
    NumberObject,
    TextStringObject,
)

from researchplot.manuscript import audit_manuscript_pdf
from researchplot.manuscript_matching import (
    PlacementMatchMethod,
    PlacementMatchStatus,
    match_manuscript_figures,
)
from researchplot.project_api import Project
from researchplot.specs import (
    DeliverableSpec,
    FigureSpec,
    ManuscriptMatchHint,
    ManuscriptSpec,
    ProjectSpec,
)


def _source_png(path: Path) -> Path:
    image = Image.new("RGB", (20, 10), "#3456c0")
    for x in range(10):
        for y in range(10):
            image.putpixel((x, y), (220, 70, 40))
    image.save(path, dpi=(100, 100))
    return path


def _font(writer: PdfWriter) -> object:
    return writer._add_object(
        DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
    )


def _image_pdf(
    path: Path,
    source: Path,
    *,
    uses: int = 1,
    provenance_id: str | None = None,
    caption: str = "Figure 1. Exact placement",
    clipped_by_crop: bool = False,
    rotated: bool = False,
) -> Path:
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        raw = rgb.tobytes()
        width, height = rgb.size
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    if clipped_by_crop:
        page.cropbox.lower_left = (100, 160)
        page.cropbox.upper_right = (200, 200)
    image_stream = DecodedStreamObject()
    image_stream.set_data(raw)
    image_stream.update(
        {
            NameObject("/Type"): NameObject("/XObject"),
            NameObject("/Subtype"): NameObject("/Image"),
            NameObject("/Width"): NumberObject(width),
            NameObject("/Height"): NumberObject(height),
            NameObject("/ColorSpace"): NameObject("/DeviceRGB"),
            NameObject("/BitsPerComponent"): NumberObject(8),
        }
    )
    if provenance_id is not None:
        image_stream[NameObject("/ResearchPlotFigureID")] = TextStringObject(provenance_id)
    image_ref = writer._add_object(image_stream)
    resources = DictionaryObject(
        {
            NameObject("/XObject"): DictionaryObject({NameObject("/Im0"): image_ref}),
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): _font(writer)}),
        }
    )
    page[NameObject("/Resources")] = resources
    first_placement = (
        "q 0 144 -72 0 216 144 cm /Im0 Do Q" if rotated else "q 144 0 0 72 72 144 cm /Im0 Do Q"
    )
    operations = [first_placement]
    if uses == 2:
        operations.append("q 144 0 0 72 300 144 cm /Im0 Do Q")
    operations.append(f"BT /F1 10 Tf 72 120 Td ({caption}) Tj ET")
    content = DecodedStreamObject()
    content.set_data(("\n".join(operations) + "\n").encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(content)
    with path.open("wb") as stream:
        writer.write(stream)
    return path


def _form_pdf(path: Path, *, with_caption: bool = True) -> Path:
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    form = DecodedStreamObject()
    form.set_data(b"0.2 0.4 0.8 rg 0 0 100 50 re f\n")
    form.update(
        {
            NameObject("/Type"): NameObject("/XObject"),
            NameObject("/Subtype"): NameObject("/Form"),
            NameObject("/BBox"): ArrayObject(
                [NumberObject(0), NumberObject(0), NumberObject(100), NumberObject(50)]
            ),
            NameObject("/Resources"): DictionaryObject(),
        }
    )
    form_ref = writer._add_object(form)
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/XObject"): DictionaryObject({NameObject("/Fm0"): form_ref}),
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): _font(writer)}),
        }
    )
    operations = ["q 1 0 0 1 72 144 cm /Fm0 Do Q"]
    if with_caption:
        operations.append("BT /F1 10 Tf 72 120 Td (Figure 2. Vector result) Tj ET")
    content = DecodedStreamObject()
    content.set_data(("\n".join(operations) + "\n").encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(content)
    with path.open("wb") as stream:
        writer.write(stream)
    return path


def _raster_figure(source: Path, figure_id: str = "figure-1") -> FigureSpec:
    return FigureSpec(
        figure_id,
        (DeliverableSpec("main", "png", source, preferred=True),),
        number=1,
        caption="Figure 1. Exact placement",
    )


def _vector_figure(tmp_path: Path) -> FigureSpec:
    return FigureSpec(
        "figure-2",
        (DeliverableSpec("main", "pdf", tmp_path / "source.pdf", preferred=True),),
        number=2,
        caption="Figure 2. Vector result",
    )


def test_exact_raster_fingerprint_measures_placement(tmp_path: Path) -> None:
    source = _source_png(tmp_path / "source.png")
    manuscript = _image_pdf(tmp_path / "paper.pdf", source)

    audit = match_manuscript_figures(manuscript, (_raster_figure(source),))

    assert audit.coverage_complete
    match = audit.matches[0]
    assert match.status is PlacementMatchStatus.MATCHED
    assert match.method is PlacementMatchMethod.RASTER_FINGERPRINT
    assert match.placement is not None
    assert match.placement.page == 1
    assert match.placement.bbox_pt == pytest.approx((72, 144, 216, 216))
    assert match.placement.width_mm == pytest.approx(50.8)
    assert match.placement.height_mm == pytest.approx(25.4)
    assert match.placement.rotation_degrees == pytest.approx(0)
    assert match.placement.effective_dpi_x == pytest.approx(10)
    assert match.to_dict()["placement"] is not None


def test_provenance_id_wins_over_matching_raster(tmp_path: Path) -> None:
    source = _source_png(tmp_path / "source.png")
    manuscript = _image_pdf(tmp_path / "paper.pdf", source, provenance_id="figure-1")

    match = match_manuscript_figures(manuscript, (_raster_figure(source),)).matches[0]

    assert match.resolved
    assert match.method is PlacementMatchMethod.PROVENANCE_ID


def test_repeated_exact_raster_is_ambiguous_and_unresolved(tmp_path: Path) -> None:
    source = _source_png(tmp_path / "source.png")
    manuscript = _image_pdf(tmp_path / "paper.pdf", source, uses=2)

    audit = match_manuscript_figures(manuscript, (_raster_figure(source),))

    assert not audit.coverage_complete
    assert len(audit.unresolved) == 1
    assert audit.matches[0].status is PlacementMatchStatus.AMBIGUOUS
    assert len(audit.matches[0].candidates) == 2


def test_caption_hint_matches_only_unique_top_level_form(tmp_path: Path) -> None:
    manuscript = _form_pdf(tmp_path / "paper.pdf")
    figure = _vector_figure(tmp_path)
    hint = ManuscriptMatchHint("figure-2", pages=(1,), number=2, caption="Figure 2. Vector result")

    match = match_manuscript_figures(manuscript, (figure,), hints=(hint,)).matches[0]

    assert match.resolved
    assert match.method is PlacementMatchMethod.CONFIGURED_HINT
    assert match.placement is not None
    assert match.placement.object_type == "form"
    assert match.placement.width_mm == pytest.approx(100 * 25.4 / 72)
    assert match.placement.height_mm == pytest.approx(50 * 25.4 / 72)


def test_missing_or_unmeasured_hint_never_passes(tmp_path: Path) -> None:
    manuscript = _form_pdf(tmp_path / "paper.pdf", with_caption=False)
    figure = _vector_figure(tmp_path)
    hint = ManuscriptMatchHint("figure-2", caption="caption is absent")

    audit = match_manuscript_figures(manuscript, (figure,), hints=(hint,))

    assert audit.matches[0].status is PlacementMatchStatus.MISSING
    assert not audit.coverage_complete


def test_one_placement_cannot_satisfy_two_figures(tmp_path: Path) -> None:
    source = _source_png(tmp_path / "source.png")
    manuscript = _image_pdf(tmp_path / "paper.pdf", source)
    figures = (_raster_figure(source), _raster_figure(source, "figure-2"))

    audit = match_manuscript_figures(manuscript, figures)

    assert {item.status for item in audit.matches} == {PlacementMatchStatus.AMBIGUOUS}
    assert len(audit.unresolved) == 2


def test_structural_audit_can_include_placement_coverage(tmp_path: Path) -> None:
    source = _source_png(tmp_path / "source.png")
    manuscript = _image_pdf(tmp_path / "paper.pdf", source)

    audit = audit_manuscript_pdf(manuscript, figures=(_raster_figure(source),))

    assert audit.placement_audit is not None
    assert audit.placement_audit.coverage_complete
    assert audit.to_dict()["placement_audit"]


def test_empty_figure_set_is_rejected(tmp_path: Path) -> None:
    manuscript = _form_pdf(tmp_path / "paper.pdf")

    with pytest.raises(ValueError, match="At least one"):
        match_manuscript_figures(manuscript, ())


def test_explicit_page_hint_can_match_without_extractable_caption(tmp_path: Path) -> None:
    manuscript = _form_pdf(tmp_path / "paper.pdf", with_caption=False)
    figure = FigureSpec(
        "figure-2",
        (DeliverableSpec("main", "pdf", tmp_path / "source.pdf", preferred=True),),
    )

    match = match_manuscript_figures(
        manuscript,
        (figure,),
        hints=(ManuscriptMatchHint("figure-2", pages=(1,)),),
    ).matches[0]

    assert match.resolved
    assert match.method is PlacementMatchMethod.CONFIGURED_HINT


def test_provenance_ambiguity_does_not_fall_back_to_fingerprint(tmp_path: Path) -> None:
    source = _source_png(tmp_path / "source.png")
    manuscript = _image_pdf(
        tmp_path / "paper.pdf",
        source,
        uses=2,
        provenance_id="figure-1",
    )

    match = match_manuscript_figures(manuscript, (_raster_figure(source),)).matches[0]

    assert match.status is PlacementMatchStatus.AMBIGUOUS
    assert match.method is PlacementMatchMethod.PROVENANCE_ID


def test_invalid_limits_paths_and_unknown_hints_are_rejected(tmp_path: Path) -> None:
    source = _source_png(tmp_path / "source.png")
    manuscript = _image_pdf(tmp_path / "paper.pdf", source)
    figure = _raster_figure(source)

    with pytest.raises(ValueError, match="max_pages"):
        match_manuscript_figures(manuscript, (figure,), max_pages=0)
    with pytest.raises(FileNotFoundError, match="not found"):
        match_manuscript_figures(tmp_path / "missing.pdf", (figure,))
    with pytest.raises(ValueError, match="unknown figures"):
        match_manuscript_figures(
            manuscript,
            (figure,),
            hints=(ManuscriptMatchHint("unknown", pages=(1,)),),
        )


def test_missing_raster_source_is_disclosed_in_limitations(tmp_path: Path) -> None:
    manuscript = _form_pdf(tmp_path / "paper.pdf")
    missing = tmp_path / "missing.png"
    figure = FigureSpec(
        "figure-2",
        (DeliverableSpec("main", "png", missing, preferred=True),),
        number=2,
        caption="Figure 2. Vector result",
    )

    match = match_manuscript_figures(manuscript, (figure,)).matches[0]

    assert match.resolved
    assert any("does not exist" in item for item in match.limitations)


def test_page_crop_clipping_is_measured_and_disclosed(tmp_path: Path) -> None:
    source = _source_png(tmp_path / "source.png")
    manuscript = _image_pdf(tmp_path / "paper.pdf", source, clipped_by_crop=True)

    audit = match_manuscript_figures(manuscript, (_raster_figure(source),))

    placement = audit.matches[0].placement
    assert placement is not None
    assert placement.clipped_by_page_crop is True
    assert placement.page_crop_box_pt == pytest.approx((100, 160, 200, 200))
    assert any("crop box clips" in warning for warning in audit.warnings)


def test_object_rotation_preserves_intrinsic_placed_dimensions(tmp_path: Path) -> None:
    source = _source_png(tmp_path / "source.png")
    manuscript = _image_pdf(tmp_path / "paper.pdf", source, rotated=True)

    placement = match_manuscript_figures(manuscript, (_raster_figure(source),)).matches[0].placement

    assert placement is not None
    assert placement.rotation_degrees == pytest.approx(90)
    assert placement.width_mm == pytest.approx(50.8)
    assert placement.height_mm == pytest.approx(25.4)
    assert placement.bbox_pt == pytest.approx((144, 144, 216, 288))


def test_project_manuscript_audit_uses_configured_figures(tmp_path: Path) -> None:
    source = _source_png(tmp_path / "source.png")
    manuscript = _image_pdf(tmp_path / "paper.pdf", source)
    figure = FigureSpec(
        "figure-1",
        (DeliverableSpec("main", "png", source, preferred=True),),
        content="photograph",
        number=1,
        caption="Figure 1. Exact placement",
    )
    project = Project(
        ProjectSpec(
            "ieee-journal@2026.08.0",
            (figure,),
            manuscript=ManuscriptSpec(manuscript, "pdf", required=True),
        )
    )

    audit = project.audit_manuscript()

    assert audit.placement_audit is not None
    assert audit.placement_audit.coverage_complete
