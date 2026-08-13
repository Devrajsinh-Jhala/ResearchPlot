"""Conservative figure-placement matching for compiled manuscript PDFs.

The matcher is intentionally passive and evidence ordered.  It reads PDF object
and content streams but never executes actions, follows links, performs OCR, or
contacts a service.  A result is resolved only when the strongest available
evidence identifies exactly one measurable placed object.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeAlias, cast

from PIL import Image, ImageOps, UnidentifiedImageError
from pypdf import PdfReader
from pypdf.errors import PyPdfError
from pypdf.generic import ContentStream

from .api_types import EvidenceConfidence
from .inspectors import ArtifactParseError, _object_key, _resolve_pdf, inspect_artifact
from .models import OutputFormat
from .specs import FigureSpec, ManuscriptMatchHint

_MM_PER_POINT = 25.4 / 72.0
_MAX_PAGES = 2_000
_MAX_CONTENT_OPERATIONS_PER_PAGE = 500_000
_MAX_XOBJECT_DEPTH = 32
_MAX_XOBJECTS_PER_PAGE = 20_000
_MAX_METADATA_BYTES = 1_048_576
_MAX_SOURCE_BYTES = 512 * 1024 * 1024
_MAX_IMAGE_PIXELS = 100_000_000
_FIGURE_ID = re.compile(r"^[a-z0-9]+(?:[a-z0-9_-]*[a-z0-9])?$")
_XMP_ATTRIBUTE = re.compile(
    rb"(?:researchplot|rp):figure[_-]?id\s*=\s*[\"']([^\"']{1,128})[\"']",
    re.IGNORECASE,
)
_XMP_ELEMENT = re.compile(
    rb"<(?:researchplot|rp):(?:figure[_-]?id|FigureID)\b[^>]*>\s*"
    rb"([^<]{1,128})\s*</(?:researchplot|rp):(?:figure[_-]?id|FigureID)\s*>",
    re.IGNORECASE,
)
_PROVENANCE_KEYS = (
    "/ResearchPlotFigureID",
    "/ResearchPlotFigureId",
    "/ResearchPlotProvenanceID",
)
_RASTER_FORMATS = {
    OutputFormat.PNG,
    OutputFormat.JPEG,
    OutputFormat.TIFF,
}

Matrix: TypeAlias = tuple[float, float, float, float, float, float]
BBox: TypeAlias = tuple[float, float, float, float]


class PlacementMatchMethod(StrEnum):
    """Evidence that associated a configured figure with a placed object."""

    PROVENANCE_ID = "provenance_id"
    RASTER_FINGERPRINT = "raster_fingerprint"
    CONFIGURED_HINT = "configured_hint"


class PlacementMatchStatus(StrEnum):
    """Resolution state for one configured figure."""

    MATCHED = "matched"
    AMBIGUOUS = "ambiguous"
    MISSING = "missing"
    UNMEASURED = "unmeasured"


@dataclass(frozen=True, slots=True)
class PlacedObject:
    """One image or Form XObject invocation measured in page user space."""

    page: int
    object_name: str
    object_type: str
    bbox_pt: BBox
    width_mm: float
    height_mm: float
    rotation_degrees: float
    page_rotation_degrees: int
    depth: int
    source_width_px: int | None = None
    source_height_px: int | None = None
    effective_dpi_x: float | None = None
    effective_dpi_y: float | None = None
    raster_fingerprint: str | None = field(default=None, repr=False)
    provenance_ids: tuple[str, ...] = ()
    page_crop_box_pt: BBox | None = None
    clipped_by_page_crop: bool | None = None

    @property
    def key(self) -> tuple[object, ...]:
        return (
            self.page,
            self.object_name,
            *(round(value, 6) for value in self.bbox_pt),
        )

    def to_dict(self) -> dict[str, object]:
        left, bottom, right, top = self.bbox_pt
        return {
            "page": self.page,
            "object_name": self.object_name,
            "object_type": self.object_type,
            "bbox_pt": {
                "left": left,
                "bottom": bottom,
                "right": right,
                "top": top,
            },
            "bbox_mm": {
                "left": left * _MM_PER_POINT,
                "bottom": bottom * _MM_PER_POINT,
                "right": right * _MM_PER_POINT,
                "top": top * _MM_PER_POINT,
            },
            "placed_width_mm": self.width_mm,
            "placed_height_mm": self.height_mm,
            "rotation_degrees": self.rotation_degrees,
            "page_rotation_degrees": self.page_rotation_degrees,
            "depth": self.depth,
            "source_width_px": self.source_width_px,
            "source_height_px": self.source_height_px,
            "effective_dpi_x": self.effective_dpi_x,
            "effective_dpi_y": self.effective_dpi_y,
            "provenance_ids": list(self.provenance_ids),
            "page_crop_box_pt": (
                {
                    "left": self.page_crop_box_pt[0],
                    "bottom": self.page_crop_box_pt[1],
                    "right": self.page_crop_box_pt[2],
                    "top": self.page_crop_box_pt[3],
                }
                if self.page_crop_box_pt is not None
                else None
            ),
            "clipped_by_page_crop": self.clipped_by_page_crop,
        }


@dataclass(frozen=True, slots=True)
class FigurePlacementMatch:
    """Placement evidence and resolution state for one logical figure."""

    figure_id: str
    status: PlacementMatchStatus
    method: PlacementMatchMethod | None
    confidence: EvidenceConfidence | None
    placement: PlacedObject | None
    candidates: tuple[PlacedObject, ...]
    detail: str
    source_path: Path | None = None
    source_width_mm: float | None = None
    source_height_mm: float | None = None
    scale_x: float | None = None
    scale_y: float | None = None
    limitations: tuple[str, ...] = ()

    @property
    def resolved(self) -> bool:
        return self.status is PlacementMatchStatus.MATCHED and self.placement is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "figure_id": self.figure_id,
            "status": self.status.value,
            "method": self.method.value if self.method is not None else None,
            "confidence": self.confidence.value if self.confidence is not None else None,
            "placement": self.placement.to_dict() if self.placement is not None else None,
            "candidates": [item.to_dict() for item in self.candidates],
            "detail": self.detail,
            "source_path": str(self.source_path) if self.source_path is not None else None,
            "source_width_mm": self.source_width_mm,
            "source_height_mm": self.source_height_mm,
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True, slots=True)
class ManuscriptPlacementAudit:
    """Coverage report for all configured figure placements in one manuscript."""

    path: Path
    sha256: str
    matches: tuple[FigurePlacementMatch, ...]
    placed_objects: tuple[PlacedObject, ...]
    warnings: tuple[str, ...] = ()

    @property
    def unresolved(self) -> tuple[FigurePlacementMatch, ...]:
        return tuple(item for item in self.matches if not item.resolved)

    @property
    def coverage_complete(self) -> bool:
        return bool(self.matches) and not self.unresolved

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "coverage": "complete" if self.coverage_complete else "indeterminate",
            "matched": sum(item.resolved for item in self.matches),
            "unresolved": len(self.unresolved),
            "matches": [item.to_dict() for item in self.matches],
            "placed_objects": [item.to_dict() for item in self.placed_objects],
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class _PageCollection:
    objects: list[PlacedObject]
    warnings: list[str]
    decoded_images: dict[tuple[str, int, int], str | None]
    object_count: int = 0
    operation_count: int = 0


@dataclass(frozen=True, slots=True)
class _SourceImage:
    path: Path
    fingerprint: str
    width_mm: float | None
    height_mm: float | None


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _multiply(left: Matrix, right: Matrix) -> Matrix:
    """Return the PDF affine product used for ``cm`` concatenation."""

    a, b, c, d, e, f = left
    g, h, i, j, k, last = right
    return (
        a * g + b * i,
        a * h + b * j,
        c * g + d * i,
        c * h + d * j,
        e * g + f * i + k,
        e * h + f * j + last,
    )


def _matrix(value: object) -> Matrix:
    resolved = _resolve_pdf(value)
    if not isinstance(resolved, (list, tuple)) or len(resolved) != 6:
        return (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    try:
        return cast(Matrix, tuple(float(item) for item in resolved))
    except (TypeError, ValueError):
        return (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def _transform(matrix: Matrix, x: float, y: float) -> tuple[float, float]:
    a, b, c, d, e, f = matrix
    return x * a + y * c + e, x * b + y * d + f


def _measure(matrix: Matrix, local_bbox: BBox) -> tuple[BBox, float, float, float]:
    x0, y0, x1, y1 = local_bbox
    origin = _transform(matrix, x0, y0)
    x_corner = _transform(matrix, x1, y0)
    y_corner = _transform(matrix, x0, y1)
    far_corner = _transform(matrix, x1, y1)
    points = (origin, x_corner, y_corner, far_corner)
    bbox = (
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
    )
    width_pt = math.dist(origin, x_corner)
    height_pt = math.dist(origin, y_corner)
    rotation = math.degrees(math.atan2(x_corner[1] - origin[1], x_corner[0] - origin[0]))
    return bbox, width_pt, height_pt, rotation % 360.0


def _outside_crop(box: BBox, crop: BBox | None) -> bool | None:
    if crop is None:
        return None
    return box[0] < crop[0] or box[1] < crop[1] or box[2] > crop[2] or box[3] > crop[3]


def _bbox(value: object) -> BBox | None:
    resolved = _resolve_pdf(value)
    if not isinstance(resolved, (list, tuple)) or len(resolved) != 4:
        return None
    try:
        result = cast(BBox, tuple(float(item) for item in resolved))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in result):
        return None
    if result[2] <= result[0] or result[3] <= result[1]:
        return None
    return result


def _clean_figure_id(value: object) -> str | None:
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return None
    selected = str(value).strip() if value is not None else ""
    return selected if _FIGURE_ID.fullmatch(selected) else None


def _provenance_ids(value_ref: Any) -> tuple[str, ...]:
    value = _resolve_pdf(value_ref)
    if not hasattr(value, "get"):
        return ()
    found: set[str] = set()
    for key in _PROVENANCE_KEYS:
        selected = _clean_figure_id(value.get(key))
        if selected is not None:
            found.add(selected)
    metadata = _resolve_pdf(value.get("/Metadata"))
    if hasattr(metadata, "get_data"):
        try:
            raw = bytes(metadata.get_data())
        except (OSError, TypeError, ValueError, NotImplementedError):
            raw = b""
        if len(raw) <= _MAX_METADATA_BYTES:
            for pattern in (_XMP_ATTRIBUTE, _XMP_ELEMENT):
                for match in pattern.finditer(raw):
                    selected = _clean_figure_id(match.group(1))
                    if selected is not None:
                        found.add(selected)
    return tuple(sorted(found))


def _image_fingerprint(image: Image.Image) -> str:
    normalized = ImageOps.exif_transpose(image)
    if normalized.width * normalized.height > _MAX_IMAGE_PIXELS:
        raise ValueError(f"image exceeds {_MAX_IMAGE_PIXELS} decoded pixels")
    with normalized.convert("RGBA") as rgba:
        digest = hashlib.sha256()
        digest.update(f"{rgba.width}x{rgba.height}:RGBA\0".encode())
        digest.update(rgba.tobytes())
        return digest.hexdigest()


def _source_dimensions(path: Path) -> tuple[float | None, float | None]:
    try:
        inspection = inspect_artifact(path)
    except Exception:  # A dimension is optional; matching evidence remains independently valid.
        return None, None
    width = inspection.get("artifact.width_mm")
    height = inspection.get("artifact.height_mm")
    width_mm = float(width) if isinstance(width, (int, float)) else None
    height_mm = float(height) if isinstance(height, (int, float)) else None
    return width_mm, height_mm


def _source_image(path: Path) -> _SourceImage:
    if path.stat().st_size > _MAX_SOURCE_BYTES:
        raise ValueError(f"source image exceeds {_MAX_SOURCE_BYTES} bytes")
    try:
        with Image.open(path) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("multipage raster sources cannot be matched as one figure")
            fingerprint = _image_fingerprint(image)
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"source image could not be decoded: {exc}") from exc
    width_mm, height_mm = _source_dimensions(path)
    return _SourceImage(path, fingerprint, width_mm, height_mm)


def _embedded_image_fingerprint(
    value_ref: Any, cache: dict[tuple[str, int, int], str | None]
) -> str | None:
    value = _resolve_pdf(value_ref)
    key = _object_key(value_ref, value)
    if key in cache:
        return cache[key]
    width = value.get("/Width") if hasattr(value, "get") else None
    height = value.get("/Height") if hasattr(value, "get") else None
    try:
        pixels = int(cast(Any, width)) * int(cast(Any, height))
    except (TypeError, ValueError):
        cache[key] = None
        return None
    if pixels <= 0 or pixels > _MAX_IMAGE_PIXELS:
        cache[key] = None
        return None
    try:
        # pypdf exposes decoded PIL images on PageObject.images but not for a specific
        # nested invocation.  Its internal decoder is used only after strict pixel caps.
        try:
            decoder_module = importlib.import_module("pypdf.generic._image_xobject")
        except ModuleNotFoundError:
            decoder_module = importlib.import_module("pypdf._xobj_image_helpers")
        decoded = decoder_module._xobj_to_image(value)[2]
        fingerprint = _image_fingerprint(decoded)
    except Exception:  # Unsupported filters/colorspaces become unavailable evidence.
        fingerprint = None
    cache[key] = fingerprint
    return fingerprint


def _resolve_property(resources: Any, operand: object) -> Any:
    if hasattr(operand, "get"):
        return operand
    properties = _resolve_pdf(resources.get("/Properties")) if hasattr(resources, "get") else None
    return _resolve_pdf(cast(Any, properties).get(operand)) if hasattr(properties, "get") else None


def _walk_content(
    stream_ref: Any,
    resources_ref: Any,
    reader: PdfReader,
    collection: _PageCollection,
    *,
    page: int,
    page_rotation: int,
    page_crop_box: BBox | None,
    ctm: Matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
    depth: int = 0,
    prefix: str = "",
    active_forms: frozenset[tuple[str, int, int]] = frozenset(),
) -> None:
    if depth > _MAX_XOBJECT_DEPTH:
        raise ArtifactParseError(
            f"Manuscript page {page} exceeds {_MAX_XOBJECT_DEPTH} nested Form XObjects."
        )
    resources = _resolve_pdf(resources_ref)
    if not hasattr(resources, "get"):
        return
    try:
        content = ContentStream(stream_ref, reader)
    except (OSError, PyPdfError, TypeError, ValueError) as exc:
        raise ArtifactParseError(f"Could not parse content stream on page {page}: {exc}") from exc
    current = ctm
    graphics_stack: list[Matrix] = []
    marked_stack: list[tuple[str, ...]] = []
    for operands, operator in content.operations:
        collection.operation_count += 1
        if collection.operation_count > _MAX_CONTENT_OPERATIONS_PER_PAGE:
            raise ArtifactParseError(
                f"Manuscript page {page} exceeds {_MAX_CONTENT_OPERATIONS_PER_PAGE} "
                "content operations."
            )
        if operator == b"q":
            graphics_stack.append(current)
            continue
        if operator == b"Q":
            if graphics_stack:
                current = graphics_stack.pop()
            else:
                collection.warnings.append(f"Page {page} contains an unmatched Q operator.")
            continue
        if operator == b"cm" and len(operands) >= 6:
            try:
                operand_matrix = cast(Matrix, tuple(float(item) for item in operands[:6]))
            except (TypeError, ValueError):
                collection.warnings.append(f"Page {page} contains a malformed cm operator.")
                continue
            current = _multiply(operand_matrix, current)
            continue
        if operator == b"BMC":
            marked_stack.append(())
            continue
        if operator == b"BDC":
            properties = _resolve_property(resources, operands[1]) if len(operands) >= 2 else None
            marked_stack.append(_provenance_ids(properties))
            continue
        if operator == b"EMC":
            if marked_stack:
                marked_stack.pop()
            continue
        if operator == b"INLINE IMAGE":
            collection.warnings.append(
                f"Page {page} contains an inline image whose placement is not fingerprinted."
            )
            continue
        if operator != b"Do" or not operands:
            continue

        xobjects = _resolve_pdf(resources.get("/XObject"))
        object_ref = xobjects.get(operands[0]) if hasattr(xobjects, "get") else None
        obj = _resolve_pdf(object_ref)
        if not hasattr(obj, "get"):
            collection.warnings.append(
                f"Page {page} references missing XObject {str(operands[0])!r}."
            )
            continue
        collection.object_count += 1
        if collection.object_count > _MAX_XOBJECTS_PER_PAGE:
            raise ArtifactParseError(
                f"Manuscript page {page} exceeds {_MAX_XOBJECTS_PER_PAGE} XObject invocations."
            )
        name = str(operands[0]).lstrip("/")
        object_name = f"{prefix}/{name}" if prefix else name
        subtype = str(obj.get("/Subtype", ""))
        direct_ids = set(_provenance_ids(obj))
        for values in marked_stack:
            direct_ids.update(values)

        if subtype == "/Image":
            bbox, width_pt, height_pt, rotation = _measure(current, (0.0, 0.0, 1.0, 1.0))
            source_width = obj.get("/Width")
            source_height = obj.get("/Height")
            source_width_px: int | None
            source_height_px: int | None
            try:
                source_width_px = int(cast(Any, source_width))
                source_height_px = int(cast(Any, source_height))
            except (TypeError, ValueError):
                source_width_px = source_height_px = None
            fingerprint = _embedded_image_fingerprint(object_ref, collection.decoded_images)
            collection.objects.append(
                PlacedObject(
                    page,
                    object_name,
                    "image",
                    bbox,
                    width_pt * _MM_PER_POINT,
                    height_pt * _MM_PER_POINT,
                    rotation,
                    page_rotation,
                    depth,
                    source_width_px,
                    source_height_px,
                    source_width_px / (width_pt / 72.0)
                    if source_width_px is not None and width_pt > 0
                    else None,
                    source_height_px / (height_pt / 72.0)
                    if source_height_px is not None and height_pt > 0
                    else None,
                    fingerprint,
                    tuple(sorted(direct_ids)),
                    page_crop_box,
                    _outside_crop(bbox, page_crop_box),
                )
            )
            continue

        if subtype != "/Form":
            continue
        form_key = _object_key(object_ref, obj)
        if form_key in active_forms:
            collection.warnings.append(
                f"Page {page} contains a recursive Form XObject at {object_name!r}."
            )
            continue
        form_matrix = _matrix(obj.get("/Matrix"))
        form_ctm = _multiply(form_matrix, current)
        local_bbox = _bbox(obj.get("/BBox"))
        if local_bbox is not None:
            measured_bbox, width_pt, height_pt, rotation = _measure(form_ctm, local_bbox)
            collection.objects.append(
                PlacedObject(
                    page,
                    object_name,
                    "form",
                    measured_bbox,
                    width_pt * _MM_PER_POINT,
                    height_pt * _MM_PER_POINT,
                    rotation,
                    page_rotation,
                    depth,
                    provenance_ids=tuple(sorted(direct_ids)),
                    page_crop_box_pt=page_crop_box,
                    clipped_by_page_crop=_outside_crop(measured_bbox, page_crop_box),
                )
            )
        nested_resources = obj.get("/Resources") or resources
        _walk_content(
            obj,
            nested_resources,
            reader,
            collection,
            page=page,
            page_rotation=page_rotation,
            page_crop_box=page_crop_box,
            ctm=form_ctm,
            depth=depth + 1,
            prefix=object_name,
            active_forms=active_forms | {form_key},
        )


def _collect_page(
    reader: PdfReader, page_number: int
) -> tuple[tuple[PlacedObject, ...], tuple[str, ...]]:
    page = reader.pages[page_number - 1]
    rotation = int(page.get("/Rotate", 0) or 0) % 360
    crop_box = _bbox(tuple(page.cropbox))
    collection = _PageCollection([], [], {})
    contents = page.get_contents()
    if contents is not None:
        _walk_content(
            contents,
            page.get("/Resources"),
            reader,
            collection,
            page=page_number,
            page_rotation=rotation,
            page_crop_box=crop_box,
        )
    if any(item.clipped_by_page_crop for item in collection.objects):
        collection.warnings.append(
            f"Page {page_number} crop box clips one or more measured XObject bounds."
        )
    return tuple(collection.objects), tuple(dict.fromkeys(collection.warnings))


def _normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _number_pattern(number: int | str) -> re.Pattern[str]:
    selected = _normalize_text(str(number))
    if re.match(r"^(?:fig(?:ure)?\.?\s+)", selected):
        return re.compile(rf"(?<!\w){re.escape(selected)}(?!\w)")
    return re.compile(rf"\bfig(?:ure)?\.?\s*{re.escape(selected)}(?!\w)")


def _matching_pages(
    texts: Mapping[int, str],
    *,
    pages: tuple[int, ...],
    number: int | str | None,
    caption: str | None,
) -> tuple[int, ...]:
    allowed = set(pages) if pages else set(texts)
    normalized_caption = _normalize_text(caption) if caption is not None else None
    number_pattern = _number_pattern(number) if number is not None else None
    matches: list[int] = []
    for page in sorted(allowed):
        text = texts.get(page)
        if text is None:
            continue
        if normalized_caption is not None and normalized_caption not in text:
            continue
        if number_pattern is not None and number_pattern.search(text) is None:
            continue
        matches.append(page)
    return tuple(matches)


def _existing_source(figure: FigureSpec) -> Path | None:
    preferred = next(
        (
            deliverable.path
            for deliverable in figure.deliverables
            if deliverable.preferred and deliverable.path is not None and deliverable.path.is_file()
        ),
        None,
    )
    if preferred is not None:
        return preferred
    return next(
        (
            deliverable.path
            for deliverable in figure.deliverables
            if deliverable.path is not None and deliverable.path.is_file()
        ),
        None,
    )


def _resolved_match(
    figure_id: str,
    method: PlacementMatchMethod,
    confidence: EvidenceConfidence,
    candidates: Sequence[PlacedObject],
    detail: str,
    *,
    source: Path | None = None,
    source_dimensions: tuple[float | None, float | None] = (None, None),
    limitations: tuple[str, ...] = (),
) -> FigurePlacementMatch:
    unique = {candidate.key: candidate for candidate in candidates}
    selected = tuple(unique.values())
    if len(selected) > 1:
        return FigurePlacementMatch(
            figure_id,
            PlacementMatchStatus.AMBIGUOUS,
            method,
            confidence,
            None,
            selected,
            f"{detail} identified {len(selected)} possible placements; no placement was chosen.",
            source,
            *source_dimensions,
            limitations=limitations,
        )
    if not selected:
        return FigurePlacementMatch(
            figure_id,
            PlacementMatchStatus.MISSING,
            method,
            confidence,
            None,
            (),
            f"{detail} did not identify a placed object.",
            source,
            *source_dimensions,
            limitations=limitations,
        )
    placement = selected[0]
    source_width, source_height = source_dimensions
    return FigurePlacementMatch(
        figure_id,
        PlacementMatchStatus.MATCHED,
        method,
        confidence,
        placement,
        selected,
        detail,
        source,
        source_width,
        source_height,
        placement.width_mm / source_width if source_width and source_width > 0 else None,
        placement.height_mm / source_height if source_height and source_height > 0 else None,
        limitations,
    )


def _figure_match(
    figure: FigureSpec,
    hint: ManuscriptMatchHint | None,
    placements: tuple[PlacedObject, ...],
    texts: Mapping[int, str],
    source_images: tuple[_SourceImage, ...],
    source_warnings: tuple[str, ...],
) -> FigurePlacementMatch:
    source = _existing_source(figure)
    source_dimensions = _source_dimensions(source) if source is not None else (None, None)

    provenance = tuple(
        placement for placement in placements if figure.id in placement.provenance_ids
    )
    if provenance:
        return _resolved_match(
            figure.id,
            PlacementMatchMethod.PROVENANCE_ID,
            EvidenceConfidence.DETERMINISTIC,
            provenance,
            f"Embedded ResearchPlot provenance ID {figure.id!r}",
            source=source,
            source_dimensions=source_dimensions,
            limitations=source_warnings,
        )

    source_by_fingerprint = {item.fingerprint: item for item in source_images}
    raster = tuple(
        placement
        for placement in placements
        if placement.raster_fingerprint in source_by_fingerprint
    )
    if raster:
        matching_source = source_by_fingerprint[cast(str, raster[0].raster_fingerprint)]
        return _resolved_match(
            figure.id,
            PlacementMatchMethod.RASTER_FINGERPRINT,
            EvidenceConfidence.DETERMINISTIC,
            raster,
            "Exact decoded-pixel SHA-256 fingerprint",
            source=matching_source.path,
            source_dimensions=(matching_source.width_mm, matching_source.height_mm),
            limitations=source_warnings,
        )

    pages = hint.pages if hint is not None else ()
    number = hint.number if hint is not None and hint.number is not None else figure.number
    caption = hint.caption if hint is not None and hint.caption is not None else figure.caption
    if not pages and number is None and caption is None:
        return FigurePlacementMatch(
            figure.id,
            PlacementMatchStatus.MISSING,
            None,
            None,
            None,
            (),
            "No embedded provenance, exact raster fingerprint, or configured matching hint "
            "was available.",
            source,
            *source_dimensions,
            limitations=source_warnings,
        )

    matched_pages = _matching_pages(
        texts,
        pages=pages,
        number=number,
        caption=caption,
    )
    if not matched_pages:
        return FigurePlacementMatch(
            figure.id,
            PlacementMatchStatus.MISSING,
            PlacementMatchMethod.CONFIGURED_HINT,
            EvidenceConfidence.MANUAL,
            None,
            (),
            "Configured page/number/caption hints did not match manuscript text.",
            source,
            *source_dimensions,
            limitations=source_warnings,
        )
    candidates = tuple(
        placement
        for placement in placements
        if placement.page in matched_pages and placement.depth == 0
    )
    if not candidates:
        return FigurePlacementMatch(
            figure.id,
            PlacementMatchStatus.UNMEASURED,
            PlacementMatchMethod.CONFIGURED_HINT,
            EvidenceConfidence.MANUAL,
            None,
            (),
            "Hints identified manuscript page(s) "
            + ", ".join(str(page) for page in matched_pages)
            + ", but no top-level measurable image or Form XObject was found.",
            source,
            *source_dimensions,
            limitations=source_warnings,
        )
    return _resolved_match(
        figure.id,
        PlacementMatchMethod.CONFIGURED_HINT,
        EvidenceConfidence.MANUAL,
        candidates,
        "Configured hints and a unique top-level graphical object on page(s) "
        + ", ".join(str(page) for page in matched_pages),
        source=source,
        source_dimensions=source_dimensions,
        limitations=source_warnings
        + (
            "Text hints cannot distinguish a caption from an in-text reference; the "
            "association remains author-configured evidence.",
        ),
    )


def _source_images(figure: FigureSpec) -> tuple[tuple[_SourceImage, ...], tuple[str, ...]]:
    images: list[_SourceImage] = []
    warnings: list[str] = []
    for deliverable in figure.deliverables:
        output_format = cast(OutputFormat, deliverable.format)
        if output_format not in _RASTER_FORMATS or deliverable.path is None:
            continue
        path = deliverable.path
        if not path.is_file():
            warnings.append(f"Raster deliverable {deliverable.id!r} does not exist: {path}")
            continue
        try:
            images.append(_source_image(path))
        except (OSError, ValueError) as exc:
            warnings.append(f"Raster deliverable {deliverable.id!r} was not fingerprinted: {exc}")
    return tuple(images), tuple(warnings)


def _extract_page_text(reader: PdfReader, page_number: int) -> tuple[str, str | None]:
    try:
        raw = reader.pages[page_number - 1].extract_text() or ""
    except (OSError, PyPdfError, TypeError, ValueError) as exc:
        return "", f"Text extraction failed on page {page_number}: {exc}"
    return _normalize_text(raw), None


def _resolve_assignment_collisions(
    matches: tuple[FigurePlacementMatch, ...],
) -> tuple[FigurePlacementMatch, ...]:
    claimed: dict[tuple[object, ...], list[str]] = {}
    for match in matches:
        if match.resolved:
            assert match.placement is not None
            claimed.setdefault(match.placement.key, []).append(match.figure_id)
    conflicts = {key: ids for key, ids in claimed.items() if len(ids) > 1}
    if not conflicts:
        return matches
    result: list[FigurePlacementMatch] = []
    for match in matches:
        if not match.resolved or match.placement is None or match.placement.key not in conflicts:
            result.append(match)
            continue
        ids = conflicts[match.placement.key]
        result.append(
            replace(
                match,
                status=PlacementMatchStatus.AMBIGUOUS,
                placement=None,
                detail=(
                    "One placed object was claimed by multiple configured figures: "
                    + ", ".join(ids)
                    + ". No assignment was chosen."
                ),
            )
        )
    return tuple(result)


def match_manuscript_figures(
    path: str | Path,
    figures: Sequence[FigureSpec],
    *,
    hints: Sequence[ManuscriptMatchHint] = (),
    max_pages: int = _MAX_PAGES,
) -> ManuscriptPlacementAudit:
    """Match configured figures to unique measurable placements in a compiled PDF.

    Evidence priority is embedded ResearchPlot provenance, exact decoded raster
    fingerprints, then author-configured page/number/caption hints.  Ambiguous
    evidence is never replaced by weaker evidence.
    """

    if max_pages <= 0 or max_pages > _MAX_PAGES:
        raise ValueError(f"max_pages must be between 1 and {_MAX_PAGES}.")
    selected_figures = tuple(figures)
    if not selected_figures:
        raise ValueError("At least one configured figure is required for placement matching.")
    if not all(isinstance(item, FigureSpec) for item in selected_figures):
        raise TypeError("figures must contain FigureSpec values.")
    figure_ids = [item.id for item in selected_figures]
    if len(figure_ids) != len(set(figure_ids)):
        raise ValueError("Configured figure IDs must be unique.")
    selected_hints = tuple(hints)
    if not all(isinstance(item, ManuscriptMatchHint) for item in selected_hints):
        raise TypeError("hints must contain ManuscriptMatchHint values.")
    hint_map = {item.figure_id: item for item in selected_hints}
    if len(hint_map) != len(selected_hints):
        raise ValueError("Only one manuscript matching hint may be supplied per figure.")
    unknown = set(hint_map) - set(figure_ids)
    if unknown:
        raise ValueError("Matching hints reference unknown figures: " + ", ".join(sorted(unknown)))

    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"Manuscript PDF not found: {file_path}")
    try:
        reader = PdfReader(file_path, strict=False)
        if len(reader.pages) > max_pages:
            raise ArtifactParseError(
                f"Manuscript has {len(reader.pages)} pages; configured matching limit is "
                f"{max_pages}."
            )
        placements: list[PlacedObject] = []
        warnings: list[str] = []
        texts: dict[int, str] = {}
        for page_number in range(1, len(reader.pages) + 1):
            page_objects, page_warnings = _collect_page(reader, page_number)
            placements.extend(page_objects)
            warnings.extend(page_warnings)
            text, text_warning = _extract_page_text(reader, page_number)
            texts[page_number] = text
            if text_warning is not None:
                warnings.append(text_warning)
    except ArtifactParseError:
        raise
    except (OSError, PyPdfError, KeyError, TypeError, ValueError) as exc:
        raise ArtifactParseError(
            f"Could not match figures in manuscript PDF {file_path}: {exc}"
        ) from exc

    source_evidence = {figure.id: _source_images(figure) for figure in selected_figures}
    matches = tuple(
        _figure_match(
            figure,
            hint_map.get(figure.id),
            tuple(placements),
            texts,
            *source_evidence[figure.id],
        )
        for figure in selected_figures
    )
    matches = _resolve_assignment_collisions(matches)

    resolved_pages = [
        cast(PlacedObject, match.placement).page for match in matches if match.resolved
    ]
    if len(resolved_pages) == len(matches) and resolved_pages != sorted(resolved_pages):
        warnings.append(
            "Resolved placements do not follow configured figure order; review numbering and "
            "caption order."
        )
    numbers = [str(figure.number) for figure in selected_figures if figure.number is not None]
    duplicates = sorted({number for number in numbers if numbers.count(number) > 1})
    if duplicates:
        warnings.append("Configured figure numbers are duplicated: " + ", ".join(duplicates) + ".")

    return ManuscriptPlacementAudit(
        file_path,
        _file_digest(file_path),
        matches,
        tuple(placements),
        tuple(dict.fromkeys(warnings)),
    )


__all__ = [
    "FigurePlacementMatch",
    "ManuscriptPlacementAudit",
    "PlacedObject",
    "PlacementMatchMethod",
    "PlacementMatchStatus",
    "match_manuscript_figures",
]
