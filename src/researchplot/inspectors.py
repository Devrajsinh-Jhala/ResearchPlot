"""Defensive, rule-independent inspection of publication figure artifacts.

The inspectors in this module only measure facts.  They deliberately know
nothing about venue profiles or compliance policy, which makes their output
safe to cache and straightforward for a rule engine to consume.
"""

from __future__ import annotations

import math
import re
import warnings
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError
from pypdf import PdfReader
from pypdf.errors import PyPdfError

_MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
_MAX_XML_BYTES = 16 * 1024 * 1024
_MAX_XML_ELEMENTS = 100_000
_MAX_XML_DEPTH = 256
_MAX_PDF_PAGES = 10_000
_MAX_PDF_RESOURCE_OBJECTS = 50_000
_MAX_PDF_FONTS = 10_000
_MAX_PDF_RECURSION = 64
_MAX_RASTER_PIXELS = 100_000_000
_MAX_RASTER_DIMENSION = 100_000
_EPS_SCAN_BYTES = 1024 * 1024

_MIME_TYPES = {
    "pdf": "application/pdf",
    "svg": "image/svg+xml",
    "png": "image/png",
    "jpeg": "image/jpeg",
    "tiff": "image/tiff",
    "eps": "application/postscript",
}
_EXTENSION_FORMATS = {
    ".pdf": "pdf",
    ".svg": "svg",
    ".png": "png",
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".tif": "tiff",
    ".tiff": "tiff",
    ".eps": "eps",
}
_SVG_ROOT = re.compile(rb"<(?:[A-Za-z_][\w.-]*:)?svg(?:\s|>)", re.IGNORECASE)
_MATPLOTLIB_SVG_DOCTYPE = re.compile(
    rb"<!DOCTYPE\s+svg\s+PUBLIC\s+['\"]-//W3C//DTD\s+SVG\s+1\.1//EN['\"]\s+"
    rb"['\"]https?://www\.w3\.org/Graphics/SVG/1\.1/DTD/svg11\.dtd['\"]\s*>",
    re.IGNORECASE,
)
_SVG_LENGTH = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*"
    r"(mm|cm|in|pt|pc|px|q)?\s*$",
    re.IGNORECASE,
)
_CSS_FONT_FAMILY = re.compile(r"(?:^|[;{])\s*font-family\s*:\s*([^;}]+)", re.IGNORECASE)
_CSS_URL = re.compile(r"url\(\s*(['\"]?)(.*?)\1\s*\)", re.IGNORECASE)
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_EPS_BOX = re.compile(
    rf"^%%(?P<kind>HiResBoundingBox|BoundingBox):\s*"
    rf"(?P<x0>{_NUMBER})\s+(?P<y0>{_NUMBER})\s+"
    rf"(?P<x1>{_NUMBER})\s+(?P<y1>{_NUMBER})\s*$",
    re.MULTILINE,
)


class ArtifactInspectionError(ValueError):
    """Base class for actionable artifact inspection failures."""


class UnsupportedArtifactError(ArtifactInspectionError):
    """Raised when a file is not one of the supported artifact formats."""


class ArtifactParseError(ArtifactInspectionError):
    """Raised when a supported artifact is malformed or unsafe to inspect."""


@dataclass(frozen=True, slots=True)
class Observation:
    """One normalized fact measured from an artifact.

    ``key`` is a stable, dotted probe identifier.  Values are scalars or
    immutable tuples; ``unit`` is supplied only for numeric measurements.
    """

    key: str
    value: object
    unit: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""

        result: dict[str, object] = {"key": self.key, "value": _jsonable(self.value)}
        if self.unit is not None:
            result["unit"] = self.unit
        return result


@dataclass(frozen=True, slots=True)
class ArtifactInspection:
    """Immutable normalized metadata obtained from one artifact."""

    path: Path
    format: str
    mime_type: str
    observations: tuple[Observation, ...]
    warnings: tuple[str, ...] = ()

    def get(self, key: str, default: object = None) -> object:
        """Return an observed value by probe key."""

        for observation in self.observations:
            if observation.key == key:
                return observation.value
        return default

    @property
    def metadata(self) -> dict[str, object]:
        """Return a fresh key-to-value mapping of all observations."""

        return {observation.key: observation.value for observation in self.observations}

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""

        return {
            "path": str(self.path),
            "format": self.format,
            "mime_type": self.mime_type,
            "observations": [observation.to_dict() for observation in self.observations],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class _FontFinding:
    name: str
    subtype: str
    embedded: bool
    unembedded_truetype: bool
    location: str


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def _observation(key: str, value: object, unit: str | None = None) -> Observation:
    return Observation(key=key, value=value, unit=unit)


def _object_key(reference: Any, resolved: Any) -> tuple[str, int, int]:
    id_number = getattr(reference, "idnum", None)
    generation = getattr(reference, "generation", 0)
    if isinstance(id_number, int):
        return ("indirect", id_number, int(generation))
    return ("direct", id(resolved), 0)


def _resolve_pdf(value: Any) -> Any:
    getter = getattr(value, "get_object", None)
    return getter() if callable(getter) else value


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return number if math.isfinite(number) else None


def _detect_format(header: bytes) -> str:
    if header.startswith(b"%PDF-"):
        return "pdf"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if header.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if header.startswith((b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")):
        return "tiff"

    stripped = header.lstrip(b"\xef\xbb\xbf\x00\t\r\n ")
    if stripped.startswith(b"%!PS-Adobe") and b"EPSF" in stripped[:256]:
        return "eps"
    if _SVG_ROOT.search(stripped[:16_384]):
        return "svg"
    raise UnsupportedArtifactError(
        "Unsupported figure artifact. Expected PDF, SVG, PNG, JPEG, TIFF, or EPS "
        "content; the file signature did not match any supported format."
    )


def _base_observations(path: Path, file_format: str, size: int) -> list[Observation]:
    observations = [
        _observation("artifact.format", file_format),
        _observation("artifact.file_size", size, "bytes"),
    ]
    extension_format = _EXTENSION_FORMATS.get(path.suffix.casefold())
    if path.suffix:
        observations.append(_observation("artifact.extension", path.suffix.casefold()))
    if extension_format is not None:
        observations.append(
            _observation("artifact.extension_matches_content", extension_format == file_format)
        )
    return observations


def _pdf_box(value: Any) -> tuple[float, float, float, float] | None:
    resolved = _resolve_pdf(value)
    try:
        numbers = (
            float(resolved[0]),
            float(resolved[1]),
            float(resolved[2]),
            float(resolved[3]),
        )
    except (IndexError, KeyError, TypeError, ValueError):
        return None
    if len(numbers) != 4 or not all(math.isfinite(item) for item in numbers):
        return None
    return numbers


def _font_descriptors(font: Any) -> tuple[Any, ...]:
    descriptor = font.get("/FontDescriptor")
    if descriptor is not None:
        return (_resolve_pdf(descriptor),)
    descendants_ref = font.get("/DescendantFonts")
    if descendants_ref is None:
        return ()
    descendants = _resolve_pdf(descendants_ref)
    results: list[Any] = []
    try:
        for descendant_ref in descendants:
            descendant = _resolve_pdf(descendant_ref)
            candidate = descendant.get("/FontDescriptor")
            if candidate is not None:
                results.append(_resolve_pdf(candidate))
    except (AttributeError, TypeError):
        return ()
    return tuple(results)


def _font_is_embedded(font: Any, subtype: str) -> bool:
    # Type 3 glyph programs are contained in the PDF itself.  They are still
    # reported independently because many publishers prohibit Type 3 fonts.
    if subtype == "/Type3":
        return True
    descriptors = _font_descriptors(font)
    if not descriptors:
        return False
    return all(
        any(key in descriptor for key in ("/FontFile", "/FontFile2", "/FontFile3"))
        for descriptor in descriptors
    )


def _font_has_unembedded_truetype(font: Any, subtype: str) -> bool:
    if subtype in {"/TrueType", "/CIDFontType2"}:
        return not _font_is_embedded(font, subtype)
    if subtype != "/Type0":
        return False
    descendants_ref = font.get("/DescendantFonts")
    if descendants_ref is None:
        return False
    try:
        descendants = _resolve_pdf(descendants_ref)
        return any(
            str((descendant := _resolve_pdf(reference)).get("/Subtype", "unknown"))
            == "/CIDFontType2"
            and not _font_is_embedded(descendant, "/CIDFontType2")
            for reference in descendants
        )
    except (AttributeError, TypeError):
        return False


def _inspect_pdf(path: Path, observations: list[Observation]) -> tuple[str, ...]:
    try:
        reader = PdfReader(path, strict=False)
        if reader.is_encrypted and reader.decrypt("") == 0:
            raise ArtifactParseError(
                f"Cannot inspect encrypted PDF {path}; remove the password or provide an "
                "unencrypted submission artifact."
            )
        page_count = len(reader.pages)
    except ArtifactInspectionError:
        raise
    except (OSError, PyPdfError, ValueError) as exc:
        raise ArtifactParseError(f"Could not parse PDF {path}: {exc}") from exc

    if page_count == 0:
        raise ArtifactParseError(f"Could not inspect PDF {path}: it contains no pages.")
    if page_count > _MAX_PDF_PAGES:
        raise ArtifactParseError(
            f"Refusing to inspect PDF {path}: {page_count} pages exceeds the safety limit "
            f"of {_MAX_PDF_PAGES}."
        )

    observations.append(_observation("artifact.page_count", page_count, "pages"))
    observations.append(_observation("pdf.page_count", page_count, "pages"))
    page_widths: list[float] = []
    page_heights: list[float] = []
    fonts: dict[tuple[str, int, int], _FontFinding] = {}
    font_resource_occurrences = 0
    seen_resources: set[tuple[str, int, int]] = set()

    def record_font(font_ref: Any, resource_name: object, location: str, depth: int) -> None:
        nonlocal font_resource_occurrences
        font_resource_occurrences += 1
        try:
            font = _resolve_pdf(font_ref)
            key = _object_key(font_ref, font)
            if key in fonts:
                return
            if len(fonts) >= _MAX_PDF_FONTS:
                raise ArtifactParseError(
                    f"Refusing to inspect PDF {path}: more than {_MAX_PDF_FONTS} unique "
                    "font resources were discovered."
                )
            subtype = str(font.get("/Subtype", "unknown"))
            name = str(font.get("/BaseFont") or font.get("/Name") or resource_name)
            fonts[key] = _FontFinding(
                name=name,
                subtype=subtype,
                embedded=_font_is_embedded(font, subtype),
                unembedded_truetype=_font_has_unembedded_truetype(font, subtype),
                location=location,
            )
            nested = font.get("/Resources")
            if nested is not None:
                inspect_resources(
                    nested,
                    f"{location}/font:{resource_name}",
                    depth + 1,
                )
        except ArtifactInspectionError:
            raise
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ArtifactParseError(
                f"Could not inspect PDF font resource {resource_name!s} at {location}: {exc}"
            ) from exc

    def inspect_container(container_ref: Any, location: str, depth: int) -> None:
        container = _resolve_pdf(container_ref)
        try:
            for name, child_ref in container.items():
                child = _resolve_pdf(child_ref)
                nested = child.get("/Resources") if hasattr(child, "get") else None
                if nested is not None:
                    inspect_resources(nested, f"{location}/{name!s}", depth + 1)
        except ArtifactInspectionError:
            raise
        except (AttributeError, TypeError, ValueError) as exc:
            raise ArtifactParseError(
                f"Could not inspect nested PDF resources at {location}: {exc}"
            ) from exc

    def inspect_resources(resources_ref: Any, location: str, depth: int) -> None:
        if depth > _MAX_PDF_RECURSION:
            raise ArtifactParseError(
                f"Refusing to inspect PDF {path}: resource nesting exceeds "
                f"{_MAX_PDF_RECURSION} levels."
            )
        resources = _resolve_pdf(resources_ref)
        key = _object_key(resources_ref, resources)
        if key in seen_resources:
            return
        if len(seen_resources) >= _MAX_PDF_RESOURCE_OBJECTS:
            raise ArtifactParseError(
                f"Refusing to inspect PDF {path}: more than {_MAX_PDF_RESOURCE_OBJECTS} "
                "resource dictionaries were discovered."
            )
        seen_resources.add(key)
        if not hasattr(resources, "get"):
            return

        fonts_ref = resources.get("/Font")
        if fonts_ref is not None:
            font_dictionary = _resolve_pdf(fonts_ref)
            try:
                for name, font_ref in font_dictionary.items():
                    record_font(font_ref, name, location, depth)
            except ArtifactInspectionError:
                raise
            except (AttributeError, TypeError) as exc:
                raise ArtifactParseError(
                    f"PDF font dictionary at {location} is malformed: {exc}"
                ) from exc

        for container_name in ("/XObject", "/Pattern"):
            container_ref = resources.get(container_name)
            if container_ref is not None:
                inspect_container(container_ref, f"{location}{container_name}", depth)

    def inspect_annotation_appearances(page: Any, page_number: int) -> None:
        annotations_ref = page.get("/Annots")
        if annotations_ref is None:
            return
        annotations = _resolve_pdf(annotations_ref)
        try:
            for annotation_index, annotation_ref in enumerate(annotations, start=1):
                annotation = _resolve_pdf(annotation_ref)
                appearances_ref = annotation.get("/AP")
                if appearances_ref is None:
                    continue
                appearances = _resolve_pdf(appearances_ref)
                for appearance_name, appearance_ref in appearances.items():
                    appearance = _resolve_pdf(appearance_ref)
                    # /N may itself be a state dictionary whose values are streams.
                    candidates: Iterable[Any]
                    if hasattr(appearance, "get") and appearance.get("/Resources") is None:
                        candidates = appearance.values()
                    else:
                        candidates = (appearance,)
                    for candidate in candidates:
                        resolved = _resolve_pdf(candidate)
                        nested = resolved.get("/Resources") if hasattr(resolved, "get") else None
                        if nested is not None:
                            inspect_resources(
                                nested,
                                f"page:{page_number}/annotation:{annotation_index}/{appearance_name}",
                                1,
                            )
        except ArtifactInspectionError:
            raise
        except (AttributeError, TypeError, ValueError) as exc:
            raise ArtifactParseError(
                f"Could not inspect annotation resources on PDF page {page_number}: {exc}"
            ) from exc

    try:
        for page_index, page in enumerate(reader.pages, start=1):
            rotation_value = _finite_float(page.get("/Rotate", 0)) or 0.0
            rotation = int(rotation_value) % 360
            observations.append(
                _observation(f"pdf.page.{page_index}.rotation", rotation, "degrees")
            )

            media_box = _pdf_box(page.get("/MediaBox")) or _pdf_box(page.mediabox)
            if media_box is None:
                raise ArtifactParseError(f"PDF page {page_index} has no valid MediaBox.")
            box_values: dict[str, tuple[float, float, float, float]] = {"media_box": media_box}
            for pdf_name, key_name in (
                ("/CropBox", "crop_box"),
                ("/BleedBox", "bleed_box"),
                ("/TrimBox", "trim_box"),
                ("/ArtBox", "art_box"),
            ):
                raw_box = page.get(pdf_name)
                if raw_box is not None:
                    parsed = _pdf_box(raw_box)
                    if parsed is None:
                        raise ArtifactParseError(
                            f"PDF page {page_index} has a malformed {pdf_name}."
                        )
                    box_values[key_name] = parsed
            for box_name, box in box_values.items():
                observations.append(
                    _observation(f"pdf.page.{page_index}.{box_name}", box, "points")
                )

            effective_box_name = "crop_box" if "crop_box" in box_values else "media_box"
            effective_box = box_values[effective_box_name]
            observations.extend(
                (
                    _observation(f"pdf.page.{page_index}.effective_box", effective_box, "points"),
                    _observation(f"pdf.page.{page_index}.effective_box_type", effective_box_name),
                )
            )

            width_points = effective_box[2] - effective_box[0]
            height_points = effective_box[3] - effective_box[1]
            if width_points <= 0 or height_points <= 0:
                raise ArtifactParseError(
                    f"PDF page {page_index} has a non-positive effective visible box."
                )
            width_mm = width_points * 25.4 / 72.0
            height_mm = height_points * 25.4 / 72.0
            if rotation in (90, 270):
                width_mm, height_mm = height_mm, width_mm
            page_widths.append(width_mm)
            page_heights.append(height_mm)
            observations.extend(
                (
                    _observation(f"pdf.page.{page_index}.width_mm", width_mm, "mm"),
                    _observation(f"pdf.page.{page_index}.height_mm", height_mm, "mm"),
                )
            )

            resources = page.get("/Resources")
            if resources is not None:
                inspect_resources(resources, f"page:{page_index}", 1)
            inspect_annotation_appearances(page, page_index)
    except ArtifactInspectionError:
        raise
    except (OSError, PyPdfError, KeyError, TypeError, ValueError) as exc:
        raise ArtifactParseError(f"Could not inspect PDF {path}: {exc}") from exc

    observations.extend(
        (
            _observation("artifact.is_single_page", page_count == 1),
            _observation("pdf.page_widths_mm", tuple(page_widths), "mm"),
            _observation("pdf.page_heights_mm", tuple(page_heights), "mm"),
        )
    )
    if page_count == 1:
        observations.extend(
            (
                _observation("artifact.width_mm", page_widths[0], "mm"),
                _observation("artifact.height_mm", page_heights[0], "mm"),
            )
        )

    sorted_fonts = sorted(fonts.values(), key=lambda item: (item.name, item.subtype, item.location))
    type3_count = sum(item.subtype == "/Type3" for item in sorted_fonts)
    embedded_count = sum(item.embedded for item in sorted_fonts)
    unembedded_truetype_count = sum(item.unembedded_truetype for item in sorted_fonts)
    observations.extend(
        (
            _observation("pdf.font_count", len(sorted_fonts), "fonts"),
            _observation("pdf.font_resource_occurrences", font_resource_occurrences, "resources"),
            _observation("pdf.embedded_font_count", embedded_count, "fonts"),
            _observation("pdf.unembedded_font_count", len(sorted_fonts) - embedded_count, "fonts"),
            _observation(
                "pdf.unembedded_truetype_font_count",
                unembedded_truetype_count,
                "fonts",
            ),
            _observation("pdf.type3_font_count", type3_count, "fonts"),
            _observation("pdf.font_names", tuple(item.name for item in sorted_fonts)),
            _observation("pdf.font_subtypes", tuple(item.subtype for item in sorted_fonts)),
            _observation(
                "pdf.font_details",
                tuple(
                    (item.name, item.subtype, item.embedded, item.location) for item in sorted_fonts
                ),
            ),
        )
    )
    if page_count != 1:
        return (
            f"PDF contains {page_count} pages; scalar figure dimensions are unavailable. "
            "Submit one figure per PDF artifact.",
        )
    return ()


def _svg_length_mm(raw_value: str | None) -> float | None:
    if raw_value is None:
        return None
    match = _SVG_LENGTH.fullmatch(raw_value)
    if match is None:
        return None
    value = float(match.group(1))
    if not math.isfinite(value) or value < 0:
        return None
    unit = (match.group(2) or "px").casefold()
    factors = {
        "mm": 1.0,
        "cm": 10.0,
        "in": 25.4,
        "pt": 25.4 / 72.0,
        "pc": 25.4 / 6.0,
        "px": 25.4 / 96.0,
        "q": 0.25,
    }
    return value * factors[unit]


def _svg_view_box(raw_value: str | None) -> tuple[float, float, float, float] | None:
    if raw_value is None:
        return None
    try:
        values = tuple(float(item) for item in re.split(r"[\s,]+", raw_value.strip()) if item)
    except ValueError:
        return None
    if len(values) != 4 or not all(math.isfinite(item) for item in values):
        return None
    if values[2] <= 0 or values[3] <= 0:
        return None
    return values[0], values[1], values[2], values[3]


def _local_name(name: str) -> str:
    return name.rsplit("}", 1)[-1].rsplit(":", 1)[-1]


def _external_reference(value: str) -> str | None:
    candidate = value.strip().strip("'\"")
    if not candidate or candidate.startswith("#") or candidate.casefold().startswith("data:"):
        return None
    return candidate


def _inspect_svg(path: Path, size: int, observations: list[Observation]) -> tuple[str, ...]:
    if size > _MAX_XML_BYTES:
        raise ArtifactParseError(
            f"Refusing to inspect SVG {path}: {size} bytes exceeds the XML safety limit "
            f"of {_MAX_XML_BYTES} bytes."
        )
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ArtifactParseError(f"Could not read SVG {path}: {exc}") from exc
    lowered = payload.lower()
    if b"<!entity" in lowered:
        raise ArtifactParseError(
            f"Refusing to inspect SVG {path}: DTD and entity declarations are not allowed."
        )
    payload = _MATPLOTLIB_SVG_DOCTYPE.sub(b"", payload, count=1)
    if b"<!doctype" in payload.lower():
        raise ArtifactParseError(
            f"Refusing to inspect SVG {path}: only the standard SVG 1.1 public doctype "
            "emitted by Matplotlib is accepted."
        )

    root: ET.Element | None = None
    element_count = 0
    depth = 0
    try:
        parser = ET.iterparse(BytesIO(payload), events=("start", "end"))
        for event, element in parser:
            if event == "start":
                if root is None:
                    root = element
                element_count += 1
                depth += 1
                if element_count > _MAX_XML_ELEMENTS:
                    raise ArtifactParseError(
                        f"Refusing to inspect SVG {path}: more than {_MAX_XML_ELEMENTS} XML "
                        "elements were found."
                    )
                if depth > _MAX_XML_DEPTH:
                    raise ArtifactParseError(
                        f"Refusing to inspect SVG {path}: XML nesting exceeds "
                        f"{_MAX_XML_DEPTH} levels."
                    )
            else:
                depth -= 1
    except ArtifactInspectionError:
        raise
    except (ET.ParseError, OSError, ValueError) as exc:
        raise ArtifactParseError(f"Could not parse SVG {path}: {exc}") from exc

    if root is None or _local_name(root.tag).casefold() != "svg":
        raise ArtifactParseError(f"Could not parse SVG {path}: the root element is not <svg>.")

    raw_width = root.get("width")
    raw_height = root.get("height")
    raw_view_box = root.get("viewBox") or root.get("viewbox")
    view_box = _svg_view_box(raw_view_box)
    if raw_view_box is not None and view_box is None:
        raise ArtifactParseError(f"Could not parse SVG {path}: viewBox is malformed.")
    width_mm = _svg_length_mm(raw_width)
    height_mm = _svg_length_mm(raw_height)
    if view_box is not None:
        aspect = view_box[2] / view_box[3]
        if width_mm is not None and height_mm is None:
            height_mm = width_mm / aspect
        elif height_mm is not None and width_mm is None:
            width_mm = height_mm * aspect

    text_count = 0
    font_declarations: set[str] = set()
    external_links: set[str] = set()
    for element in root.iter():
        if _local_name(element.tag).casefold() == "text":
            text_count += 1
        for raw_name, raw_value in element.attrib.items():
            name = _local_name(raw_name).casefold()
            if name == "font-family":
                font_declarations.add(raw_value.strip())
            if name in {"href", "src"}:
                reference = _external_reference(raw_value)
                if reference is not None:
                    external_links.add(reference)
            if name == "style":
                font_declarations.update(
                    match.group(1).strip() for match in _CSS_FONT_FAMILY.finditer(raw_value)
                )
            for match in _CSS_URL.finditer(raw_value):
                reference = _external_reference(match.group(2))
                if reference is not None:
                    external_links.add(reference)
        if _local_name(element.tag).casefold() == "style" and element.text:
            font_declarations.update(
                match.group(1).strip() for match in _CSS_FONT_FAMILY.finditer(element.text)
            )
            for match in _CSS_URL.finditer(element.text):
                reference = _external_reference(match.group(2))
                if reference is not None:
                    external_links.add(reference)

    font_declarations.discard("")
    observations.extend(
        (
            _observation("artifact.page_count", 1, "pages"),
            _observation("svg.element_count", element_count, "elements"),
            _observation("svg.width", raw_width),
            _observation("svg.height", raw_height),
            _observation("svg.view_box", view_box),
            _observation("svg.text_element_count", text_count, "elements"),
            _observation("svg.has_editable_text", text_count > 0),
            _observation("svg.font_declarations", tuple(sorted(font_declarations))),
            _observation("svg.external_links", tuple(sorted(external_links))),
            _observation("svg.external_link_count", len(external_links), "links"),
        )
    )
    result_warnings: list[str] = []
    if width_mm is not None:
        observations.append(_observation("artifact.width_mm", width_mm, "mm"))
    if height_mm is not None:
        observations.append(_observation("artifact.height_mm", height_mm, "mm"))
    if width_mm is None or height_mm is None:
        result_warnings.append(
            "SVG physical dimensions could not be established from absolute width/height "
            "metadata; viewBox coordinates alone are not physical units."
        )
    return tuple(result_warnings)


def _coerce_dpi_pair(raw_value: object) -> tuple[float, float] | None:
    values: tuple[object, ...]
    if isinstance(raw_value, (tuple, list)):
        values = tuple(raw_value)
    elif raw_value is not None:
        values = (raw_value, raw_value)
    else:
        return None
    if not values:
        return None
    x = _finite_float(values[0])
    y = _finite_float(values[1] if len(values) > 1 else values[0])
    if x is None or y is None or x <= 0 or y <= 0:
        return None
    return x, y


def _tiff_dpi(image: Any) -> tuple[float, float] | None:
    tags = getattr(image, "tag_v2", None)
    if tags is None:
        return None
    x = _finite_float(tags.get(282))
    y = _finite_float(tags.get(283))
    if x is None or y is None or x <= 0 or y <= 0:
        return None
    unit = int(_finite_float(tags.get(296, 2)) or 2)
    if unit == 3:  # pixels per centimetre
        x *= 2.54
        y *= 2.54
    elif unit != 2:  # unitless resolution cannot establish a physical size
        return None
    return x, y


def _raster_bit_depth(path: Path, image: Any, file_format: str) -> int | tuple[int, ...] | None:
    if file_format == "png":
        try:
            with path.open("rb") as stream:
                header = stream.read(25)
            if len(header) == 25 and header[12:16] == b"IHDR":
                return int(header[24])
        except OSError:
            return None
    if file_format == "tiff":
        tags = getattr(image, "tag_v2", None)
        raw_bits = tags.get(258) if tags is not None else None
        if isinstance(raw_bits, (tuple, list)):
            bits = tuple(int(item) for item in raw_bits)
            return bits[0] if bits and len(set(bits)) == 1 else bits
        if raw_bits is not None:
            return int(raw_bits)
    image_bits = getattr(image, "bits", None)
    if isinstance(image_bits, int) and image_bits > 0:
        return image_bits
    mode_bits = {
        "1": 1,
        "L": 8,
        "LA": 8,
        "P": 8,
        "RGB": 8,
        "RGBA": 8,
        "CMYK": 8,
        "YCbCr": 8,
        "LAB": 8,
        "HSV": 8,
        "I;16": 16,
        "I;16L": 16,
        "I;16B": 16,
        "I": 32,
        "F": 32,
    }
    return mode_bits.get(str(image.mode))


def _raster_compression(image: Any, file_format: str) -> str | None:
    if file_format == "png":
        return "deflate"
    if file_format == "jpeg":
        return "jpeg-progressive" if image.info.get("progressive") else "jpeg-baseline"
    if file_format == "tiff":
        tags = getattr(image, "tag_v2", None)
        code_value = tags.get(259) if tags is not None else None
        code = int(_finite_float(code_value) or 0)
        names = {
            1: "none",
            2: "ccitt-rle",
            3: "ccitt-t4",
            4: "ccitt-t6",
            5: "lzw",
            7: "jpeg",
            8: "deflate",
            32773: "packbits",
            32946: "deflate",
        }
        return names.get(code, f"tiff-code-{code}" if code else None)
    return None


def _inspect_raster(
    path: Path, file_format: str, observations: list[Observation]
) -> tuple[str, ...]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as verify_image:
                actual_format = str(verify_image.format or "").casefold()
                verify_width = int(verify_image.width)
                verify_height = int(verify_image.height)
                if verify_width <= 0 or verify_height <= 0:
                    raise ArtifactParseError(f"Raster image {path} has non-positive dimensions.")
                if (
                    verify_width > _MAX_RASTER_DIMENSION
                    or verify_height > _MAX_RASTER_DIMENSION
                    or verify_width * verify_height > _MAX_RASTER_PIXELS
                ):
                    raise ArtifactParseError(
                        f"Refusing to inspect raster image {path}: {verify_width}x"
                        f"{verify_height} pixels exceeds the safety limit of "
                        f"{_MAX_RASTER_PIXELS} total pixels and "
                        f"{_MAX_RASTER_DIMENSION} pixels per dimension."
                    )
                verify_image.verify()
            with Image.open(path) as image:
                if actual_format in {"jpg", "jpeg"}:
                    actual_format = "jpeg"
                elif actual_format in {"tif", "tiff"}:
                    actual_format = "tiff"
                if actual_format != file_format:
                    raise ArtifactParseError(
                        f"Raster decoder identified {path} as {actual_format or 'unknown'}, "
                        f"not {file_format}."
                    )
                width = int(image.width)
                height = int(image.height)
                if width <= 0 or height <= 0:
                    raise ArtifactParseError(f"Raster image {path} has non-positive dimensions.")
                dpi = _coerce_dpi_pair(image.info.get("dpi"))
                if dpi is None and file_format == "tiff":
                    dpi = _tiff_dpi(image)
                bit_depth = _raster_bit_depth(path, image, file_format)
                icc_profile = image.info.get("icc_profile")
                icc_size = len(icc_profile) if isinstance(icc_profile, bytes) else 0
                frame_count = int(getattr(image, "n_frames", 1))
                compression = _raster_compression(image, file_format)
                observations.extend(
                    (
                        _observation("artifact.page_count", frame_count, "frames"),
                        _observation("raster.pixel_width", width, "pixels"),
                        _observation("raster.pixel_height", height, "pixels"),
                        _observation("raster.pixel_count", width * height, "pixels"),
                        _observation("raster.frame_count", frame_count, "frames"),
                        _observation("raster.mode", str(image.mode)),
                        _observation("raster.bit_depth", bit_depth, "bits/channel"),
                        _observation("raster.has_icc_profile", icc_size > 0),
                        _observation("raster.icc_profile_size", icc_size, "bytes"),
                        _observation("raster.compression", compression),
                    )
                )
                if dpi is not None:
                    observations.extend(
                        (
                            _observation("raster.dpi_x", dpi[0], "dpi"),
                            _observation("raster.dpi_y", dpi[1], "dpi"),
                            _observation("artifact.width_mm", width / dpi[0] * 25.4, "mm"),
                            _observation("artifact.height_mm", height / dpi[1] * 25.4, "mm"),
                        )
                    )
    except ArtifactInspectionError:
        raise
    except Image.DecompressionBombError as exc:
        raise ArtifactParseError(f"Refusing to inspect raster image {path}: {exc}") from exc
    except Image.DecompressionBombWarning as exc:
        raise ArtifactParseError(f"Refusing to inspect raster image {path}: {exc}") from exc
    except (OSError, UnidentifiedImageError, ValueError, SyntaxError) as exc:
        raise ArtifactParseError(f"Could not parse raster image {path}: {exc}") from exc

    if observations[-1].key not in {"artifact.width_mm", "artifact.height_mm"} and not any(
        item.key == "artifact.width_mm" for item in observations
    ):
        return (
            "Raster physical dimensions could not be established because valid DPI metadata "
            "is absent.",
        )
    return ()


def _read_eps_sections(path: Path, size: int) -> str:
    try:
        with path.open("rb") as stream:
            head = stream.read(_EPS_SCAN_BYTES)
            if size > _EPS_SCAN_BYTES:
                stream.seek(max(0, size - _EPS_SCAN_BYTES))
                tail = stream.read(_EPS_SCAN_BYTES)
            else:
                tail = b""
    except OSError as exc:
        raise ArtifactParseError(f"Could not read EPS {path}: {exc}") from exc
    return (head + (b"\n" if tail else b"") + tail).decode("latin-1", errors="replace")


def _inspect_eps(path: Path, size: int, observations: list[Observation]) -> tuple[str, ...]:
    text = _read_eps_sections(path, size)
    first_line = text.splitlines()[0] if text.splitlines() else ""
    if not first_line.startswith("%!PS-Adobe") or "EPSF" not in first_line:
        raise ArtifactParseError(
            f"Could not parse EPS {path}: the PostScript header does not declare EPSF."
        )

    boxes: dict[str, tuple[float, float, float, float]] = {}
    for match in _EPS_BOX.finditer(text):
        box = (
            float(match.group("x0")),
            float(match.group("y0")),
            float(match.group("x1")),
            float(match.group("y1")),
        )
        if not all(math.isfinite(item) for item in box) or box[2] <= box[0] or box[3] <= box[1]:
            raise ArtifactParseError(
                f"Could not parse EPS {path}: {match.group('kind')} is non-positive or invalid."
            )
        boxes[match.group("kind")] = box  # Later trailer values intentionally win.

    observations.extend(
        (
            _observation("artifact.page_count", 1, "pages"),
            _observation("eps.bounding_box", boxes.get("BoundingBox"), "points"),
            _observation("eps.hires_bounding_box", boxes.get("HiResBoundingBox"), "points"),
        )
    )
    preferred = boxes.get("HiResBoundingBox") or boxes.get("BoundingBox")
    if preferred is None:
        return (
            "EPS physical dimensions could not be established because neither BoundingBox nor "
            "HiResBoundingBox has a concrete value in the DSC header or trailer.",
        )
    observations.extend(
        (
            _observation("artifact.width_mm", (preferred[2] - preferred[0]) * 25.4 / 72.0, "mm"),
            _observation("artifact.height_mm", (preferred[3] - preferred[1]) * 25.4 / 72.0, "mm"),
        )
    )
    return ()


def inspect_artifact(path: str | Path) -> ArtifactInspection:
    """Inspect a PDF, SVG, PNG, JPEG, TIFF, or EPS artifact offline.

    Content signatures are used instead of filename extensions.  The function
    raises :class:`UnsupportedArtifactError` for unknown content and
    :class:`ArtifactParseError` with a corrective message for malformed,
    encrypted, oversized, or unsafe artifacts.
    """

    file_path = Path(path).expanduser().resolve()
    try:
        if not file_path.is_file():
            raise FileNotFoundError(f"Figure artifact not found or not a regular file: {file_path}")
        size = file_path.stat().st_size
        if size == 0:
            raise ArtifactParseError(f"Cannot inspect empty figure artifact: {file_path}")
        if size > _MAX_ARTIFACT_BYTES:
            raise ArtifactParseError(
                f"Refusing to inspect {file_path}: {size} bytes exceeds the safety limit of "
                f"{_MAX_ARTIFACT_BYTES} bytes."
            )
        with file_path.open("rb") as stream:
            header = stream.read(16_384)
    except ArtifactInspectionError:
        raise
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise ArtifactInspectionError(
            f"Could not access figure artifact {file_path}: {exc}"
        ) from exc

    file_format = _detect_format(header)
    observations = _base_observations(file_path, file_format, size)
    warnings_found: tuple[str, ...]
    if file_format == "pdf":
        warnings_found = _inspect_pdf(file_path, observations)
    elif file_format == "svg":
        warnings_found = _inspect_svg(file_path, size, observations)
    elif file_format in {"png", "jpeg", "tiff"}:
        warnings_found = _inspect_raster(file_path, file_format, observations)
    elif file_format == "eps":
        warnings_found = _inspect_eps(file_path, size, observations)
    else:  # pragma: no cover - kept as a defensive boundary for future detectors.
        raise UnsupportedArtifactError(f"No inspector is registered for {file_format!r}.")

    extension_format = _EXTENSION_FORMATS.get(file_path.suffix.casefold())
    warnings_list = list(warnings_found)
    if extension_format is not None and extension_format != file_format:
        warnings_list.append(
            f"Filename extension {file_path.suffix!r} suggests {extension_format}, but the "
            f"content is {file_format}."
        )
    return ArtifactInspection(
        path=file_path,
        format=file_format,
        mime_type=_MIME_TYPES[file_format],
        observations=tuple(observations),
        warnings=tuple(warnings_list),
    )


__all__ = [
    "ArtifactInspection",
    "ArtifactInspectionError",
    "ArtifactParseError",
    "Observation",
    "UnsupportedArtifactError",
    "inspect_artifact",
]
