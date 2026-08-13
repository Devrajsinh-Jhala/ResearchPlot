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
_MAX_PDF_NAME_TREE_ENTRIES = 10_000
_MAX_RASTER_PIXELS = 100_000_000
_MAX_RASTER_DIMENSION = 100_000
_MAX_RASTER_FRAMES = 1_024
_MAX_RASTER_TOTAL_FRAME_PIXELS = 200_000_000
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
_CSS_ACTIVE = re.compile(r"(?:@import\b|expression\s*\(|javascript\s*:)", re.IGNORECASE)
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


def _pdf_color_space_name(value: Any) -> str:
    """Return a bounded, human-readable PDF colour-space description."""

    resolved = _resolve_pdf(value)
    if isinstance(resolved, (str, bytes)):
        return str(resolved)
    try:
        if isinstance(resolved, Iterable):
            items = list(resolved)[:4]
            return "[" + ", ".join(str(_resolve_pdf(item)) for item in items) + "]"
    except (TypeError, ValueError):
        pass
    return str(resolved)


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
    seen_xobjects: set[tuple[str, int, int]] = set()
    image_xobject_count = 0
    form_xobject_count = 0
    other_xobject_count = 0
    image_color_spaces: set[str] = set()
    image_xobject_details: list[
        tuple[str, float | None, float | None, float | None, str | None]
    ] = []
    resource_color_spaces: set[str] = set()
    transparency_reasons: set[str] = set()
    active_content_types: set[str] = set()
    external_uris: set[str] = set()
    embedded_file_count = 0
    action_objects_seen: set[tuple[str, int, int]] = set()

    def count_name_tree(node_ref: Any, depth: int = 0) -> int:
        if depth > _MAX_PDF_RECURSION:
            raise ArtifactParseError(
                f"Refusing to inspect PDF {path}: name-tree nesting exceeds "
                f"{_MAX_PDF_RECURSION} levels."
            )
        node = _resolve_pdf(node_ref)
        if not hasattr(node, "get"):
            return 0
        names_ref = node.get("/Names")
        count = 0
        if names_ref is not None:
            names = _resolve_pdf(names_ref)
            try:
                count += len(names) // 2
            except TypeError as exc:
                raise ArtifactParseError(f"PDF name tree is malformed: {exc}") from exc
            if count > _MAX_PDF_NAME_TREE_ENTRIES:
                raise ArtifactParseError(
                    f"Refusing to inspect PDF {path}: a name tree exceeds "
                    f"{_MAX_PDF_NAME_TREE_ENTRIES} entries."
                )
        kids_ref = node.get("/Kids")
        if kids_ref is not None:
            kids = _resolve_pdf(kids_ref)
            try:
                for kid in kids:
                    count += count_name_tree(kid, depth + 1)
                    if count > _MAX_PDF_NAME_TREE_ENTRIES:
                        raise ArtifactParseError(
                            f"Refusing to inspect PDF {path}: a name tree exceeds "
                            f"{_MAX_PDF_NAME_TREE_ENTRIES} entries."
                        )
            except ArtifactInspectionError:
                raise
            except TypeError as exc:
                raise ArtifactParseError(f"PDF name tree is malformed: {exc}") from exc
        return count

    def record_actions(value_ref: Any, depth: int = 0) -> None:
        """Record action dictionaries without decoding or executing their payloads."""

        if value_ref is None:
            return
        if depth > _MAX_PDF_RECURSION:
            raise ArtifactParseError(
                f"Refusing to inspect PDF {path}: action nesting exceeds "
                f"{_MAX_PDF_RECURSION} levels."
            )
        value = _resolve_pdf(value_ref)
        if isinstance(value, (list, tuple)):
            for item in value:
                record_actions(item, depth + 1)
            return
        if not hasattr(value, "get"):
            return
        key = _object_key(value_ref, value)
        if key in action_objects_seen:
            return
        if len(action_objects_seen) >= _MAX_PDF_RESOURCE_OBJECTS:
            raise ArtifactParseError(
                f"Refusing to inspect PDF {path}: more than {_MAX_PDF_RESOURCE_OBJECTS} "
                "action dictionaries were discovered."
            )
        action_objects_seen.add(key)
        action_type = str(value.get("/S", "")).lstrip("/")
        if action_type:
            active_content_types.add(action_type)
        uri = value.get("/URI")
        if uri is not None and len(external_uris) < _MAX_PDF_NAME_TREE_ENTRIES:
            external_uris.add(str(_resolve_pdf(uri))[:2_048])
        for nested_key in ("/Next", "/A", "/AA", "/OpenAction"):
            nested = value.get(nested_key)
            if nested is not None:
                record_actions(nested, depth + 1)
        if not action_type:
            for nested in value.values():
                record_actions(nested, depth + 1)

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

    def inspect_container(
        container_ref: Any, location: str, depth: int, container_name: str
    ) -> None:
        nonlocal image_xobject_count, form_xobject_count, other_xobject_count
        container = _resolve_pdf(container_ref)
        try:
            for name, child_ref in container.items():
                child = _resolve_pdf(child_ref)
                if container_name == "/XObject":
                    child_key = _object_key(child_ref, child)
                    if child_key not in seen_xobjects:
                        seen_xobjects.add(child_key)
                        subtype = str(child.get("/Subtype", "")) if hasattr(child, "get") else ""
                        if subtype == "/Image":
                            image_xobject_count += 1
                            color_space = child.get("/ColorSpace")
                            color_space_name = (
                                _pdf_color_space_name(color_space)
                                if color_space is not None
                                else None
                            )
                            if color_space is not None:
                                image_color_spaces.add(color_space_name or "unknown")
                            image_xobject_details.append(
                                (
                                    f"{location}/{name!s}",
                                    _finite_float(child.get("/Width")),
                                    _finite_float(child.get("/Height")),
                                    _finite_float(child.get("/BitsPerComponent")),
                                    color_space_name,
                                )
                            )
                            if child.get("/SMask") is not None:
                                transparency_reasons.add("image-soft-mask")
                            if child.get("/Mask") is not None:
                                transparency_reasons.add("image-mask")
                        elif subtype == "/Form":
                            form_xobject_count += 1
                            group = _resolve_pdf(child.get("/Group"))
                            if (
                                hasattr(group, "get")
                                and str(group.get("/S", "")) == "/Transparency"
                            ):
                                transparency_reasons.add("transparency-group")
                        else:
                            other_xobject_count += 1
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

        color_spaces_ref = resources.get("/ColorSpace")
        if color_spaces_ref is not None:
            color_spaces = _resolve_pdf(color_spaces_ref)
            try:
                for _, color_space_ref in color_spaces.items():
                    resource_color_spaces.add(_pdf_color_space_name(color_space_ref))
            except (AttributeError, TypeError) as exc:
                raise ArtifactParseError(
                    f"PDF colour-space dictionary at {location} is malformed: {exc}"
                ) from exc

        graphics_states_ref = resources.get("/ExtGState")
        if graphics_states_ref is not None:
            graphics_states = _resolve_pdf(graphics_states_ref)
            try:
                for _, state_ref in graphics_states.items():
                    state = _resolve_pdf(state_ref)
                    if not hasattr(state, "get"):
                        continue
                    for alpha_key, reason in (
                        ("/ca", "non-opaque-fill"),
                        ("/CA", "non-opaque-stroke"),
                    ):
                        alpha = _finite_float(state.get(alpha_key))
                        if alpha is not None and alpha < 1.0:
                            transparency_reasons.add(reason)
                    if state.get("/SMask") not in (None, "/None"):
                        transparency_reasons.add("graphics-state-soft-mask")
                    blend_mode = state.get("/BM")
                    if blend_mode not in (None, "/Normal", "/Compatible"):
                        transparency_reasons.add("non-normal-blend-mode")
            except (AttributeError, TypeError) as exc:
                raise ArtifactParseError(
                    f"PDF graphics-state dictionary at {location} is malformed: {exc}"
                ) from exc

        for container_name in ("/XObject", "/Pattern"):
            container_ref = resources.get(container_name)
            if container_ref is not None:
                inspect_container(
                    container_ref, f"{location}{container_name}", depth, container_name
                )

    def inspect_annotation_appearances(page: Any, page_number: int) -> None:
        annotations_ref = page.get("/Annots")
        if annotations_ref is None:
            return
        annotations = _resolve_pdf(annotations_ref)
        try:
            for annotation_index, annotation_ref in enumerate(annotations, start=1):
                annotation = _resolve_pdf(annotation_ref)
                subtype = str(annotation.get("/Subtype", "")).lstrip("/")
                if subtype in {"RichMedia", "Movie", "Sound", "FileAttachment", "Screen", "3D"}:
                    active_content_types.add(subtype)
                record_actions(annotation.get("/A"))
                record_actions(annotation.get("/AA"))
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
        catalog = _resolve_pdf(reader.trailer.get("/Root"))
        if hasattr(catalog, "get"):
            record_actions(catalog.get("/OpenAction"))
            record_actions(catalog.get("/AA"))
            names = _resolve_pdf(catalog.get("/Names"))
            if hasattr(names, "get"):
                javascript_tree = names.get("/JavaScript")
                if javascript_tree is not None and count_name_tree(javascript_tree) > 0:
                    active_content_types.add("JavaScript")
                embedded_tree = names.get("/EmbeddedFiles")
                if embedded_tree is not None:
                    embedded_file_count = count_name_tree(embedded_tree)
                    if embedded_file_count:
                        active_content_types.add("EmbeddedFile")
            form = _resolve_pdf(catalog.get("/AcroForm"))
            if hasattr(form, "get") and form.get("/XFA") is not None:
                active_content_types.add("XFA")
            if catalog.get("/Collection") is not None:
                active_content_types.add("Collection")

        for page_index, page in enumerate(reader.pages, start=1):
            record_actions(page.get("/AA"))
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
            _observation("pdf.xobject_count", len(seen_xobjects), "objects"),
            _observation("pdf.image_xobject_count", image_xobject_count, "objects"),
            _observation("pdf.form_xobject_count", form_xobject_count, "objects"),
            _observation("pdf.other_xobject_count", other_xobject_count, "objects"),
            _observation("pdf.image_color_spaces", tuple(sorted(image_color_spaces))),
            _observation(
                "pdf.image_xobject_details",
                tuple(sorted(image_xobject_details, key=lambda item: item[0])),
            ),
            _observation("pdf.resource_color_spaces", tuple(sorted(resource_color_spaces))),
            _observation("pdf.has_transparency", bool(transparency_reasons)),
            _observation("pdf.transparency_reasons", tuple(sorted(transparency_reasons))),
            _observation("pdf.active_content", bool(active_content_types)),
            _observation("pdf.active_content_types", tuple(sorted(active_content_types))),
            _observation("pdf.embedded_file_count", embedded_file_count, "files"),
            _observation("pdf.external_uri_count", len(external_uris), "links"),
            _observation("pdf.external_uris", tuple(sorted(external_uris))),
        )
    )
    result_warnings: list[str] = []
    if active_content_types:
        result_warnings.append(
            "PDF contains passive indicators of active or embedded content: "
            + ", ".join(sorted(active_content_types))
            + ". The inspector did not execute or fetch that content."
        )
    if page_count != 1:
        result_warnings.append(
            f"PDF contains {page_count} pages; scalar figure dimensions are unavailable. "
            "Submit one figure per PDF artifact."
        )
    return tuple(result_warnings)


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
    active_content_types: set[str] = set()
    script_element_count = 0
    foreign_object_count = 0
    event_handler_count = 0
    javascript_link_count = 0
    animation_element_count = 0
    if b"<?xml-stylesheet" in lowered:
        active_content_types.add("xml-stylesheet")
    for element in root.iter():
        tag_name = _local_name(element.tag).casefold()
        if tag_name == "text":
            text_count += 1
        if tag_name == "script":
            script_element_count += 1
            active_content_types.add("script")
        elif tag_name == "foreignobject":
            foreign_object_count += 1
            active_content_types.add("foreignObject")
        elif tag_name in {"iframe", "audio", "video", "object", "embed"}:
            active_content_types.add(tag_name)
        if tag_name in {"animate", "set", "animatetransform", "animatemotion", "animatecolor"}:
            animation_element_count += 1
            active_content_types.add("animation")
        for raw_name, raw_value in element.attrib.items():
            name = _local_name(raw_name).casefold()
            value_lower = raw_value.strip().casefold()
            if name.startswith("on"):
                event_handler_count += 1
                active_content_types.add("event-handler")
            if name == "font-family":
                font_declarations.add(raw_value.strip())
            if name in {"href", "src"}:
                if value_lower.startswith(("javascript:", "vbscript:")):
                    javascript_link_count += 1
                    active_content_types.add("script-link")
                reference = _external_reference(raw_value)
                if reference is not None:
                    external_links.add(reference)
            if name == "style":
                font_declarations.update(
                    match.group(1).strip() for match in _CSS_FONT_FAMILY.finditer(raw_value)
                )
                if _CSS_ACTIVE.search(raw_value):
                    active_content_types.add("active-css")
            for match in _CSS_URL.finditer(raw_value):
                reference = _external_reference(match.group(2))
                if reference is not None:
                    external_links.add(reference)
        if tag_name == "style" and element.text:
            font_declarations.update(
                match.group(1).strip() for match in _CSS_FONT_FAMILY.finditer(element.text)
            )
            if _CSS_ACTIVE.search(element.text):
                active_content_types.add("active-css")
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
            _observation("svg.script_element_count", script_element_count, "elements"),
            _observation("svg.foreign_object_count", foreign_object_count, "elements"),
            _observation("svg.event_handler_count", event_handler_count, "attributes"),
            _observation("svg.javascript_link_count", javascript_link_count, "links"),
            _observation("svg.animation_element_count", animation_element_count, "elements"),
            _observation("svg.active_content", bool(active_content_types)),
            _observation("svg.active_content_types", tuple(sorted(active_content_types))),
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
    if active_content_types:
        result_warnings.append(
            "SVG contains passive indicators of active content: "
            + ", ".join(sorted(active_content_types))
            + ". The inspector did not execute or fetch that content."
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
    result_warnings: list[str] = []
    has_physical_dimensions = False
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
                frame_count = int(getattr(image, "n_frames", 1))
                if frame_count <= 0 or frame_count > _MAX_RASTER_FRAMES:
                    raise ArtifactParseError(
                        f"Refusing to inspect raster image {path}: {frame_count} frames exceeds "
                        f"the safety limit of {_MAX_RASTER_FRAMES}."
                    )
                frame_sizes: list[tuple[int, int]] = []
                frame_modes: list[str] = []
                total_frame_pixels = 0
                has_alpha = False
                for frame_index in range(frame_count):
                    image.seek(frame_index)
                    frame_width = int(image.width)
                    frame_height = int(image.height)
                    if frame_width <= 0 or frame_height <= 0:
                        raise ArtifactParseError(
                            f"Raster image {path} frame {frame_index + 1} has non-positive "
                            "dimensions."
                        )
                    frame_pixels = frame_width * frame_height
                    if (
                        frame_width > _MAX_RASTER_DIMENSION
                        or frame_height > _MAX_RASTER_DIMENSION
                        or frame_pixels > _MAX_RASTER_PIXELS
                    ):
                        raise ArtifactParseError(
                            f"Refusing to inspect raster image {path}: frame {frame_index + 1} "
                            f"is {frame_width}x{frame_height}, exceeding the raster safety limit."
                        )
                    total_frame_pixels += frame_pixels
                    if total_frame_pixels > _MAX_RASTER_TOTAL_FRAME_PIXELS:
                        raise ArtifactParseError(
                            f"Refusing to inspect raster image {path}: decoded frame dimensions "
                            f"represent more than {_MAX_RASTER_TOTAL_FRAME_PIXELS} total pixels."
                        )
                    frame_sizes.append((frame_width, frame_height))
                    frame_modes.append(str(image.mode))
                    try:
                        has_alpha = (
                            has_alpha or "A" in image.getbands() or "transparency" in image.info
                        )
                    except (TypeError, ValueError):
                        has_alpha = has_alpha or "A" in str(image.mode)
                image.seek(0)

                orientation = 1
                try:
                    raw_orientation = image.getexif().get(274, 1)
                    parsed_orientation = int(raw_orientation)
                    if 1 <= parsed_orientation <= 8:
                        orientation = parsed_orientation
                except (AttributeError, TypeError, ValueError):
                    orientation = 1
                display_width, display_height = width, height
                if orientation in {5, 6, 7, 8}:
                    display_width, display_height = height, width

                dpi = _coerce_dpi_pair(image.info.get("dpi"))
                if dpi is None and file_format == "tiff":
                    dpi = _tiff_dpi(image)
                bit_depth = _raster_bit_depth(path, image, file_format)
                icc_profile = image.info.get("icc_profile")
                icc_size = len(icc_profile) if isinstance(icc_profile, bytes) else 0
                compression = _raster_compression(image, file_format)
                observations.extend(
                    (
                        _observation("artifact.page_count", frame_count, "frames"),
                        _observation("artifact.is_single_frame", frame_count == 1),
                        _observation("raster.pixel_width", width, "pixels"),
                        _observation("raster.pixel_height", height, "pixels"),
                        _observation("raster.pixel_count", width * height, "pixels"),
                        _observation("raster.display_pixel_width", display_width, "pixels"),
                        _observation("raster.display_pixel_height", display_height, "pixels"),
                        _observation("raster.frame_count", frame_count, "frames"),
                        _observation("raster.frame_sizes", tuple(frame_sizes)),
                        _observation("raster.frame_modes", tuple(frame_modes)),
                        _observation(
                            "raster.frames_uniform",
                            len(set(frame_sizes)) == 1 and len(set(frame_modes)) == 1,
                        ),
                        _observation("raster.total_frame_pixels", total_frame_pixels, "pixels"),
                        _observation("raster.mode", str(image.mode)),
                        _observation("raster.has_alpha", has_alpha),
                        _observation("raster.exif_orientation", orientation),
                        _observation("raster.requires_orientation_transform", orientation != 1),
                        _observation("raster.bit_depth", bit_depth, "bits/channel"),
                        _observation("raster.has_icc_profile", icc_size > 0),
                        _observation("raster.icc_profile_size", icc_size, "bytes"),
                        _observation("raster.compression", compression),
                    )
                )
                if dpi is not None and frame_count == 1:
                    dpi_x, dpi_y = dpi
                    if orientation in {5, 6, 7, 8}:
                        dpi_x, dpi_y = dpi_y, dpi_x
                    observations.extend(
                        (
                            _observation("raster.dpi_x", dpi_x, "dpi"),
                            _observation("raster.dpi_y", dpi_y, "dpi"),
                            _observation("artifact.width_mm", display_width / dpi_x * 25.4, "mm"),
                            _observation("artifact.height_mm", display_height / dpi_y * 25.4, "mm"),
                        )
                    )
                    has_physical_dimensions = True
                elif dpi is not None:
                    observations.extend(
                        (
                            _observation("raster.dpi_x", dpi[0], "dpi"),
                            _observation("raster.dpi_y", dpi[1], "dpi"),
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

    if frame_count != 1:
        result_warnings.append(
            f"Raster contains {frame_count} frames; scalar physical dimensions are unavailable. "
            "Export one static figure per artifact."
        )
    if not has_physical_dimensions and frame_count == 1:
        result_warnings.append(
            "Raster physical dimensions could not be established because valid DPI metadata "
            "is absent."
        )
    if orientation != 1:
        result_warnings.append(
            f"Raster uses EXIF orientation {orientation}; normalize pixel orientation before "
            "submission when downstream software may ignore EXIF metadata."
        )
    return tuple(result_warnings)


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
