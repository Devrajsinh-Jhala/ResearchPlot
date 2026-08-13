"""Passive, page-level audit of manuscript PDFs.

The manuscript auditor measures layout and embedded figure-resource facts.  It
does not render, execute actions, fetch links, or claim that a manuscript meets
a publisher's full submission policy.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from .inspectors import ArtifactParseError, _object_key, _resolve_pdf, inspect_artifact

if TYPE_CHECKING:
    from .manuscript_matching import ManuscriptPlacementAudit
    from .specs import FigureSpec, ManuscriptMatchHint

_MAX_MANUSCRIPT_PAGES = 2_000
_MAX_PAGE_RESOURCES = 20_000
_MAX_RESOURCE_DEPTH = 64
_MAX_ANNOTATIONS_PER_PAGE = 10_000


@dataclass(frozen=True, slots=True)
class ManuscriptPageAudit:
    """Measured resources and visible size for one manuscript page."""

    number: int
    width_mm: float
    height_mm: float
    rotation: int
    image_xobjects: int
    form_xobjects: int
    annotation_count: int
    has_transparency: bool
    active_content_types: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "number": self.number,
            "width_mm": self.width_mm,
            "height_mm": self.height_mm,
            "rotation": self.rotation,
            "image_xobjects": self.image_xobjects,
            "form_xobjects": self.form_xobjects,
            "annotation_count": self.annotation_count,
            "has_transparency": self.has_transparency,
            "active_content_types": list(self.active_content_types),
        }


@dataclass(frozen=True, slots=True)
class ManuscriptAudit:
    """Immutable manuscript-wide audit result."""

    path: Path
    sha256: str
    page_count: int
    pages: tuple[ManuscriptPageAudit, ...]
    font_count: int
    unembedded_font_count: int
    type3_font_count: int
    active_content_types: tuple[str, ...]
    page_sizes_uniform: bool
    warnings: tuple[str, ...]
    placement_audit: ManuscriptPlacementAudit | None = None

    @property
    def has_active_content(self) -> bool:
        return bool(self.active_content_types)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "path": str(self.path),
            "sha256": self.sha256,
            "page_count": self.page_count,
            "pages": [page.to_dict() for page in self.pages],
            "font_count": self.font_count,
            "unembedded_font_count": self.unembedded_font_count,
            "type3_font_count": self.type3_font_count,
            "active_content_types": list(self.active_content_types),
            "page_sizes_uniform": self.page_sizes_uniform,
            "warnings": list(self.warnings),
        }
        if self.placement_audit is not None:
            payload["placement_audit"] = self.placement_audit.to_dict()
        return payload


@dataclass(slots=True)
class _PageResources:
    seen_resources: set[tuple[str, int, int]]
    seen_xobjects: set[tuple[str, int, int]]
    image_count: int = 0
    form_count: int = 0
    has_transparency: bool = False


def _page_resource_audit(resources_ref: Any, path: Path) -> _PageResources:
    result = _PageResources(set(), set())

    def inspect(resources_value: Any, depth: int) -> None:
        if depth > _MAX_RESOURCE_DEPTH:
            raise ArtifactParseError(
                f"Refusing to audit manuscript {path}: page-resource nesting exceeds "
                f"{_MAX_RESOURCE_DEPTH} levels."
            )
        resources = _resolve_pdf(resources_value)
        if not hasattr(resources, "get"):
            return
        resource_key = _object_key(resources_value, resources)
        if resource_key in result.seen_resources:
            return
        result.seen_resources.add(resource_key)
        if len(result.seen_resources) > _MAX_PAGE_RESOURCES:
            raise ArtifactParseError(
                f"Refusing to audit manuscript {path}: a page exceeds "
                f"{_MAX_PAGE_RESOURCES} resource dictionaries."
            )

        states = _resolve_pdf(resources.get("/ExtGState"))
        if hasattr(states, "items"):
            for _, state_ref in states.items():
                state = _resolve_pdf(state_ref)
                if not hasattr(state, "get"):
                    continue
                for key in ("/ca", "/CA"):
                    try:
                        if state.get(key) is not None and float(state.get(key)) < 1.0:
                            result.has_transparency = True
                    except (TypeError, ValueError):
                        pass
                if state.get("/SMask") not in (None, "/None"):
                    result.has_transparency = True
                if state.get("/BM") not in (None, "/Normal", "/Compatible"):
                    result.has_transparency = True

        xobjects = _resolve_pdf(resources.get("/XObject"))
        if hasattr(xobjects, "items"):
            for _, object_ref in xobjects.items():
                obj = _resolve_pdf(object_ref)
                object_key = _object_key(object_ref, obj)
                if object_key in result.seen_xobjects:
                    continue
                result.seen_xobjects.add(object_key)
                subtype = str(obj.get("/Subtype", "")) if hasattr(obj, "get") else ""
                if subtype == "/Image":
                    result.image_count += 1
                    if obj.get("/SMask") is not None or obj.get("/Mask") is not None:
                        result.has_transparency = True
                elif subtype == "/Form":
                    result.form_count += 1
                    group = _resolve_pdf(obj.get("/Group"))
                    if hasattr(group, "get") and str(group.get("/S", "")) == "/Transparency":
                        result.has_transparency = True
                nested = obj.get("/Resources") if hasattr(obj, "get") else None
                if nested is not None:
                    inspect(nested, depth + 1)

        patterns = _resolve_pdf(resources.get("/Pattern"))
        if hasattr(patterns, "items"):
            for _, pattern_ref in patterns.items():
                pattern = _resolve_pdf(pattern_ref)
                nested = pattern.get("/Resources") if hasattr(pattern, "get") else None
                if nested is not None:
                    inspect(nested, depth + 1)

    if resources_ref is not None:
        inspect(resources_ref, 1)
    return result


def _action_types(
    value_ref: Any, *, depth: int = 0, seen: set[tuple[str, int, int]] | None = None
) -> set[str]:
    if value_ref is None or depth > _MAX_RESOURCE_DEPTH:
        return set()
    found: set[str] = set()
    active_seen = seen if seen is not None else set()
    value = _resolve_pdf(value_ref)
    if isinstance(value, (list, tuple)):
        for item in value:
            found.update(_action_types(item, depth=depth + 1, seen=active_seen))
        return found
    if not hasattr(value, "get"):
        return found
    key = _object_key(value_ref, value)
    if key in active_seen:
        return found
    active_seen.add(key)
    action_type = str(value.get("/S", "")).lstrip("/")
    if action_type:
        found.add(action_type)
    nested_values = (
        (value.get(key) for key in ("/Next", "/A", "/AA")) if action_type else value.values()
    )
    for nested in nested_values:
        found.update(_action_types(nested, depth=depth + 1, seen=active_seen))
    return found


def _annotation_audit(page: Any) -> tuple[int, set[str]]:
    annotations_ref = page.get("/Annots")
    if annotations_ref is None:
        return 0, set()
    annotations = _resolve_pdf(annotations_ref)
    try:
        count = len(annotations)
    except TypeError as exc:
        raise ArtifactParseError(f"PDF annotation array is malformed: {exc}") from exc
    if count > _MAX_ANNOTATIONS_PER_PAGE:
        raise ArtifactParseError(
            f"A manuscript page has more than {_MAX_ANNOTATIONS_PER_PAGE} annotations."
        )
    active: set[str] = set()
    for annotation_ref in annotations:
        annotation = _resolve_pdf(annotation_ref)
        subtype = str(annotation.get("/Subtype", "")).lstrip("/")
        if subtype in {"RichMedia", "Movie", "Sound", "FileAttachment", "Screen", "3D"}:
            active.add(subtype)
        active.update(_action_types(annotation.get("/A")))
        active.update(_action_types(annotation.get("/AA")))
    return count, active


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _tuple(value: object) -> tuple[object, ...]:
    return value if isinstance(value, tuple) else ()


def audit_manuscript_pdf(
    path: str | Path,
    *,
    max_pages: int = _MAX_MANUSCRIPT_PAGES,
    figures: Sequence[FigureSpec] | None = None,
    matching_hints: Sequence[ManuscriptMatchHint] = (),
) -> ManuscriptAudit:
    """Audit passive PDF structure and, when configured, conservative placements."""

    if max_pages <= 0 or max_pages > _MAX_MANUSCRIPT_PAGES:
        raise ValueError(f"max_pages must be between 1 and {_MAX_MANUSCRIPT_PAGES}.")
    file_path = Path(path).expanduser().resolve()
    inspection = inspect_artifact(file_path)
    if inspection.format != "pdf":
        raise ValueError(f"Manuscript audit requires PDF content, received {inspection.format}.")
    page_count = int(_number(inspection.get("pdf.page_count", 0)))
    if page_count > max_pages:
        raise ArtifactParseError(
            f"Manuscript has {page_count} pages; configured audit limit is {max_pages}."
        )
    try:
        reader = PdfReader(file_path, strict=False)
        pages: list[ManuscriptPageAudit] = []
        for number, page in enumerate(reader.pages, start=1):
            resources = _page_resource_audit(page.get("/Resources"), file_path)
            annotation_count, page_active = _annotation_audit(page)
            page_active.update(_action_types(page.get("/AA")))
            width = _number(inspection.get(f"pdf.page.{number}.width_mm", 0.0))
            height = _number(inspection.get(f"pdf.page.{number}.height_mm", 0.0))
            rotation = int(_number(inspection.get(f"pdf.page.{number}.rotation", 0)))
            pages.append(
                ManuscriptPageAudit(
                    number,
                    width,
                    height,
                    rotation,
                    resources.image_count,
                    resources.form_count,
                    annotation_count,
                    resources.has_transparency,
                    tuple(sorted(page_active)),
                )
            )
    except (OSError, PyPdfError, KeyError, TypeError, ValueError) as exc:
        raise ArtifactParseError(f"Could not audit manuscript PDF {file_path}: {exc}") from exc

    first_size = (pages[0].width_mm, pages[0].height_mm)
    uniform = all(
        abs(page.width_mm - first_size[0]) <= 0.5 and abs(page.height_mm - first_size[1]) <= 0.5
        for page in pages[1:]
    )
    active = set(str(item) for item in _tuple(inspection.get("pdf.active_content_types", ())))
    for page in pages:
        active.update(page.active_content_types)
    warnings = list(inspection.warnings)
    if not uniform:
        warnings.append("Manuscript page sizes vary by more than 0.5 mm.")
    unembedded = int(_number(inspection.get("pdf.unembedded_font_count", 0)))
    if unembedded:
        warnings.append(f"Manuscript contains {unembedded} unembedded font resources.")
    if any(page.has_transparency for page in pages):
        warnings.append("Manuscript contains transparency in one or more page resources.")
    placement_audit = None
    if figures is not None:
        from .manuscript_matching import match_manuscript_figures

        placement_audit = match_manuscript_figures(
            file_path,
            figures,
            hints=matching_hints,
            max_pages=max_pages,
        )
    return ManuscriptAudit(
        file_path,
        _file_digest(file_path),
        page_count,
        tuple(pages),
        int(_number(inspection.get("pdf.font_count", 0))),
        unembedded,
        int(_number(inspection.get("pdf.type3_font_count", 0))),
        tuple(sorted(active)),
        uniform,
        tuple(dict.fromkeys(warnings)),
        placement_audit,
    )


__all__ = ["ManuscriptAudit", "ManuscriptPageAudit", "audit_manuscript_pdf"]
