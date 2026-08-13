"""Deterministic JATS 1.4 and RO-Crate 1.3 metadata exports."""

from __future__ import annotations

import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from typing import Any

JATS_VERSION = "1.4"
RO_CRATE_VERSION = "1.3"
RO_CRATE_CONTEXT = "https://w3id.org/ro/crate/1.3/context"
RO_CRATE_PROFILE = "https://w3id.org/ro/crate/1.3"
XLINK_NAMESPACE = "http://www.w3.org/1999/xlink"
_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
_XML_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_MIME_TYPES = {
    "pdf": "application/pdf",
    "svg": "image/svg+xml",
    "png": "image/png",
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "tiff": "image/tiff",
    "tif": "image/tiff",
    "eps": "application/postscript",
    "csv": "text/csv",
    "json": "application/json",
}


def _manifest(value: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    path = Path(value)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Could not read submission manifest {path}: {exc}") from exc
    if not payload or len(payload) > _MAX_MANIFEST_BYTES:
        raise ValueError(f"Submission manifest must be between 1 and {_MAX_MANIFEST_BYTES} bytes.")
    try:
        parsed = json.loads(payload.decode("utf-8"), parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Submission manifest is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Submission manifest must contain a JSON object.")
    return parsed


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number {value!r} is not permitted.")


def _safe_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"Manifest path is not portable: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Manifest path must remain inside the bundle: {value!r}")
    return PurePosixPath(*(unicodedata.normalize("NFC", part) for part in path.parts)).as_posix()


def _text(value: object, field: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text when supplied.")
    if _XML_CONTROL.search(value):
        raise ValueError(f"{field} contains XML-forbidden control characters.")
    return value.strip()


def _slug(value: str, fallback: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", normalized).strip("-.")
    if not slug or not re.match(r"[A-Za-z_]", slug):
        slug = f"{fallback}-{slug}" if slug else fallback
    return slug[:128]


def _artifact_index(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_artifacts = manifest.get("artifacts", [])
    if not isinstance(raw_artifacts, list):
        raise ValueError("Manifest artifacts must be an array.")
    index: dict[str, dict[str, Any]] = {}
    for raw in raw_artifacts:
        if not isinstance(raw, dict):
            raise ValueError("Every manifest artifact must be an object.")
        path = _safe_relative_path(raw.get("path"))
        key = path.casefold()
        if key in index:
            raise ValueError(f"Manifest artifact path is duplicated: {path}")
        index[key] = raw
    return index


def _metadata_paths(metadata: dict[str, Any], field: str) -> tuple[str, ...]:
    """Return portable bundle paths from a scalar-or-array metadata field."""

    raw = metadata.get(field)
    if raw is None:
        return ()
    values = (raw,) if isinstance(raw, str) else raw
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"figure metadata field {field!r} must be a path or path array.")
    return tuple(_safe_relative_path(value) for value in values)


def submission_manifest_to_jats(
    manifest: dict[str, Any] | str | Path,
    *,
    group_id: str = "researchplot-figures",
) -> str:
    """Return a JATS 1.4 ``fig-group`` fragment for submission figures.

    Captions and alt text are exported only when present; missing prose is not
    invented.  Paths become XLink references and no referenced file is read.
    """

    data = _manifest(manifest)
    figures = data.get("figures")
    if not isinstance(figures, list) or not figures:
        raise ValueError("Submission manifest must contain at least one figure.")
    _artifact_index(data)
    ET.register_namespace("xlink", XLINK_NAMESPACE)
    root = ET.Element("fig-group", {"id": _slug(group_id, "researchplot-figures")})
    used_ids: set[str] = set()
    for index, raw_figure in enumerate(figures, start=1):
        if not isinstance(raw_figure, dict):
            raise ValueError(f"Figure record {index} must be an object.")
        name = _text(raw_figure.get("name"), f"figures[{index}].name") or f"figure-{index}"
        figure_id = _slug(name, f"fig-{index}")
        if figure_id in used_ids:
            figure_id = f"{figure_id}-{index}"
        used_ids.add(figure_id)
        figure = ET.SubElement(root, "fig", {"id": figure_id})
        metadata = raw_figure.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError(f"Figure {index} metadata must be an object.")
        number = metadata.get("number")
        if number is not None and not isinstance(number, (str, int)):
            raise ValueError(f"Figure {index} number must be text or an integer.")
        label = f"Figure {number if number is not None else index}"
        ET.SubElement(figure, "label").text = label
        caption = _text(metadata.get("caption"), f"figures[{index}].metadata.caption")
        alt_text = _text(metadata.get("alt_text"), f"figures[{index}].metadata.alt_text")
        long_description = _text(
            metadata.get("long_description"),
            f"figures[{index}].metadata.long_description",
        )
        if caption:
            caption_element = ET.SubElement(figure, "caption")
            ET.SubElement(caption_element, "p").text = caption
        if long_description:
            ET.SubElement(figure, "long-desc").text = long_description
        raw_paths = raw_figure.get("paths")
        if not isinstance(raw_paths, list) or not raw_paths:
            raise ValueError(f"Figure {index} must reference at least one artifact path.")
        container = ET.SubElement(figure, "alternatives") if len(raw_paths) > 1 else figure
        for raw_path in raw_paths:
            path = _safe_relative_path(raw_path)
            extension = PurePosixPath(path).suffix.casefold().lstrip(".")
            mime = _MIME_TYPES.get(extension)
            attributes = {f"{{{XLINK_NAMESPACE}}}href": path}
            if mime and "/" in mime:
                media_type, subtype = mime.split("/", 1)
                attributes.update({"mimetype": media_type, "mime-subtype": subtype})
            graphic = ET.SubElement(container, "graphic", attributes)
            if alt_text:
                ET.SubElement(graphic, "alt-text").text = alt_text
    return ET.tostring(root, encoding="unicode", short_empty_elements=True)


def _entity_type(path: str, artifact_format: str) -> list[str] | str:
    extension = PurePosixPath(path).suffix.casefold().lstrip(".")
    normalized = artifact_format.casefold() or extension
    if normalized in {"pdf", "svg", "png", "jpeg", "jpg", "tiff", "tif", "eps"}:
        return ["File", "ImageObject"]
    if normalized in {"csv", "tsv", "json", "xml", "parquet"}:
        return ["File", "Dataset"]
    return "File"


def submission_manifest_to_ro_crate(
    manifest: dict[str, Any] | str | Path,
    *,
    name: str = "ResearchPlot submission evidence",
    description: str = "Reproducible figure artifacts and provenance generated by ResearchPlot.",
    license: str | None = None,
    creators: tuple[str, ...] | list[str] = (),
) -> dict[str, object]:
    """Return a flat RO-Crate 1.3 JSON-LD metadata graph."""

    data = _manifest(manifest)
    artifacts = _artifact_index(data)
    crate_name = _text(name, "name")
    crate_description = _text(description, "description")
    graph: list[dict[str, object]] = [
        {
            "@id": "ro-crate-metadata.json",
            "@type": "CreativeWork",
            "conformsTo": {"@id": RO_CRATE_PROFILE},
            "about": {"@id": "./"},
        }
    ]
    has_part = [
        {"@id": raw["path"]}
        for raw in sorted(artifacts.values(), key=lambda item: str(item["path"]))
    ]
    root: dict[str, object] = {
        "@id": "./",
        "@type": "Dataset",
        "name": crate_name,
        "description": crate_description,
        "hasPart": has_part,
        "subjectOf": {"@id": "#researchplot-profile"},
    }
    if license:
        root["license"] = {"@id": _text(license, "license")}
    creator_refs: list[dict[str, str]] = []
    people: list[dict[str, object]] = []
    used_person_ids: set[str] = set()
    for index, creator in enumerate(creators, start=1):
        creator_name = _text(creator, f"creators[{index}]")
        if not creator_name:
            continue
        person_id = f"#person-{_slug(creator_name, str(index)).casefold()}"
        if person_id in used_person_ids:
            person_id = f"{person_id}-{index}"
        used_person_ids.add(person_id)
        creator_refs.append({"@id": person_id})
        people.append({"@id": person_id, "@type": "Person", "name": creator_name})
    if creator_refs:
        root["creator"] = creator_refs
    graph.append(root)

    profile_entity: dict[str, object] = {
        "@id": "#researchplot-profile",
        "@type": "CreativeWork",
        "name": "ResearchPlot venue profile",
        "identifier": str(data.get("profile", "unspecified")),
    }
    profile_digest = data.get("profile_digest")
    if isinstance(profile_digest, str) and profile_digest:
        profile_entity["sha256"] = profile_digest
    sources = data.get("sources")
    if isinstance(sources, list):
        source_urls = sorted(
            {
                str(source.get("url"))
                for source in sources
                if isinstance(source, dict) and source.get("url")
            }
        )
        if source_urls:
            profile_entity["citation"] = [{"@id": url} for url in source_urls]
    graph.append(profile_entity)

    figures = data.get("figures", [])
    if not isinstance(figures, list):
        raise ValueError("Manifest figures must be an array when supplied.")
    figure_entities: list[dict[str, object]] = []
    for index, raw_figure in enumerate(figures, start=1):
        if not isinstance(raw_figure, dict):
            raise ValueError(f"Figure record {index} must be an object.")
        figure_name = _text(raw_figure.get("name"), f"figures[{index}].name")
        if not figure_name:
            figure_name = f"figure-{index}"
        metadata = raw_figure.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError(f"Figure {index} metadata must be an object.")
        raw_paths = raw_figure.get("paths")
        if not isinstance(raw_paths, list) or not raw_paths:
            raise ValueError(f"Figure {index} must reference at least one artifact path.")
        figure_paths = tuple(_safe_relative_path(path) for path in raw_paths)
        related_paths = tuple(
            dict.fromkeys(
                (
                    *_metadata_paths(metadata, "source_data"),
                    *_metadata_paths(metadata, "data_table"),
                    *_metadata_paths(metadata, "attachments"),
                )
            )
        )
        for referenced in (*figure_paths, *related_paths):
            if referenced.casefold() not in artifacts:
                raise ValueError(
                    f"Figure {index} references artifact {referenced!r} that is absent "
                    "from the manifest artifact index."
                )
        figure_id = f"#figure-{_slug(figure_name, str(index)).casefold()}"
        figure_entity: dict[str, object] = {
            "@id": figure_id,
            "@type": "ImageObject",
            "name": figure_name,
            "isPartOf": {"@id": "./"},
            "associatedMedia": [{"@id": path} for path in figure_paths],
        }
        caption = _text(metadata.get("caption"), f"figures[{index}].metadata.caption")
        alt_text = _text(metadata.get("alt_text"), f"figures[{index}].metadata.alt_text")
        long_description = _text(
            metadata.get("long_description"),
            f"figures[{index}].metadata.long_description",
        )
        if caption:
            figure_entity["caption"] = caption
        if alt_text:
            figure_entity["abstract"] = alt_text
        if long_description:
            figure_entity["description"] = long_description
        if related_paths:
            figure_entity["hasPart"] = [{"@id": path} for path in related_paths]
        figure_entities.append(figure_entity)
        has_part.append({"@id": figure_id})
    graph.extend(figure_entities)

    for _, raw in sorted(artifacts.items()):
        path = _safe_relative_path(raw.get("path"))
        artifact_format = str(raw.get("format", ""))
        entity: dict[str, object] = {
            "@id": path,
            "@type": _entity_type(path, artifact_format),
            "name": PurePosixPath(path).name,
            "encodingFormat": _MIME_TYPES.get(
                artifact_format.casefold(),
                _MIME_TYPES.get(
                    PurePosixPath(path).suffix.casefold().lstrip("."), "application/octet-stream"
                ),
            ),
            "isPartOf": {"@id": "./"},
        }
        size = raw.get("bytes")
        if isinstance(size, int) and not isinstance(size, bool) and size >= 0:
            entity["contentSize"] = str(size)
        digest = raw.get("sha256")
        if isinstance(digest, str) and digest:
            entity["sha256"] = digest
        graph.append(entity)
    graph.extend(people)
    graph.sort(
        key=lambda entity: (
            0 if entity["@id"] == "ro-crate-metadata.json" else 1,
            str(entity["@id"]),
        )
    )
    return {"@context": RO_CRATE_CONTEXT, "@graph": graph}


def write_ro_crate_metadata(
    crate: dict[str, object],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write canonical RO-Crate JSON, exclusively unless overwrite is explicit."""

    destination = Path(path)
    mode = "w" if overwrite else "x"
    with destination.open(mode, encoding="utf-8", newline="\n") as stream:
        json.dump(crate, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return destination


__all__ = [
    "JATS_VERSION",
    "RO_CRATE_CONTEXT",
    "RO_CRATE_PROFILE",
    "RO_CRATE_VERSION",
    "submission_manifest_to_jats",
    "submission_manifest_to_ro_crate",
    "write_ro_crate_metadata",
]
