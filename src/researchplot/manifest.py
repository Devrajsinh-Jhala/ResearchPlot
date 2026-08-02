"""Reproducible evidence manifests for exported figure artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .compliance import Report, TargetContext
from .models import SourceRef, VenueProfile

MANIFEST_SCHEMA_VERSION = 1


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version() -> str:
    try:
        return version("researchplot-venues")
    except PackageNotFoundError:
        return "0+unknown"


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """One immutable exported artifact entry."""

    path: str
    sha256: str
    bytes: int
    format: str

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        relative_to: str | Path | None = None,
        artifact_format: str | None = None,
    ) -> ArtifactRecord:
        file_path = Path(path)
        stored = file_path.relative_to(relative_to).as_posix() if relative_to else file_path.name
        selected_format = artifact_format or file_path.suffix.casefold().lstrip(".")
        selected_format = {"jpg": "jpeg", "tif": "tiff"}.get(
            selected_format.casefold(), selected_format.casefold()
        )
        if not re.fullmatch(r"[a-z0-9]+", selected_format):
            selected_format = "data"
        return cls(
            path=stored,
            sha256=sha256_file(file_path),
            bytes=file_path.stat().st_size,
            format=selected_format,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "format": self.format,
        }


@dataclass(frozen=True, slots=True)
class ExportManifest:
    """Machine-readable evidence connecting one export, profile, and report."""

    profile: str
    profile_digest: str
    target: TargetContext
    artifacts: tuple[ArtifactRecord, ...]
    report: Report
    metadata: dict[str, object]
    sources: tuple[SourceRef, ...] = ()
    caveats: tuple[str, ...] = ()
    researchplot_version: str = ""
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.researchplot_version:
            object.__setattr__(self, "researchplot_version", _package_version())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "researchplot_version": self.researchplot_version,
            "profile": self.profile,
            "profile_digest": self.profile_digest,
            "sources": [source.to_dict() for source in self.sources],
            "caveats": list(self.caveats),
            "target": self.target.to_dict(),
            "artifacts": [item.to_dict() for item in self.artifacts],
            "metadata": self.metadata,
            "report": self.report.to_dict(),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    def write(self, path: str | Path) -> Path:
        output = Path(path)
        output.write_text(self.to_json() + "\n", encoding="utf-8")
        return output


def profile_digest(profile: VenueProfile) -> str:
    value = getattr(profile, "digest", "")
    if value:
        return str(value)
    payload: dict[str, Any] = profile.to_dict()
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
