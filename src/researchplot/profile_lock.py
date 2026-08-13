"""Deterministic profile locks and enforcement."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .models import ProfileCoordinate, VenueProfile
from .registry import resolve_profile

_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_LOCK_BYTES = 128_000


class ProfileLockError(ValueError):
    """A profile lock is malformed, stale, or content-mismatched."""


@dataclass(frozen=True, slots=True)
class ProfileLock:
    coordinate: str
    digest: str
    document_digest: str
    source_digests: tuple[tuple[str, str | None], ...]
    schema_version: int = 2

    @classmethod
    def from_profile(cls, profile: VenueProfile) -> ProfileLock:
        return cls(
            coordinate=profile.coordinate,
            digest=profile.digest,
            document_digest=profile.document_digest,
            source_digests=tuple(
                sorted((source.id, source.content_sha256) for source in profile.sources)
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "profile": self.coordinate,
            "digest": self.digest,
            "document_digest": self.document_digest,
            "sources": [
                {"id": source_id, "content_sha256": digest}
                for source_id, digest in self.source_digests
            ],
        }


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ProfileLockError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def load_profile_lock(path: str | Path = "researchplot.lock.json") -> ProfileLock:
    """Load a bounded, strictly-shaped lock file."""

    lock_path = Path(path)
    try:
        if lock_path.stat().st_size > _MAX_LOCK_BYTES:
            raise ProfileLockError("Profile lock exceeds the 128 KB size limit.")
        data = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProfileLockError(f"Unable to read profile lock {lock_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ProfileLockError("Profile lock must be a JSON object.")
    expected = {"schema_version", "profile", "digest", "document_digest", "sources"}
    if set(data) != expected or data.get("schema_version") != 2:
        raise ProfileLockError("Profile lock must use the exact schema-version 2 shape.")
    coordinate = data.get("profile")
    if not isinstance(coordinate, str):
        raise ProfileLockError("Profile lock coordinate must be a string.")
    parsed = ProfileCoordinate.parse(coordinate)
    if parsed.digest is not None:
        raise ProfileLockError("Lock profile coordinates must not include a digest suffix.")
    raw_sources = data.get("sources")
    if not isinstance(raw_sources, list):
        raise ProfileLockError("Profile lock sources must be an array.")
    source_digests: list[tuple[str, str | None]] = []
    for index, item in enumerate(raw_sources):
        if not isinstance(item, dict) or set(item) != {"id", "content_sha256"}:
            raise ProfileLockError(f"sources[{index}] has an invalid shape.")
        source_id = item.get("id")
        digest = item.get("content_sha256")
        if not isinstance(source_id, str) or not source_id:
            raise ProfileLockError(f"sources[{index}].id must be a non-empty string.")
        if digest is not None:
            digest = _require_digest(digest, f"sources[{index}].content_sha256")
        source_digests.append((source_id, digest))
    if len({item[0] for item in source_digests}) != len(source_digests):
        raise ProfileLockError("Profile lock repeats a source id.")
    return ProfileLock(
        coordinate=str(parsed),
        digest=_require_digest(data.get("digest"), "digest"),
        document_digest=_require_digest(data.get("document_digest"), "document_digest"),
        source_digests=tuple(sorted(source_digests)),
    )


def write_profile_lock(
    profile: VenueProfile,
    path: str | Path = "researchplot.lock.json",
) -> Path:
    """Atomically write a deterministic schema-v2 profile lock."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(ProfileLock.from_profile(profile).to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


def verify_profile_lock(lock: ProfileLock, profile: VenueProfile) -> None:
    """Raise when any locked identity or evidence digest has drifted."""

    expected = ProfileLock.from_profile(profile)
    mismatches: list[str] = []
    for label in ("coordinate", "digest", "document_digest", "source_digests"):
        if getattr(lock, label) != getattr(expected, label):
            mismatches.append(label)
    if mismatches:
        raise ProfileLockError(
            f"Profile lock does not match {profile.coordinate!r}: {', '.join(mismatches)}."
        )


def resolve_locked_profile(lock: ProfileLock | str | Path) -> VenueProfile:
    """Resolve the exact coordinate and enforce every lock digest."""

    selected = load_profile_lock(lock) if isinstance(lock, (str, Path)) else lock
    profile = resolve_profile(selected.coordinate)
    verify_profile_lock(selected, profile)
    return profile
