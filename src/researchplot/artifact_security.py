"""Bounded artifact inspection and reproducible evidence-archive verification.

This module deliberately keeps parser isolation and archive integrity separate
from venue policy.  It never executes embedded content and never extracts an
archive to disk while verifying it.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import multiprocessing
import os
import stat
import tarfile
import tempfile
import time
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .inspectors import (
    ArtifactInspection,
    ArtifactInspectionError,
    ArtifactParseError,
    inspect_artifact,
)

_MANIFEST_NAME = "researchplot-manifest.json"
_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
_MAX_ARCHIVE_ENTRIES = 10_000
_MAX_ARCHIVE_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024
_MAX_COMPRESSION_RATIO = 1_000
_FIXED_ZIP_DATE = (1980, 1, 1, 0, 0, 0)
_FIXED_ARCHIVE_EPOCH = 315532800
_WINDOWS_FORBIDDEN = frozenset('<>:"/\\|?*')
_WINDOWS_RESERVED = {
    "aux",
    "con",
    "nul",
    "prn",
    *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
}


class InspectionTimeoutError(ArtifactInspectionError):
    """Raised when an isolated inspector exceeds its wall-clock budget."""


class InspectionResourceError(ArtifactInspectionError):
    """Raised when an isolated inspector fails at its process boundary."""


class ManifestVerificationError(ValueError):
    """Raised when a manifest or archive cannot be safely interpreted."""


@dataclass(frozen=True, slots=True)
class InspectionBudget:
    """Per-artifact limits for an isolated inspection process."""

    max_bytes: int = 512 * 1024 * 1024
    timeout_seconds: float = 15.0
    max_memory_bytes: int = 1024 * 1024 * 1024
    max_cpu_seconds: int = 10

    def __post_init__(self) -> None:
        if self.max_bytes <= 0:
            raise ValueError("max_bytes must be positive.")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive.")
        if self.max_memory_bytes <= 0:
            raise ValueError("max_memory_bytes must be positive.")
        if self.max_cpu_seconds <= 0:
            raise ValueError("max_cpu_seconds must be positive.")


@dataclass(frozen=True, slots=True)
class BatchInspectionBudget:
    """Aggregate limits for inspecting an artifact collection."""

    max_files: int = 256
    max_total_bytes: int = 1024 * 1024 * 1024
    per_artifact: InspectionBudget = InspectionBudget()

    def __post_init__(self) -> None:
        if self.max_files <= 0:
            raise ValueError("max_files must be positive.")
        if self.max_total_bytes <= 0:
            raise ValueError("max_total_bytes must be positive.")


@dataclass(frozen=True, slots=True)
class InspectionExecution:
    """Result and accounting metadata from an isolated parser process."""

    inspection: ArtifactInspection
    duration_seconds: float
    isolated: bool
    limits_applied: tuple[str, ...]


def _inspection_worker(
    connection: Any,
    path: str,
    max_memory_bytes: int,
    max_cpu_seconds: int,
) -> None:
    limits = ["process-boundary"]
    try:
        try:
            import resource

            resource_module: Any = resource
            resource_module.setrlimit(
                resource_module.RLIMIT_AS, (max_memory_bytes, max_memory_bytes)
            )
            limits.append("address-space")
            resource_module.setrlimit(
                resource_module.RLIMIT_CPU, (max_cpu_seconds, max_cpu_seconds + 1)
            )
            limits.append("cpu-time")
        except (ImportError, AttributeError, OSError, ValueError):
            # Windows and some sandboxed POSIX systems do not expose these limits.
            # The process boundary and parent wall-clock limit still apply.
            pass
        connection.send(("ok", inspect_artifact(path), tuple(limits)))
    except BaseException as exc:  # The child must report parser failures, then exit.
        try:
            connection.send(("error", type(exc).__name__, str(exc)))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


def inspect_artifact_isolated(
    path: str | Path, *, budget: InspectionBudget | None = None
) -> InspectionExecution:
    """Inspect one artifact in a disposable child process.

    On POSIX, address-space and CPU rlimits are applied when the host permits
    them.  Every platform receives a process boundary, a parent-enforced file
    size check, and a wall-clock timeout.  This is containment, not a claim of
    an operating-system filesystem sandbox.
    """

    selected = budget or InspectionBudget()
    file_path = Path(path).expanduser().resolve()
    try:
        size = file_path.stat().st_size
    except OSError as exc:
        raise ArtifactInspectionError(f"Could not access artifact {file_path}: {exc}") from exc
    if not file_path.is_file():
        raise ArtifactInspectionError(f"Artifact is not a regular file: {file_path}")
    if size > selected.max_bytes:
        raise InspectionResourceError(
            f"Refusing to inspect {file_path}: {size} bytes exceeds the isolated-parser "
            f"budget of {selected.max_bytes} bytes."
        )

    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_inspection_worker,
        args=(
            child_connection,
            str(file_path),
            selected.max_memory_bytes,
            selected.max_cpu_seconds,
        ),
        daemon=True,
    )
    started = time.monotonic()
    process.start()
    child_connection.close()
    try:
        if not parent_connection.poll(selected.timeout_seconds):
            process.terminate()
            process.join(timeout=2.0)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(timeout=2.0)
            raise InspectionTimeoutError(
                f"Inspection of {file_path} exceeded {selected.timeout_seconds:g} seconds."
            )
        try:
            message = parent_connection.recv()
        except EOFError as exc:
            process.join(timeout=2.0)
            raise InspectionResourceError(
                f"Inspector process for {file_path} exited without returning a result "
                f"(exit code {process.exitcode})."
            ) from exc
    finally:
        parent_connection.close()
        if process.is_alive():
            process.join(timeout=2.0)

    process.join(timeout=2.0)
    duration = time.monotonic() - started
    if message[0] == "ok":
        inspection = message[1]
        if not isinstance(inspection, ArtifactInspection):
            raise InspectionResourceError("Inspector process returned an invalid result.")
        return InspectionExecution(inspection, duration, True, tuple(message[2]))

    error_name, detail = str(message[1]), str(message[2])
    if error_name == "ArtifactParseError":
        raise ArtifactParseError(detail)
    if error_name == "UnsupportedArtifactError":
        from .inspectors import UnsupportedArtifactError

        raise UnsupportedArtifactError(detail)
    if error_name == "FileNotFoundError":
        raise FileNotFoundError(detail)
    raise InspectionResourceError(f"Isolated inspector failed with {error_name}: {detail}")


def inspect_artifacts_bounded(
    paths: list[str | Path] | tuple[str | Path, ...],
    *,
    budget: BatchInspectionBudget | None = None,
) -> tuple[InspectionExecution, ...]:
    """Inspect a deterministic, duplicate-free artifact collection within a budget."""

    selected = budget or BatchInspectionBudget()
    resolved = tuple(sorted({Path(path).expanduser().resolve() for path in paths}))
    if len(resolved) > selected.max_files:
        raise InspectionResourceError(
            f"Artifact collection has {len(resolved)} files; limit is {selected.max_files}."
        )
    try:
        total_bytes = sum(path.stat().st_size for path in resolved)
    except OSError as exc:
        raise ArtifactInspectionError(f"Could not stat artifact collection: {exc}") from exc
    if total_bytes > selected.max_total_bytes:
        raise InspectionResourceError(
            f"Artifact collection has {total_bytes} bytes; aggregate limit is "
            f"{selected.max_total_bytes} bytes."
        )
    return tuple(inspect_artifact_isolated(path, budget=selected.per_artifact) for path in resolved)


@dataclass(frozen=True, slots=True)
class ManifestIssue:
    """One stable integrity or portability issue."""

    code: str
    path: str | None
    message: str

    def to_dict(self) -> dict[str, object]:
        return {"code": self.code, "path": self.path, "message": self.message}


@dataclass(frozen=True, slots=True)
class ManifestVerification:
    """Integrity result for a directory or archive manifest."""

    valid: bool
    issues: tuple[ManifestIssue, ...]
    checked_artifacts: int
    manifest_digest: str

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "issues": [issue.to_dict() for issue in self.issues],
            "checked_artifacts": self.checked_artifacts,
            "manifest_digest": self.manifest_digest,
        }


@dataclass(frozen=True, slots=True)
class ArchiveResult:
    """A reproducibly encoded ZIP and its integrity result."""

    path: Path
    sha256: str
    bytes: int
    verification: ManifestVerification


def _reject_json_constant(value: str) -> None:
    raise ManifestVerificationError(f"Manifest contains non-finite JSON number {value!r}.")


def _load_manifest(payload: bytes, source: str) -> dict[str, Any]:
    if not payload or len(payload) > _MAX_MANIFEST_BYTES:
        raise ManifestVerificationError(
            f"Manifest {source} must be between 1 and {_MAX_MANIFEST_BYTES} bytes."
        )
    try:
        parsed = json.loads(payload.decode("utf-8"), parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestVerificationError(
            f"Manifest {source} is not valid UTF-8 JSON: {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ManifestVerificationError(f"Manifest {source} must contain a JSON object.")
    artifacts = parsed.get("artifacts")
    if not isinstance(artifacts, list):
        raise ManifestVerificationError(f"Manifest {source} must contain an artifacts array.")
    if len(artifacts) > _MAX_ARCHIVE_ENTRIES:
        raise ManifestVerificationError(
            f"Manifest {source} has more than {_MAX_ARCHIVE_ENTRIES} artifact records."
        )
    return parsed


def _safe_manifest_path(raw_path: object) -> tuple[PurePosixPath | None, ManifestIssue | None]:
    if not isinstance(raw_path, str) or not raw_path or "\\" in raw_path or "\x00" in raw_path:
        return None, ManifestIssue("unsafe-path", str(raw_path), "Artifact path is not portable.")
    path = PurePosixPath(raw_path)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        return None, ManifestIssue(
            "unsafe-path", raw_path, "Artifact path must be relative and remain in the bundle."
        )
    normalized_parts: list[str] = []
    for part in path.parts:
        normalized_part = unicodedata.normalize("NFC", part)
        device_name = normalized_part.split(".", 1)[0].casefold()
        if (
            normalized_part.endswith((" ", "."))
            or any(
                ord(character) < 32 or character in _WINDOWS_FORBIDDEN
                for character in normalized_part
            )
            or device_name in _WINDOWS_RESERVED
        ):
            return None, ManifestIssue(
                "unsafe-path",
                raw_path,
                "Artifact path contains a non-portable Windows path component.",
            )
        normalized_parts.append(normalized_part)
    normalized = PurePosixPath(*normalized_parts)
    return normalized, None


def _manifest_records(
    manifest: dict[str, Any],
) -> tuple[list[tuple[PurePosixPath, int, str]], list[ManifestIssue]]:
    records: list[tuple[PurePosixPath, int, str]] = []
    issues: list[ManifestIssue] = []
    portable_keys: set[str] = set()
    for index, raw in enumerate(manifest["artifacts"]):
        if not isinstance(raw, dict):
            issues.append(
                ManifestIssue("invalid-record", None, f"Artifact record {index} is not an object.")
            )
            continue
        path, path_issue = _safe_manifest_path(raw.get("path"))
        if path_issue is not None:
            issues.append(path_issue)
            continue
        assert path is not None
        key = path.as_posix().casefold()
        if key in portable_keys:
            issues.append(
                ManifestIssue(
                    "duplicate-path",
                    path.as_posix(),
                    "Artifact path collides after Unicode normalization and case folding.",
                )
            )
            continue
        portable_keys.add(key)
        size, digest = raw.get("bytes"), raw.get("sha256")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            issues.append(
                ManifestIssue("invalid-size", path.as_posix(), "Artifact byte count is invalid.")
            )
            continue
        if not isinstance(digest, str) or not len(digest) == 64:
            issues.append(
                ManifestIssue("invalid-digest", path.as_posix(), "Artifact SHA-256 is invalid.")
            )
            continue
        try:
            int(digest, 16)
        except ValueError:
            issues.append(
                ManifestIssue("invalid-digest", path.as_posix(), "Artifact SHA-256 is invalid.")
            )
            continue
        records.append((path, size, digest.casefold()))
    return records, issues


def _sha256_stream(stream: Any) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _safe_directory_file(root: Path, relative: PurePosixPath) -> Path | None:
    candidate = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except OSError:
            return None
        if stat.S_ISLNK(mode):
            return None
    try:
        return candidate if stat.S_ISREG(candidate.lstat().st_mode) else None
    except OSError:
        return None


def verify_manifest(
    root: str | Path,
    *,
    manifest_name: str = _MANIFEST_NAME,
    strict: bool = False,
) -> ManifestVerification:
    """Verify every declared file in a bundle directory without following symlinks."""

    safe_manifest, issue = _safe_manifest_path(manifest_name)
    if issue is not None or safe_manifest is None or len(safe_manifest.parts) != 1:
        raise ManifestVerificationError("manifest_name must be one safe filename.")
    root_path = Path(root).expanduser()
    if root_path.is_symlink() or not root_path.is_dir():
        raise ManifestVerificationError(f"Bundle root is not a regular directory: {root_path}")
    root_path = root_path.resolve()
    manifest_path = _safe_directory_file(root_path, safe_manifest)
    if manifest_path is None:
        raise ManifestVerificationError(f"Bundle manifest is missing or unsafe: {manifest_name}")
    try:
        payload = manifest_path.read_bytes()
    except OSError as exc:
        raise ManifestVerificationError(f"Could not read bundle manifest: {exc}") from exc
    manifest = _load_manifest(payload, str(manifest_path))
    records, issues = _manifest_records(manifest)
    declared = {path.as_posix().casefold() for path, _, _ in records}
    checked = 0
    for relative, expected_size, expected_digest in records:
        file_path = _safe_directory_file(root_path, relative)
        if file_path is None:
            issues.append(
                ManifestIssue(
                    "missing-or-unsafe",
                    relative.as_posix(),
                    "Declared artifact is missing or unsafe.",
                )
            )
            continue
        try:
            with file_path.open("rb") as stream:
                actual_digest, actual_size = _sha256_stream(stream)
        except OSError as exc:
            issues.append(ManifestIssue("unreadable", relative.as_posix(), str(exc)))
            continue
        checked += 1
        if actual_size != expected_size:
            issues.append(
                ManifestIssue(
                    "size-mismatch",
                    relative.as_posix(),
                    f"Expected {expected_size} bytes, found {actual_size}.",
                )
            )
        if not hmac.compare_digest(actual_digest, expected_digest):
            issues.append(
                ManifestIssue("digest-mismatch", relative.as_posix(), "SHA-256 does not match.")
            )
    if strict:
        manifest_key = safe_manifest.as_posix().casefold()
        for candidate in sorted(root_path.rglob("*")):
            if candidate.is_symlink():
                issues.append(
                    ManifestIssue(
                        "unsafe-entry",
                        candidate.relative_to(root_path).as_posix(),
                        "Bundle contains a symbolic link.",
                    )
                )
            elif candidate.is_file():
                key = candidate.relative_to(root_path).as_posix().casefold()
                if key != manifest_key and key not in declared:
                    issues.append(
                        ManifestIssue(
                            "undeclared-file",
                            candidate.relative_to(root_path).as_posix(),
                            "File is not covered by the manifest.",
                        )
                    )
    return ManifestVerification(
        not issues,
        tuple(issues),
        checked,
        hashlib.sha256(payload).hexdigest(),
    )


def _archive_epoch() -> int:
    raw = os.environ.get("SOURCE_DATE_EPOCH")
    if raw is None:
        return _FIXED_ARCHIVE_EPOCH
    try:
        value = int(raw)
    except ValueError as exc:
        raise ManifestVerificationError("SOURCE_DATE_EPOCH must be an integer.") from exc
    if value < 0:
        raise ManifestVerificationError("SOURCE_DATE_EPOCH must not be negative.")
    return value


def _zip_date(epoch: int) -> tuple[int, int, int, int, int, int]:
    selected = max(epoch, _FIXED_ARCHIVE_EPOCH)
    try:
        parts = time.gmtime(selected)
    except (OverflowError, OSError, ValueError) as exc:
        raise ManifestVerificationError(
            "SOURCE_DATE_EPOCH is outside the supported range."
        ) from exc
    if parts.tm_year > 2107:
        raise ManifestVerificationError("SOURCE_DATE_EPOCH exceeds the ZIP timestamp range.")
    return (
        parts.tm_year,
        parts.tm_mon,
        parts.tm_mday,
        parts.tm_hour,
        parts.tm_min,
        parts.tm_sec - parts.tm_sec % 2,
    )


def create_deterministic_archive(
    source_dir: str | Path,
    destination: str | Path,
    *,
    manifest_name: str = _MANIFEST_NAME,
    strict: bool = True,
) -> ArchiveResult:
    """Encode a verified manifest bundle as a byte-reproducible ZIP or TAR archive."""

    source = Path(source_dir).expanduser().resolve()
    output = Path(destination).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Archive destination already exists: {output}")
    if source == output or source in output.parents:
        raise ManifestVerificationError("Archive destination must be outside the source bundle.")
    verification = verify_manifest(source, manifest_name=manifest_name, strict=strict)
    if not verification.valid:
        raise ManifestVerificationError(
            "Bundle manifest verification failed: "
            + "; ".join(issue.message for issue in verification.issues[:5])
        )
    manifest = _load_manifest((source / manifest_name).read_bytes(), manifest_name)
    records, issues = _manifest_records(manifest)
    if issues:
        raise ManifestVerificationError("Manifest became invalid during archive creation.")
    archive_paths = sorted(
        {PurePosixPath(manifest_name), *(record[0] for record in records)},
        key=lambda item: item.as_posix(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        suffix = output.suffix.casefold()
        if suffix not in {".zip", ".tar"}:
            raise ValueError("Deterministic archives must use a .zip or .tar suffix.")
        epoch = _archive_epoch()
        if suffix == ".zip":
            with zipfile.ZipFile(
                temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
            ) as archive:
                for relative in archive_paths:
                    source_path = _safe_directory_file(source, relative)
                    if source_path is None:
                        raise ManifestVerificationError(
                            f"Archive input changed or became unsafe: {relative.as_posix()}"
                        )
                    zip_info = zipfile.ZipInfo(relative.as_posix(), date_time=_zip_date(epoch))
                    zip_info.compress_type = zipfile.ZIP_DEFLATED
                    zip_info.create_system = 3
                    zip_info.external_attr = (stat.S_IFREG | 0o644) << 16
                    with source_path.open("rb") as reader, archive.open(zip_info, "w") as writer:
                        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                            writer.write(chunk)
        else:
            with tarfile.open(temporary, "w", format=tarfile.PAX_FORMAT) as archive:
                for relative in archive_paths:
                    source_path = _safe_directory_file(source, relative)
                    if source_path is None:
                        raise ManifestVerificationError(
                            f"Archive input changed or became unsafe: {relative.as_posix()}"
                        )
                    tar_info = tarfile.TarInfo(relative.as_posix())
                    tar_info.size = source_path.stat().st_size
                    tar_info.mtime = epoch
                    tar_info.mode = 0o644
                    tar_info.uid = tar_info.gid = 0
                    tar_info.uname = tar_info.gname = ""
                    with source_path.open("rb") as reader:
                        archive.addfile(tar_info, reader)
        try:
            os.link(temporary, output)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Archive destination was created concurrently: {output}"
            ) from exc
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    with output.open("rb") as stream:
        digest, size = _sha256_stream(stream)
    return ArchiveResult(output, digest, size, verification)


def _verify_tar_archive(
    archive_path: Path,
    *,
    manifest_name: str,
    require_deterministic_metadata: bool,
) -> ManifestVerification:
    issues: list[ManifestIssue] = []
    expected_epoch = _archive_epoch()
    payload = b""
    checked = 0
    try:
        with tarfile.open(archive_path, "r:") as archive:
            entries = archive.getmembers()
            if len(entries) > _MAX_ARCHIVE_ENTRIES:
                raise ManifestVerificationError(
                    f"Archive has more than {_MAX_ARCHIVE_ENTRIES} entries."
                )
            names: dict[str, tarfile.TarInfo] = {}
            total_size = 0
            for info in entries:
                relative, path_issue = _safe_manifest_path(info.name)
                if path_issue is not None or relative is None:
                    issues.append(path_issue or ManifestIssue("unsafe-path", info.name, "Unsafe."))
                    continue
                if not info.isreg():
                    issues.append(
                        ManifestIssue(
                            "unsafe-entry",
                            info.name,
                            "Only regular TAR members are accepted.",
                        )
                    )
                    continue
                key = relative.as_posix().casefold()
                if key in names:
                    issues.append(
                        ManifestIssue("duplicate-path", info.name, "Archive path collides.")
                    )
                    continue
                names[key] = info
                total_size += info.size
                if total_size > _MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                    raise ManifestVerificationError("Archive exceeds the uncompressed-size budget.")
                if require_deterministic_metadata and (
                    info.mtime != expected_epoch
                    or info.mode != 0o644
                    or info.uid != 0
                    or info.gid != 0
                    or info.uname
                    or info.gname
                ):
                    issues.append(
                        ManifestIssue(
                            "non-deterministic-metadata",
                            info.name,
                            "TAR timestamp, ownership, or permissions are not canonical.",
                        )
                    )
            manifest_key = PurePosixPath(manifest_name).as_posix().casefold()
            manifest_entry = names.get(manifest_key)
            if manifest_entry is None:
                raise ManifestVerificationError(f"Archive does not contain {manifest_name}.")
            if manifest_entry.size > _MAX_MANIFEST_BYTES:
                raise ManifestVerificationError("Archived manifest exceeds the size budget.")
            manifest_stream = archive.extractfile(manifest_entry)
            if manifest_stream is None:
                raise ManifestVerificationError("Archived manifest is unreadable.")
            payload = manifest_stream.read(_MAX_MANIFEST_BYTES + 1)
            manifest = _load_manifest(payload, f"{archive_path}!/{manifest_name}")
            records, record_issues = _manifest_records(manifest)
            issues.extend(record_issues)
            declared = {relative.as_posix().casefold() for relative, _, _ in records}
            for relative, expected_size, expected_digest in records:
                artifact_entry = names.get(relative.as_posix().casefold())
                if artifact_entry is None:
                    issues.append(
                        ManifestIssue(
                            "missing", relative.as_posix(), "Declared artifact is absent."
                        )
                    )
                    continue
                stream = archive.extractfile(artifact_entry)
                if stream is None:
                    issues.append(
                        ManifestIssue(
                            "unreadable", relative.as_posix(), "Archive member is unreadable."
                        )
                    )
                    continue
                digest, size = _sha256_stream(stream)
                checked += 1
                if size != expected_size:
                    issues.append(
                        ManifestIssue("size-mismatch", relative.as_posix(), "Byte count differs.")
                    )
                if not hmac.compare_digest(digest, expected_digest):
                    issues.append(
                        ManifestIssue("digest-mismatch", relative.as_posix(), "SHA-256 differs.")
                    )
            for key, info in names.items():
                if key != manifest_key and key not in declared:
                    issues.append(
                        ManifestIssue(
                            "undeclared-file", info.name, "Archive member is not in the manifest."
                        )
                    )
    except (OSError, tarfile.TarError, RuntimeError) as exc:
        raise ManifestVerificationError(f"Could not verify archive {archive_path}: {exc}") from exc
    return ManifestVerification(
        not issues,
        tuple(issues),
        checked,
        hashlib.sha256(payload).hexdigest(),
    )


def verify_deterministic_archive(
    path: str | Path,
    *,
    manifest_name: str = _MANIFEST_NAME,
    require_deterministic_metadata: bool = True,
) -> ManifestVerification:
    """Verify a ResearchPlot ZIP or TAR in place without extracting any member."""

    archive_path = Path(path).expanduser().resolve()
    if archive_path.suffix.casefold() == ".tar":
        return _verify_tar_archive(
            archive_path,
            manifest_name=manifest_name,
            require_deterministic_metadata=require_deterministic_metadata,
        )
    if archive_path.suffix.casefold() != ".zip":
        raise ManifestVerificationError("Archive must use a .zip or .tar suffix.")
    issues: list[ManifestIssue] = []
    expected_zip_date = _zip_date(_archive_epoch())
    try:
        with zipfile.ZipFile(archive_path) as archive:
            entries = archive.infolist()
            if len(entries) > _MAX_ARCHIVE_ENTRIES:
                raise ManifestVerificationError(
                    f"Archive has more than {_MAX_ARCHIVE_ENTRIES} entries."
                )
            names: dict[str, zipfile.ZipInfo] = {}
            total_uncompressed = 0
            for info in entries:
                relative, path_issue = _safe_manifest_path(info.filename)
                if path_issue is not None or relative is None:
                    issues.append(
                        path_issue or ManifestIssue("unsafe-path", info.filename, "Unsafe.")
                    )
                    continue
                unix_type = (info.external_attr >> 16) & 0o170000
                if info.is_dir() or unix_type == stat.S_IFLNK:
                    issues.append(
                        ManifestIssue(
                            "unsafe-entry",
                            info.filename,
                            "Archive directories and symbolic links are not accepted.",
                        )
                    )
                    continue
                key = relative.as_posix().casefold()
                if key in names:
                    issues.append(
                        ManifestIssue("duplicate-path", info.filename, "Archive path collides.")
                    )
                    continue
                names[key] = info
                total_uncompressed += info.file_size
                if total_uncompressed > _MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                    raise ManifestVerificationError("Archive exceeds the uncompressed-size budget.")
                if info.file_size and info.compress_size == 0:
                    raise ManifestVerificationError(
                        "Archive contains an impossible compression ratio."
                    )
                if (
                    info.compress_size
                    and info.file_size / info.compress_size > _MAX_COMPRESSION_RATIO
                ):
                    raise ManifestVerificationError("Archive exceeds the compression-ratio budget.")
                if require_deterministic_metadata and (
                    info.date_time != expected_zip_date
                    or info.create_system != 3
                    or ((info.external_attr >> 16) & 0o777) != 0o644
                ):
                    issues.append(
                        ManifestIssue(
                            "non-deterministic-metadata",
                            info.filename,
                            "ZIP timestamp, creator, or permissions are not canonical.",
                        )
                    )

            manifest_key = PurePosixPath(manifest_name).as_posix().casefold()
            manifest_entry = names.get(manifest_key)
            if manifest_entry is None:
                raise ManifestVerificationError(f"Archive does not contain {manifest_name}.")
            if manifest_entry.file_size > _MAX_MANIFEST_BYTES:
                raise ManifestVerificationError("Archived manifest exceeds the size budget.")
            payload = archive.read(manifest_entry)
            manifest = _load_manifest(payload, f"{archive_path}!/{manifest_name}")
            records, record_issues = _manifest_records(manifest)
            issues.extend(record_issues)
            declared = {relative.as_posix().casefold() for relative, _, _ in records}
            checked = 0
            for relative, expected_size, expected_digest in records:
                artifact_entry = names.get(relative.as_posix().casefold())
                if artifact_entry is None:
                    issues.append(
                        ManifestIssue(
                            "missing", relative.as_posix(), "Declared artifact is absent."
                        )
                    )
                    continue
                with archive.open(artifact_entry) as stream:
                    digest, size = _sha256_stream(stream)
                checked += 1
                if size != expected_size:
                    issues.append(
                        ManifestIssue("size-mismatch", relative.as_posix(), "Byte count differs.")
                    )
                if not hmac.compare_digest(digest, expected_digest):
                    issues.append(
                        ManifestIssue("digest-mismatch", relative.as_posix(), "SHA-256 differs.")
                    )
            for key, info in names.items():
                if key != manifest_key and key not in declared:
                    issues.append(
                        ManifestIssue(
                            "undeclared-file",
                            info.filename,
                            "Archive member is not in the manifest.",
                        )
                    )
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise ManifestVerificationError(f"Could not verify archive {archive_path}: {exc}") from exc
    return ManifestVerification(
        not issues,
        tuple(issues),
        checked,
        hashlib.sha256(payload).hexdigest(),
    )


__all__ = [
    "ArchiveResult",
    "BatchInspectionBudget",
    "InspectionBudget",
    "InspectionExecution",
    "InspectionResourceError",
    "InspectionTimeoutError",
    "ManifestIssue",
    "ManifestVerification",
    "ManifestVerificationError",
    "create_deterministic_archive",
    "inspect_artifact_isolated",
    "inspect_artifacts_bounded",
    "verify_deterministic_archive",
    "verify_manifest",
]
