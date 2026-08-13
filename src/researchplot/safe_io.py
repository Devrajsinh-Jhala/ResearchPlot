"""Small fail-closed filesystem and JSON helpers used by public workflows."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any


def _is_reparse_point(path_stat: os.stat_result) -> bool:
    attributes = int(getattr(path_stat, "st_file_attributes", 0))
    flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    return bool(flag and attributes & flag)


def validate_output_path(path: str | Path, *, allow_existing: bool = True) -> Path:
    """Reject symlink, junction, directory, and non-regular output targets."""

    destination = Path(path)
    if not destination.name:
        raise ValueError("Output path must name a file.")
    try:
        target_stat = destination.lstat()
    except FileNotFoundError:
        target_stat = None
    if target_stat is not None:
        if stat.S_ISLNK(target_stat.st_mode) or _is_reparse_point(target_stat):
            raise ValueError(f"Refusing to write through symlink or junction {destination}.")
        if not stat.S_ISREG(target_stat.st_mode):
            raise ValueError(f"Output target {destination} is not a regular file.")
        if not allow_existing:
            raise FileExistsError(destination)
    parent = destination.parent if destination.parent != Path("") else Path(".")
    parent.mkdir(parents=True, exist_ok=True)
    parent_stat = parent.lstat()
    if stat.S_ISLNK(parent_stat.st_mode) or _is_reparse_point(parent_stat):
        raise ValueError(f"Refusing to write inside symlink or junction directory {parent}.")
    if not stat.S_ISDIR(parent_stat.st_mode):
        raise ValueError(f"Output parent {parent} is not a directory.")
    return destination


def atomic_write_bytes(
    path: str | Path,
    payload: bytes,
    *,
    allow_existing: bool = True,
) -> Path:
    """Write bytes through an exclusive same-directory temporary file and replace."""

    destination = validate_output_path(path, allow_existing=allow_existing)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # Re-check after staging to narrow replacement races and never follow links.
        validate_output_path(destination, allow_existing=allow_existing)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def atomic_write_text(
    path: str | Path,
    text: str,
    *,
    allow_existing: bool = True,
) -> Path:
    """Atomically write UTF-8 text without following a final symlink."""

    return atomic_write_bytes(path, text.encode("utf-8"), allow_existing=allow_existing)


def strict_json_dumps(payload: Any, *, indent: int | None = 2) -> str:
    """Serialize finite JSON, rejecting NaN and non-JSON values."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        indent=indent,
        sort_keys=False,
    )


def atomic_write_json(
    path: str | Path,
    payload: Any,
    *,
    allow_existing: bool = True,
) -> Path:
    """Serialize strict JSON and write it atomically."""

    return atomic_write_text(
        path,
        strict_json_dumps(payload) + "\n",
        allow_existing=allow_existing,
    )


__all__ = [
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_text",
    "strict_json_dumps",
    "validate_output_path",
]
