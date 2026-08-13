"""Privacy-safe environment provenance for reproducible compliance evidence."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
from importlib.metadata import PackageNotFoundError, version

import matplotlib
from matplotlib import rcParams


def _version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"


def _rcparams_digest() -> str:
    payload = {
        key: repr(value) for key, value in sorted(rcParams.items()) if not key.startswith("backend")
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def collect_environment_provenance() -> tuple[tuple[str, str], ...]:
    """Return stable keys without usernames, hostnames, or filesystem paths."""

    values = {
        "researchplot_version": _version("researchplot-venues"),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "operating_system": platform.system() or sys.platform,
        "matplotlib_version": matplotlib.__version__,
        "matplotlib_backend": str(matplotlib.get_backend()),
        "pillow_version": _version("pillow"),
        "pypdf_version": _version("pypdf"),
        "latex_available": str(shutil.which("latex") is not None).lower(),
        "font_families": ",".join(str(item) for item in rcParams["font.family"]),
        "rcparams_sha256": _rcparams_digest(),
        "inspector_protocol": "researchplot-inspector-v1",
    }
    return tuple(sorted(values.items()))


__all__ = ["collect_environment_provenance"]
