"""Opt-in TUF-verified remote profile registry client.

There is deliberately no unsigned HTTP fallback. Base installations can
inspect capability diagnostics offline; callers must install the ``registry``
extra and supply a trusted root before any refresh is possible.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from .models import ProfileCoordinate, VenueProfile
from .registry import load_profile


class RegistryCapabilityError(RuntimeError):
    """The signed registry capability is unavailable or unsafe to use."""


@dataclass(frozen=True, slots=True)
class RegistryDiagnostic:
    available: bool
    code: str
    message: str


class RegistryClient:
    """A signed, explicitly configured profile registry."""

    def __init__(self, base_url: str, *, trusted_root: str | Path, cache_dir: str | Path):
        parts = urlsplit(base_url)
        if (
            parts.scheme.casefold() != "https"
            or not parts.hostname
            or parts.username
            or parts.password
        ):
            raise ValueError("Registry base URL must be an HTTPS origin without credentials.")
        self.base_url = base_url.rstrip("/") + "/"
        self.trusted_root = Path(trusted_root)
        self.cache_dir = Path(cache_dir)

    def diagnostic(self) -> RegistryDiagnostic:
        if importlib.util.find_spec("tuf") is None:
            return RegistryDiagnostic(
                False,
                "tuf-missing",
                "Install researchplot-venues[registry] to enable signed profile updates.",
            )
        if not self.trusted_root.is_file():
            return RegistryDiagnostic(
                False,
                "trusted-root-missing",
                "A local TUF root.json is required; unsigned fallback is disabled.",
            )
        return RegistryDiagnostic(True, "ready", "TUF client and trusted root are available.")

    def _updater(self) -> object:
        diagnostic = self.diagnostic()
        if not diagnostic.available:
            raise RegistryCapabilityError(diagnostic.message)
        from tuf.ngclient import Updater  # type: ignore[import-not-found]

        metadata_dir = self.cache_dir / "metadata"
        targets_dir = self.cache_dir / "targets"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        targets_dir.mkdir(parents=True, exist_ok=True)
        try:
            if self.trusted_root.stat().st_size > 1024 * 1024:
                raise RegistryCapabilityError("Trusted TUF root exceeds the 1 MiB safety limit.")
            bootstrap = self.trusted_root.read_bytes()
        except OSError as exc:
            raise RegistryCapabilityError(f"Could not read the trusted TUF root: {exc}") from exc
        return Updater(
            metadata_dir=str(metadata_dir),
            metadata_base_url=self.base_url + "metadata/",
            target_dir=str(targets_dir),
            target_base_url=self.base_url + "targets/",
            bootstrap=bootstrap,
        )

    def refresh(self) -> None:
        updater = self._updater()
        updater.refresh()  # type: ignore[attr-defined]

    def fetch_profile(self, coordinate: str) -> VenueProfile:
        parsed = ProfileCoordinate.parse(coordinate)
        target_path = f"{parsed.namespace}/{parsed.profile_id}/{parsed.revision}.json"
        updater = self._updater()
        updater.refresh()  # type: ignore[attr-defined]
        target = updater.get_targetinfo(target_path)  # type: ignore[attr-defined]
        if target is None:
            raise ValueError(f"Signed registry has no target for {parsed}.")
        downloaded = Path(updater.download_target(target))  # type: ignore[attr-defined]
        profile = load_profile(downloaded)
        if (
            profile.namespace != parsed.namespace
            or profile.id != parsed.profile_id
            or profile.revision != parsed.revision
        ):
            raise RegistryCapabilityError("Verified target content does not match its coordinate.")
        if parsed.digest is not None and profile.digest != parsed.digest:
            raise RegistryCapabilityError("Verified target profile digest does not match the pin.")
        return profile
