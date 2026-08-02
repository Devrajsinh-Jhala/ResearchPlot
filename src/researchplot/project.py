"""Project-level TOML configuration and immutable profile locks."""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from .compliance import Policy
from .models import ContentKind, FigureRole, VenueProfile
from .registry import resolve_profile
from .target import coerce_content, coerce_role


def _string(value: object, label: str, *, required: bool = True) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    return value


@dataclass(frozen=True, slots=True)
class FigureConfig:
    path: Path
    role: FigureRole
    width: str | None
    content: ContentKind
    alt_text: str | None = None
    caption: str | None = None
    source_data: Path | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path.as_posix(),
            "role": self.role.value,
            "width": self.width,
            "content": self.content.value,
            "alt_text": self.alt_text,
            "caption": self.caption,
            "source_data": self.source_data.as_posix() if self.source_data is not None else None,
        }


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    path: Path
    profile: VenueProfile
    policy: Policy
    figures: tuple[FigureConfig, ...]

    @property
    def root(self) -> Path:
        return self.path.parent

    @classmethod
    def load(cls, path: str | Path = "researchplot.toml") -> ProjectConfig:
        config_path = Path(path).resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"ResearchPlot configuration not found: {config_path}")
        try:
            payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ValueError(f"Could not parse {config_path}: {exc}") from exc
        root_payload: object = (
            payload.get("tool", {}).get("researchplot")
            if isinstance(payload.get("tool"), dict)
            else None
        )
        settings = root_payload if isinstance(root_payload, dict) else payload
        profile_name = _string(settings.get("profile"), "profile")
        assert profile_name is not None
        profile = resolve_profile(profile_name)
        try:
            policy = Policy(str(settings.get("policy", Policy.COMPLETE.value)))
        except ValueError as exc:
            choices = ", ".join(item.value for item in Policy)
            raise ValueError(f"policy must be one of: {choices}.") from exc
        raw_figures = settings.get("figures", payload.get("figures", []))
        if not isinstance(raw_figures, list):
            raise ValueError("figures must be an array of tables.")
        figures: list[FigureConfig] = []
        for index, raw in enumerate(raw_figures):
            if not isinstance(raw, dict):
                raise ValueError(f"figures[{index}] must be a table.")
            data = cast(dict[str, Any], raw)
            raw_path = _string(data.get("path"), f"figures[{index}].path")
            width = _string(data.get("width"), f"figures[{index}].width", required=False)
            assert raw_path is not None
            path_value = Path(raw_path)
            resolved_path = (
                path_value if path_value.is_absolute() else config_path.parent / path_value
            )
            role = coerce_role(str(data.get("role", FigureRole.MAIN.value)))
            content = coerce_content(str(data.get("content", ContentKind.DATA_VISUALIZATION.value)))
            source_value = _string(
                data.get("source_data"),
                f"figures[{index}].source_data",
                required=False,
            )
            source_path = Path(source_value) if source_value is not None else None
            resolved_source = (
                source_path
                if source_path is None or source_path.is_absolute()
                else config_path.parent / source_path
            )
            figures.append(
                FigureConfig(
                    path=resolved_path.resolve(),
                    role=role,
                    width=width if width is not None else profile.default_width,
                    content=content,
                    alt_text=_string(
                        data.get("alt_text"), f"figures[{index}].alt_text", required=False
                    ),
                    caption=_string(
                        data.get("caption"), f"figures[{index}].caption", required=False
                    ),
                    source_data=resolved_source.resolve() if resolved_source is not None else None,
                )
            )
        return cls(config_path, profile, policy, tuple(figures))


def write_profile_lock(
    profile: VenueProfile,
    path: str | Path = "researchplot.lock.json",
) -> Path:
    """Write a deterministic lock for one resolved profile revision."""

    output = Path(path)
    data = {
        "schema_version": 1,
        "profile": str(getattr(profile, "coordinate", profile.id)),
        "digest": str(getattr(profile, "digest", "")),
        "sources": [source.to_dict() for source in profile.sources],
    }
    output.write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output
