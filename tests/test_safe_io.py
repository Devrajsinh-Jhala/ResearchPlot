from __future__ import annotations

import math

import pytest

from researchplot.safe_io import atomic_write_json, atomic_write_text, strict_json_dumps


def test_atomic_write_replaces_regular_file(tmp_path) -> None:
    destination = tmp_path / "report.json"
    destination.write_text("old", encoding="utf-8")
    atomic_write_text(destination, "new")
    assert destination.read_text(encoding="utf-8") == "new"
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_write_rejects_final_symlink(tmp_path) -> None:
    destination = tmp_path / "destination.txt"
    destination.write_text("sentinel", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(destination)
    except OSError:
        pytest.skip("Symlink creation is unavailable on this platform.")
    with pytest.raises(ValueError, match="symlink or junction"):
        atomic_write_text(link, "changed")
    assert destination.read_text(encoding="utf-8") == "sentinel"


def test_strict_json_rejects_non_finite_numbers(tmp_path) -> None:
    with pytest.raises(ValueError):
        strict_json_dumps({"value": math.nan})
    with pytest.raises(ValueError):
        atomic_write_json(tmp_path / "bad.json", {"value": math.inf})
    assert not (tmp_path / "bad.json").exists()
