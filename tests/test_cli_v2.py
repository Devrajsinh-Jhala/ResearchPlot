from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from matplotlib import pyplot as plt
from PIL import Image

import researchplot as rp
import researchplot.cli as cli
from researchplot.cli import main


def _nature_pdf(path: Path) -> Path:
    selected = rp.target("nature@2026.08.0", width="single", content="line-art")
    with selected.style() as style:
        fig, ax = style.subplots()
        ax.plot([0, 1], [0, 1])
        ax.set(xlabel="Input", ylabel="Response")
    return selected.export(fig, path, policy="violations").paths[0]


def _init_project(tmp_path: Path, artifact: Path) -> tuple[Path, Path]:
    config = tmp_path / "researchplot.toml"
    lock = tmp_path / "researchplot.lock.json"
    assert (
        main(
            [
                "init",
                "--profile",
                "nature@2026.08.0",
                "--figure",
                str(artifact),
                "--width",
                "single",
                "--content",
                "line-art",
                "--output",
                str(config),
                "--lock",
                str(lock),
            ]
        )
        == 0
    )
    return config, lock


def test_init_frozen_check_audit_and_html(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artifact = _nature_pdf(tmp_path / "figure.pdf")
    config, lock = _init_project(tmp_path, artifact)
    capsys.readouterr()

    spec = rp.ProjectSpec.load(config)
    assert spec.lock_path == lock
    assert main(["profiles", "verify", "--lock", str(lock), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True

    assert main(["check", "--config", str(config), "--frozen", "--json"]) == 3
    project_report = json.loads(capsys.readouterr().out)
    assert project_report["verdict"] == "indeterminate"

    assert (
        main(
            [
                "audit",
                str(artifact),
                "--profile",
                "nature@2026.08.0",
                "--width",
                "single",
                "--content",
                "line-art",
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)[0]["report"]["verdict"] == "compliant"

    sarif = tmp_path / "report.sarif"
    assert (
        main(
            [
                "audit",
                str(artifact),
                "--profile",
                "nature@2026.08.0",
                "--width",
                "double",
                "--content",
                "line-art",
                "--sarif",
                "--output",
                str(sarif),
            ]
        )
        == 1
    )
    assert json.loads(sarif.read_text(encoding="utf-8"))["version"] == "2.1.0"

    html = tmp_path / "report.html"
    assert (
        main(
            [
                "audit",
                str(artifact),
                "--profile",
                "nature@2026.08.0",
                "--width",
                "single",
                "--content",
                "line-art",
                "--html",
                str(html),
            ]
        )
        == 0
    )
    assert "<!doctype html>" in html.read_text(encoding="utf-8")


def test_frozen_direct_audit_verifies_lock_before_missing_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "audit",
            str(tmp_path / "missing.pdf"),
            "--profile",
            "nature@2026.08.0",
            "--frozen",
            "--lock",
            str(tmp_path / "missing.lock.json"),
        ]
    )
    assert code == 2
    error = capsys.readouterr().err
    assert "lock" in error.casefold()
    assert "missing.pdf" not in error


def test_migrate_writes_loadable_v3_project_and_lock(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artifact = _nature_pdf(tmp_path / "figure.pdf")
    legacy = tmp_path / "legacy.toml"
    legacy.write_text(
        f'''profile = "nature@2026.08.0"
policy = "violations"

[[figures]]
path = "{artifact.name}"
width = "single"
content = "line-art"
''',
        encoding="utf-8",
    )
    output = tmp_path / "migrated.toml"
    lock = tmp_path / "migrated.lock.json"

    assert (
        main(
            [
                "migrate",
                "--config",
                str(legacy),
                "--output",
                str(output),
                "--lock",
                str(lock),
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["project"] == output.as_posix()
    project = rp.Project.load(output)
    project.plan(frozen=True)
    assert project.spec.figures[0].deliverables[0].path == artifact


def test_bundle_build_verify_archive_jats_and_ro_crate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artifact = _nature_pdf(tmp_path / "figure.pdf")
    config, _ = _init_project(tmp_path, artifact)
    capsys.readouterr()
    bundle = tmp_path / "submission"

    assert (
        main(
            [
                "bundle",
                "build",
                "--config",
                str(config),
                "--output",
                str(bundle),
                "--policy",
                "violations",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert main(["bundle", "verify", str(bundle), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True

    archive = tmp_path / "submission.zip"
    assert main(["bundle", "archive", str(bundle), str(archive), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["sha256"]
    assert main(["bundle", "verify", str(archive)]) == 0
    capsys.readouterr()

    jats = tmp_path / "figures.xml"
    assert main(["bundle", "jats", str(bundle), "--output", str(jats)]) == 0
    assert "fig-group" in jats.read_text(encoding="utf-8")
    crate = tmp_path / "ro-crate-metadata.json"
    assert (
        main(
            [
                "bundle",
                "ro-crate",
                str(bundle),
                "--output",
                str(crate),
            ]
        )
        == 0
    )
    assert json.loads(crate.read_text(encoding="utf-8"))["@context"].endswith("/context")


def test_manuscript_and_fix_report_indeterminate_without_mutation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manuscript = tmp_path / "paper.pdf"
    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])
    fig.savefig(manuscript)
    before = manuscript.read_bytes()

    assert main(["manuscript", "check", str(manuscript), "--json"]) == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"] == "indeterminate"
    assert manuscript.read_bytes() == before

    raster = tmp_path / "figure.png"
    Image.new("RGB", (20, 20), "white").save(raster)
    plan = tmp_path / "fix.json"
    assert main(["fix", str(raster), "--plan", "--format", "json", "--output", str(plan)]) == 3
    assert json.loads(plan.read_text(encoding="utf-8"))["remediations"]
    assert Image.open(raster).size == (20, 20)


def test_retarget_is_plan_only_and_serve_delegates_safely(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _nature_pdf(tmp_path / "figure.pdf")
    config, _ = _init_project(tmp_path, artifact)
    capsys.readouterr()
    plan = tmp_path / "retarget.json"
    assert (
        main(
            [
                "project",
                "retarget",
                "--config",
                str(config),
                "--profile",
                "ieee-journal@2026.08.0",
                "--plan",
                "--output",
                str(plan),
            ]
        )
        == 3
    )
    assert json.loads(plan.read_text(encoding="utf-8"))["safe_automatic_apply"] is False
    assert (
        main(
            [
                "project",
                "retarget",
                "--config",
                str(config),
                "--profile",
                "ieee-journal@2026.08.0",
                "--plan",
                "--apply",
            ]
        )
        == 2
    )
    assert "comment-preserving" in capsys.readouterr().err

    called: dict[str, object] = {}

    def fake_serve(*, port: int, open_browser: bool) -> SimpleNamespace:
        called.update(port=port, open_browser=open_browser)
        return SimpleNamespace()

    monkeypatch.setattr("researchplot.webapp.serve", fake_serve)
    assert main(["serve", "--port", "8123", "--no-browser"]) == 0
    assert called == {"port": 8123, "open_browser": False}


def test_profile_alias_status_sync_and_doctor(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert main(["venues", "status", "nature@2026.08.0", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["passed"] is True

    class FakeRegistry:
        def __init__(self, base_url: str, *, trusted_root: str, cache_dir: str):
            self.base_url = base_url

        def diagnostic(self) -> SimpleNamespace:
            return SimpleNamespace(available=True, code="ready", message="ready")

        def refresh(self) -> None:
            return None

        def fetch_profile(self, coordinate: str) -> rp.VenueProfile:
            return rp.resolve_profile(coordinate)

    monkeypatch.setattr(cli, "RegistryClient", FakeRegistry)
    assert (
        main(
            [
                "profiles",
                "sync",
                "--base-url",
                "https://profiles.example/",
                "--trusted-root",
                str(tmp_path / "root.json"),
                "--cache-dir",
                str(tmp_path / "cache"),
                "--coordinate",
                "nature@2026.08.0",
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["refreshed"] is True
    assert main(["doctor", "--profile", "nature@2026.08.0", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["profile"] == "nature@2026.08.0"
