from __future__ import annotations

import hashlib
import io
import json
import stat
import tarfile
import zipfile
from pathlib import Path

import pytest
from PIL import Image

import researchplot.artifact_security as security
from researchplot.artifact_security import (
    BatchInspectionBudget,
    InspectionBudget,
    InspectionResourceError,
    ManifestVerificationError,
    create_deterministic_archive,
    inspect_artifact_isolated,
    inspect_artifacts_bounded,
    verify_deterministic_archive,
    verify_manifest,
)
from researchplot.inspectors import ArtifactInspectionError, UnsupportedArtifactError


def _manifest(files: dict[str, bytes]) -> bytes:
    return json.dumps(
        {
            "schema_version": 1,
            "profile": "nature@2026.08.0",
            "profile_digest": "a" * 64,
            "sources": [],
            "artifacts": [
                {
                    "path": name,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "format": Path(name).suffix.lstrip("."),
                }
                for name, payload in files.items()
            ],
            "figures": [],
        },
        sort_keys=True,
    ).encode("utf-8")


def _bundle(root: Path, files: dict[str, bytes] | None = None) -> Path:
    selected = files or {"figures/figure.png": b"figure bytes"}
    for name, payload in selected.items():
        path = root / Path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    (root / "researchplot-manifest.json").write_bytes(_manifest(selected))
    return root


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_bytes": 0},
        {"timeout_seconds": 0},
        {"timeout_seconds": float("inf")},
        {"max_memory_bytes": 0},
        {"max_cpu_seconds": 0},
    ],
)
def test_inspection_budgets_reject_non_positive_or_unbounded_values(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        InspectionBudget(**kwargs)  # type: ignore[arg-type]


def test_batch_inspection_deduplicates_and_enforces_aggregate_limits(tmp_path: Path) -> None:
    artifact = tmp_path / "figure.png"
    Image.new("RGB", (2, 2), "white").save(artifact)

    executions = inspect_artifacts_bounded(
        [artifact, artifact],
        budget=BatchInspectionBudget(
            max_files=1,
            max_total_bytes=artifact.stat().st_size,
            per_artifact=InspectionBudget(timeout_seconds=20),
        ),
    )
    assert len(executions) == 1
    assert executions[0].inspection.format == "png"

    second = tmp_path / "second.png"
    second.write_bytes(artifact.read_bytes())
    with pytest.raises(InspectionResourceError, match="2 files"):
        inspect_artifacts_bounded([artifact, second], budget=BatchInspectionBudget(max_files=1))
    with pytest.raises(InspectionResourceError, match="aggregate limit"):
        inspect_artifacts_bounded(
            [artifact],
            budget=BatchInspectionBudget(max_total_bytes=artifact.stat().st_size - 1),
        )


def test_isolated_inspection_reports_input_and_parser_failures(tmp_path: Path) -> None:
    with pytest.raises(ArtifactInspectionError, match="Could not access"):
        inspect_artifact_isolated(tmp_path / "missing.png")
    with pytest.raises(ArtifactInspectionError, match="regular file"):
        inspect_artifact_isolated(tmp_path)

    unsupported = tmp_path / "figure.bin"
    unsupported.write_bytes(b"not an image")
    with pytest.raises(UnsupportedArtifactError):
        inspect_artifact_isolated(unsupported, budget=InspectionBudget(timeout_seconds=20))


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"", "between 1"),
        (b"not json", "UTF-8 JSON"),
        (b"[]", "JSON object"),
        (b'{"artifacts":null}', "artifacts array"),
        (b'{"artifacts":[],"value":NaN}', "non-finite"),
    ],
)
def test_manifest_parser_rejects_ambiguous_or_nonportable_json(
    tmp_path: Path, payload: bytes, message: str
) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "researchplot-manifest.json").write_bytes(payload)
    with pytest.raises(ManifestVerificationError, match=message):
        verify_manifest(root)


@pytest.mark.parametrize(
    ("record", "expected_code"),
    [
        ("not-an-object", "invalid-record"),
        ({"path": "CON.txt", "bytes": 0, "sha256": "0" * 64}, "unsafe-path"),
        ({"path": "figure.png ", "bytes": 0, "sha256": "0" * 64}, "unsafe-path"),
        ({"path": "figure.png", "bytes": True, "sha256": "0" * 64}, "invalid-size"),
        ({"path": "figure.png", "bytes": -1, "sha256": "0" * 64}, "invalid-size"),
        ({"path": "figure.png", "bytes": 0, "sha256": "short"}, "invalid-digest"),
        ({"path": "figure.png", "bytes": 0, "sha256": "z" * 64}, "invalid-digest"),
    ],
)
def test_manifest_records_fail_closed(record: object, expected_code: str, tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    payload = {"schema_version": 1, "artifacts": [record]}
    (root / "researchplot-manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    result = verify_manifest(root)
    assert result.valid is False
    assert expected_code in {issue.code for issue in result.issues}


def test_manifest_strict_mode_reports_missing_and_undeclared_files(tmp_path: Path) -> None:
    root = _bundle(tmp_path / "bundle")
    declared = root / "figures" / "figure.png"
    declared.unlink()
    (root / "notes.txt").write_text("undeclared", encoding="utf-8")

    result = verify_manifest(root, strict=True)
    assert result.checked_artifacts == 0
    assert {issue.code for issue in result.issues} == {
        "missing-or-unsafe",
        "undeclared-file",
    }

    with pytest.raises(ManifestVerificationError, match="safe filename"):
        verify_manifest(root, manifest_name="../manifest.json")
    with pytest.raises(ManifestVerificationError, match="regular directory"):
        verify_manifest(root / "notes.txt")


def _write_zip(
    path: Path,
    members: list[tuple[zipfile.ZipInfo, bytes]],
) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for info, payload in members:
            archive.writestr(info, payload)


def _zip_info(name: str, *, canonical: bool = False) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
    if canonical:
        info.create_system = 3
        info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info


def test_zip_verification_reports_integrity_and_metadata_failures(tmp_path: Path) -> None:
    files = {"figure.png": b"original"}
    manifest = _manifest(files)
    archive = tmp_path / "hostile.zip"
    _write_zip(
        archive,
        [
            (_zip_info("researchplot-manifest.json"), manifest),
            (_zip_info("figure.png"), b"tampered"),
            (_zip_info("extra.txt"), b"extra"),
            (_zip_info("nested/"), b""),
        ],
    )

    result = verify_deterministic_archive(archive)
    codes = {issue.code for issue in result.issues}
    assert {
        "non-deterministic-metadata",
        "digest-mismatch",
        "undeclared-file",
        "unsafe-entry",
    } <= codes

    relaxed = verify_deterministic_archive(archive, require_deterministic_metadata=False)
    assert "non-deterministic-metadata" not in {issue.code for issue in relaxed.issues}
    assert relaxed.valid is False


def test_zip_verification_rejects_missing_manifest_duplicate_and_corruption(tmp_path: Path) -> None:
    no_manifest = tmp_path / "missing.zip"
    _write_zip(no_manifest, [(_zip_info("figure.png", canonical=True), b"x")])
    with pytest.raises(ManifestVerificationError, match="does not contain"):
        verify_deterministic_archive(no_manifest)

    duplicate = tmp_path / "duplicate.zip"
    _write_zip(
        duplicate,
        [
            (_zip_info("researchplot-manifest.json", canonical=True), _manifest({})),
            (_zip_info("FIGURE.PNG", canonical=True), b"a"),
            (_zip_info("figure.png", canonical=True), b"b"),
        ],
    )
    result = verify_deterministic_archive(duplicate)
    assert "duplicate-path" in {issue.code for issue in result.issues}

    corrupt = tmp_path / "corrupt.zip"
    corrupt.write_bytes(b"not a zip")
    with pytest.raises(ManifestVerificationError, match="Could not verify"):
        verify_deterministic_archive(corrupt)
    with pytest.raises(ManifestVerificationError, match=".zip or .tar"):
        verify_deterministic_archive(tmp_path / "archive.tgz")


def test_tar_verification_rejects_nonregular_and_nondeterministic_members(tmp_path: Path) -> None:
    archive = tmp_path / "hostile.tar"
    manifest = _manifest({"missing.png": b"expected"})
    with tarfile.open(archive, "w") as output:
        info = tarfile.TarInfo("researchplot-manifest.json")
        info.size = len(manifest)
        info.mtime = 123
        info.mode = 0o600
        info.uid = 42
        info.gid = 42
        info.uname = "author"
        output.addfile(info, io.BytesIO(manifest))
        directory = tarfile.TarInfo("unsafe/")
        directory.type = tarfile.DIRTYPE
        output.addfile(directory)
        extra = tarfile.TarInfo("extra.txt")
        extra.size = 1
        output.addfile(extra, io.BytesIO(b"x"))

    result = verify_deterministic_archive(archive)
    codes = {issue.code for issue in result.issues}
    assert {"non-deterministic-metadata", "unsafe-entry", "missing", "undeclared-file"} <= codes


@pytest.mark.parametrize("value", ["not-an-integer", "-1", "4354819200"])
def test_source_date_epoch_is_validated_before_archive_creation(
    value: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _bundle(tmp_path / "bundle")
    monkeypatch.setenv("SOURCE_DATE_EPOCH", value)
    with pytest.raises(ManifestVerificationError, match="SOURCE_DATE_EPOCH"):
        create_deterministic_archive(root, tmp_path / "bundle.zip")


def test_archive_creation_never_overwrites_or_writes_inside_source(tmp_path: Path) -> None:
    root = _bundle(tmp_path / "bundle")
    destination = tmp_path / "bundle.zip"
    destination.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        create_deterministic_archive(root, destination)
    assert destination.read_bytes() == b"keep"

    with pytest.raises(ManifestVerificationError, match="outside"):
        create_deterministic_archive(root, root / "archive.zip")


def test_archive_entry_limit_is_enforced_without_large_fixtures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "many.zip"
    _write_zip(
        archive,
        [
            (_zip_info("researchplot-manifest.json", canonical=True), _manifest({})),
            (_zip_info("extra.txt", canonical=True), b"x"),
        ],
    )
    monkeypatch.setattr(security, "_MAX_ARCHIVE_ENTRIES", 1)
    with pytest.raises(ManifestVerificationError, match="more than 1"):
        verify_deterministic_archive(archive)
