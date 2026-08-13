from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

import researchplot as rp
from researchplot.api_types import EvidenceConfidence, EvidencePhase
from researchplot.inspector_protocol import (
    PROTOCOL_VERSION,
    InspectorAllowlist,
    InspectorAllowlistError,
    InspectorExecutable,
    InspectorFilePin,
    InspectorIntegrityError,
    InspectorLimits,
    InspectorProtocolClient,
    InspectorProtocolError,
    InspectorRemoteError,
    InspectorResourceError,
    InspectorResponseError,
    InspectorTimeoutError,
    _set_resource_limit,
)

_SUCCESS_PLUGIN = r"""
import json
import sys

request = json.load(sys.stdin)
response = {
    "protocol": request["protocol"],
    "request_id": request["request_id"],
    "status": "ok",
}
if request["method"] == "capabilities":
    response["result"] = {
        "inspector": {"id": "fixture", "version": "1.2.3"},
        "capabilities": [{
            "id": "pdf.physical-size",
            "formats": ["pdf"],
            "phases": ["file"],
            "probes": ["artifact.width_mm"],
            "confidence": "deterministic",
        }],
    }
else:
    response["result"] = {
        "capability": request["params"]["capability"],
        "observations": [{
            "probe": "artifact.width_mm",
            "value": 89.0,
            "available": True,
            "phase": "file",
            "detail": "Measured from the PDF media box.",
            "unit": "mm",
        }],
    }
json.dump(response, sys.stdout, allow_nan=False, separators=(",", ":"), sort_keys=True)
"""


def test_protocol_types_are_available_from_the_primary_namespace() -> None:
    assert rp.PROTOCOL_VERSION == PROTOCOL_VERSION
    assert rp.InspectorProtocolClient is InspectorProtocolClient
    assert rp.InspectorAllowlist is InspectorAllowlist


def _client(
    code: str,
    *,
    limits: InspectorLimits | None = None,
    inspector_id: str = "fixture",
) -> InspectorProtocolClient:
    entry = InspectorExecutable(
        inspector_id,
        Path(sys.executable).resolve(),
        arguments=("-I", "-c", code),
    )
    return InspectorProtocolClient(InspectorAllowlist((entry,)), limits=limits)


def _response_plugin(result_expression: str) -> str:
    return f"""
import json
import sys
request = json.load(sys.stdin)
response = {result_expression}
json.dump(response, sys.stdout, allow_nan=False, separators=(",", ":"), sort_keys=True)
"""


def test_capability_negotiation_and_typed_inspection(tmp_path: Path) -> None:
    artifact = tmp_path / "figure.pdf"
    artifact.write_bytes(b"%PDF-fixture")
    client = _client(_SUCCESS_PLUGIN)

    diagnostic = client.diagnostic("fixture")
    capabilities = client.capabilities("fixture")
    result = client.inspect(
        "fixture",
        artifact.resolve(),
        capability="pdf.physical-size",
        options={"page": 1},
    )

    assert diagnostic.available
    assert diagnostic.code == "inspector.ready"
    assert capabilities.inspector_id == "fixture"
    assert capabilities.inspector_version == "1.2.3"
    assert capabilities.capabilities[0].phases == (EvidencePhase.FILE,)
    assert capabilities.capabilities[0].confidence is EvidenceConfidence.DETERMINISTIC
    observations = tuple(result.observations)
    assert result.artifact_sha256 == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert observations[0].probe == "artifact.width_mm"
    assert observations[0].value == 89.0
    assert observations[0].producer == "external:fixture"
    assert observations[0].supported_formats == ("pdf",)


def test_unknown_inspector_and_capability_never_fall_back(tmp_path: Path) -> None:
    client = _client(_SUCCESS_PLUGIN)
    artifact = tmp_path / "figure.pdf"
    artifact.write_bytes(b"%PDF-fixture")

    diagnostic = client.diagnostic("unknown")
    assert not diagnostic.available
    assert diagnostic.code == "inspector.not_allowlisted"
    with pytest.raises(InspectorAllowlistError, match="explicit executable allowlist"):
        client.capabilities("unknown")
    with pytest.raises(InspectorProtocolError, match="does not declare capability"):
        client.inspect("fixture", artifact.resolve(), capability="pdf.active-content")


def test_executable_and_support_file_integrity_pins_are_checked(tmp_path: Path) -> None:
    support = tmp_path / "inspector.py"
    support.write_text("# trusted\n", encoding="utf-8")
    support_digest = hashlib.sha256(support.read_bytes()).hexdigest()
    executable = Path(sys.executable).resolve()
    entry = InspectorExecutable(
        "fixture",
        executable,
        sha256="0" * 64,
        arguments=("-I", "-c", _SUCCESS_PLUGIN),
        support_files=(InspectorFilePin(support.resolve(), support_digest),),
    )
    client = InspectorProtocolClient(InspectorAllowlist((entry,)))

    diagnostic = client.diagnostic("fixture")
    assert not diagnostic.available
    assert diagnostic.code == "inspector.integrity"
    with pytest.raises(InspectorIntegrityError, match="SHA-256 pin"):
        client.capabilities("fixture")

    valid_entry = InspectorExecutable(
        "fixture",
        executable,
        arguments=("-I", "-c", _SUCCESS_PLUGIN),
        support_files=(InspectorFilePin(support.resolve(), support_digest),),
    )
    support.write_text("# replaced\n", encoding="utf-8")
    with pytest.raises(InspectorIntegrityError, match="support file"):
        InspectorProtocolClient(InspectorAllowlist((valid_entry,))).capabilities("fixture")


def test_remote_errors_remain_structured() -> None:
    code = _response_plugin(
        "{"
        "'protocol': request['protocol'], "
        "'request_id': request['request_id'], "
        "'status': 'error', "
        "'error': {'code': 'fixture.unsupported', 'message': 'No PDF engine.'}"
        "}"
    )
    client = _client(code)

    with pytest.raises(InspectorRemoteError) as caught:
        client.capabilities("fixture")
    assert caught.value.remote_code == "fixture.unsupported"
    assert caught.value.to_dict()["code"] == "inspector.remote_error"


@pytest.mark.parametrize(
    "result_expression, match",
    [
        (
            "{'protocol': request['protocol'], 'request_id': request['request_id'], "
            "'status': 'ok', 'result': {'inspector': {'id': 'fixture', 'version': '1'}, "
            "'capabilities': []}, 'surprise': True}",
            "unknown keys",
        ),
        (
            "{'protocol': 'researchplot-inspector-v0', "
            "'request_id': request['request_id'], 'status': 'ok', "
            "'result': {'inspector': {'id': 'fixture', 'version': '1'}, "
            "'capabilities': []}}",
            "incompatible protocol",
        ),
        (
            "{'protocol': request['protocol'], 'request_id': 'wrong', 'status': 'ok', "
            "'result': {'inspector': {'id': 'fixture', 'version': '1'}, "
            "'capabilities': []}}",
            "request_id",
        ),
    ],
)
def test_response_envelope_is_strict(result_expression: str, match: str) -> None:
    client = _client(_response_plugin(result_expression))
    with pytest.raises(InspectorResponseError, match=match):
        client.capabilities("fixture")


def test_observations_must_match_negotiated_contract(tmp_path: Path) -> None:
    code = _SUCCESS_PLUGIN.replace(
        '"available": True,',
        '"available": "yes",',
    )
    artifact = tmp_path / "figure.pdf"
    artifact.write_bytes(b"%PDF-fixture")

    with pytest.raises(InspectorResponseError, match="available must be a boolean"):
        _client(code).inspect("fixture", artifact.resolve(), capability="pdf.physical-size")


def test_timeout_and_output_budgets_terminate_inspectors() -> None:
    timeout_code = "import time; time.sleep(2)"
    timeout_limits = InspectorLimits(timeout_seconds=0.2, poll_interval_seconds=0.01)
    with pytest.raises(InspectorTimeoutError, match="wall-clock budget"):
        _client(timeout_code, limits=timeout_limits).capabilities("fixture")

    output_code = "import sys; sys.stdout.write('x' * 4096)"
    output_limits = InspectorLimits(max_response_bytes=256)
    with pytest.raises(InspectorResourceError, match="stdout exceeded"):
        _client(output_code, limits=output_limits).capabilities("fixture")


def test_protocol_constant_and_configuration_validation(tmp_path: Path) -> None:
    assert PROTOCOL_VERSION == "researchplot-inspector-v1"
    with pytest.raises(ValueError, match="absolute path"):
        InspectorExecutable("fixture", Path("python"))
    with pytest.raises(ValueError, match="Duplicate allowlisted"):
        entry = InspectorExecutable(
            "fixture",
            Path(sys.executable).resolve(),
            arguments=("-I", "-c", _SUCCESS_PLUGIN),
        )
        InspectorAllowlist((entry, entry))
    with pytest.raises(ValueError, match="64 hexadecimal"):
        InspectorFilePin((tmp_path / "missing").resolve(), "not-a-digest")


def test_unsupported_posix_resource_limit_is_best_effort() -> None:
    class UnsupportedLimit:
        RLIM_INFINITY = -1

        @staticmethod
        def getrlimit(_kind: int) -> tuple[int, int]:
            return (-1, -1)

        @staticmethod
        def setrlimit(_kind: int, _limits: tuple[int, int]) -> None:
            raise ValueError("unsupported by this POSIX host")

    _set_resource_limit(UnsupportedLimit(), 1, 64, 64)


def test_options_must_be_bounded_json(tmp_path: Path) -> None:
    artifact = tmp_path / "figure.pdf"
    artifact.write_bytes(b"%PDF-fixture")
    client = _client(_SUCCESS_PLUGIN)

    with pytest.raises(InspectorProtocolError, match="unsupported value type"):
        client.inspect(
            "fixture",
            artifact.resolve(),
            capability="pdf.physical-size",
            options={"invalid": object()},  # type: ignore[dict-item]
        )


def test_successful_process_must_not_write_unstructured_stderr() -> None:
    code = "import sys; sys.stderr.write('debug output'); sys.stdout.write('{}')"
    with pytest.raises(InspectorResponseError, match="stderr"):
        _client(code).capabilities("fixture")
