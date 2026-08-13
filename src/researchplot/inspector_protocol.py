"""Secure process boundary for explicitly trusted third-party inspectors.

The protocol is deliberately small and data-only.  A client sends one canonical
JSON request on standard input and accepts one JSON response on standard output.
The executable is selected by an application-owned allowlist; neither a response
nor an artifact can influence the command line.  Every inspection performs a
capability handshake before the artifact request.

This is process isolation, not an operating-system sandbox.  An allowlisted
program still has the filesystem and network privileges of the current user.
POSIX hosts receive best-effort CPU, address-space, file-size, and descriptor
limits; Windows receives parent-enforced wall-clock and output limits.  Exact
paths and optional SHA-256 pins are checked immediately before launch, but a
hostile actor able to replace files concurrently could still exploit a TOCTOU
window.  Only locally trusted inspectors should ever be allowlisted.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import subprocess
import tempfile
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias, cast

from .api_types import EvidenceConfidence, EvidencePhase
from .observations import Observation, ObservationSet

PROTOCOL_VERSION = "researchplot-inspector-v1"

_IDENTIFIER = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,127})\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FORMAT = re.compile(r"\*|[a-z0-9](?:[a-z0-9.+_-]{0,31})\Z")
_MAX_IDENTIFIER_LENGTH = 128
_MAX_VERSION_LENGTH = 128
_MAX_PATH_LENGTH = 32_768
_MAX_ARGUMENT_LENGTH = 32_768
_MAX_ARGUMENTS = 64
_MAX_JSON_DEPTH = 24
_MAX_JSON_ITEMS = 50_000
_MAX_JSON_STRING_LENGTH = 256 * 1024

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class InspectorProtocolError(RuntimeError):
    """Base class with a stable machine-readable error code."""

    code = "inspector.protocol"

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": str(self)}


class InspectorAllowlistError(InspectorProtocolError):
    """An inspector was not explicitly allowlisted."""

    code = "inspector.not_allowlisted"


class InspectorIntegrityError(InspectorProtocolError):
    """An allowlisted executable or pinned support file changed."""

    code = "inspector.integrity"


class InspectorTimeoutError(InspectorProtocolError):
    """An inspector exceeded its wall-clock budget."""

    code = "inspector.timeout"


class InspectorResourceError(InspectorProtocolError):
    """An inspector exceeded a size or process resource budget."""

    code = "inspector.resource_limit"


class InspectorExecutionError(InspectorProtocolError):
    """An inspector failed before returning a protocol response."""

    code = "inspector.execution"


class InspectorResponseError(InspectorProtocolError):
    """An inspector returned malformed or incompatible protocol data."""

    code = "inspector.invalid_response"


class InspectorRemoteError(InspectorProtocolError):
    """An inspector returned a valid structured error response."""

    code = "inspector.remote_error"

    def __init__(self, remote_code: str, message: str) -> None:
        self.remote_code = remote_code
        super().__init__(f"{remote_code}: {message}")

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "remote_code": self.remote_code, "message": str(self)}


@dataclass(frozen=True, slots=True)
class InspectorLimits:
    """Budgets applied independently to each protocol exchange."""

    timeout_seconds: float = 15.0
    max_request_bytes: int = 1024 * 1024
    max_response_bytes: int = 8 * 1024 * 1024
    max_stderr_bytes: int = 256 * 1024
    max_artifact_bytes: int = 1024 * 1024 * 1024
    max_memory_bytes: int = 1024 * 1024 * 1024
    max_cpu_seconds: int = 10
    poll_interval_seconds: float = 0.01

    def __post_init__(self) -> None:
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive.")
        if (
            not math.isfinite(self.poll_interval_seconds)
            or self.poll_interval_seconds <= 0
            or self.poll_interval_seconds > self.timeout_seconds
        ):
            raise ValueError(
                "poll_interval_seconds must be finite, positive, and no greater than the timeout."
            )
        for label in (
            "max_request_bytes",
            "max_response_bytes",
            "max_stderr_bytes",
            "max_artifact_bytes",
            "max_memory_bytes",
            "max_cpu_seconds",
        ):
            if getattr(self, label) <= 0:
                raise ValueError(f"{label} must be positive.")


@dataclass(frozen=True, slots=True)
class InspectorFilePin:
    """A support file whose content is trusted by SHA-256."""

    path: Path
    sha256: str

    def __post_init__(self) -> None:
        digest = _normalize_sha256(self.sha256, label="Pinned support-file SHA-256")
        resolved = _resolve_regular_file(self.path, label="Pinned inspector support file")
        object.__setattr__(self, "path", resolved)
        object.__setattr__(self, "sha256", digest)


@dataclass(frozen=True, slots=True)
class InspectorExecutable:
    """One exact command prefix admitted to the inspector allowlist.

    ``arguments`` are static, application-controlled arguments.  If an
    interpreter executes a script from those arguments, pin that script with
    ``support_files`` so integrity checks cover more than the interpreter.
    """

    id: str
    executable: Path
    sha256: str | None = None
    arguments: tuple[str, ...] = ()
    support_files: tuple[InspectorFilePin, ...] = ()

    def __post_init__(self) -> None:
        _validate_identifier(self.id, label="Inspector ID")
        executable = _resolve_regular_file(self.executable, label="Inspector executable")
        arguments = tuple(self.arguments)
        if len(arguments) > _MAX_ARGUMENTS:
            raise ValueError(f"Inspector arguments cannot exceed {_MAX_ARGUMENTS} entries.")
        if any(
            not isinstance(argument, str)
            or not argument
            or "\x00" in argument
            or len(argument) > _MAX_ARGUMENT_LENGTH
            for argument in arguments
        ):
            raise ValueError("Inspector arguments must be non-empty bounded strings without NULs.")
        pins = tuple(self.support_files)
        if len({pin.path for pin in pins}) != len(pins):
            raise ValueError("Inspector support-file pins must use unique paths.")
        digest = (
            None
            if self.sha256 is None
            else _normalize_sha256(self.sha256, label="Inspector executable SHA-256")
        )
        object.__setattr__(self, "executable", executable)
        object.__setattr__(self, "sha256", digest)
        object.__setattr__(self, "arguments", arguments)
        object.__setattr__(self, "support_files", pins)

    @property
    def command(self) -> tuple[str, ...]:
        """Return the immutable command prefix; no request data is appended."""

        return (str(self.executable), *self.arguments)


class InspectorAllowlist:
    """Immutable lookup table of commands explicitly trusted by the caller."""

    def __init__(self, inspectors: Iterable[InspectorExecutable]) -> None:
        entries = tuple(inspectors)
        duplicates = sorted(
            identifier
            for identifier in {entry.id for entry in entries}
            if sum(item.id == identifier for item in entries) > 1
        )
        if duplicates:
            raise ValueError(f"Duplicate allowlisted inspector IDs: {', '.join(duplicates)}.")
        self._inspectors = {entry.id: entry for entry in entries}

    def get(self, inspector_id: str) -> InspectorExecutable:
        """Return an allowlisted command or fail without launching a process."""

        entry = self._inspectors.get(inspector_id)
        if entry is None:
            raise InspectorAllowlistError(
                f"Inspector {inspector_id!r} is not in the explicit executable allowlist."
            )
        return entry

    def __iter__(self) -> Iterable[InspectorExecutable]:
        return iter(tuple(self._inspectors[key] for key in sorted(self._inspectors)))


@dataclass(frozen=True, slots=True)
class InspectorDiagnostic:
    """Deterministic availability result that does not execute the inspector."""

    inspector_id: str
    available: bool
    code: str
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "inspector_id": self.inspector_id,
            "available": self.available,
            "code": self.code,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class InspectorCapability:
    """Evidence contract declared during capability negotiation."""

    id: str
    formats: tuple[str, ...]
    phases: tuple[EvidencePhase, ...]
    probes: tuple[str, ...]
    confidence: EvidenceConfidence


@dataclass(frozen=True, slots=True)
class InspectorCapabilities:
    """Validated identity and capabilities reported by one process."""

    inspector_id: str
    inspector_version: str
    capabilities: tuple[InspectorCapability, ...]

    def get(self, capability_id: str) -> InspectorCapability | None:
        return next((item for item in self.capabilities if item.id == capability_id), None)


@dataclass(frozen=True, slots=True)
class ExternalInspectionResult:
    """Validated observations returned by an allowlisted external inspector."""

    inspector_id: str
    inspector_version: str
    capability: InspectorCapability
    artifact: Path
    artifact_sha256: str
    observations: ObservationSet


class InspectorProtocolClient:
    """Launch allowlisted inspectors through the v1 JSON subprocess protocol."""

    def __init__(
        self,
        allowlist: InspectorAllowlist,
        *,
        limits: InspectorLimits | None = None,
    ) -> None:
        self.allowlist = allowlist
        self.limits = limits or InspectorLimits()

    def diagnostic(self, inspector_id: str) -> InspectorDiagnostic:
        """Check path and pins without running third-party code."""

        try:
            entry = self.allowlist.get(inspector_id)
            _verify_entry(entry)
        except InspectorAllowlistError as exc:
            return InspectorDiagnostic(inspector_id, False, exc.code, str(exc))
        except InspectorIntegrityError as exc:
            return InspectorDiagnostic(inspector_id, False, exc.code, str(exc))
        return InspectorDiagnostic(
            inspector_id,
            True,
            "inspector.ready",
            "Inspector path and configured integrity pins are valid.",
        )

    def capabilities(self, inspector_id: str) -> InspectorCapabilities:
        """Negotiate and validate the capabilities of one allowlisted inspector."""

        entry = self.allowlist.get(inspector_id)
        response = self._exchange(entry, method="capabilities", params={})
        return _parse_capabilities(response, expected_inspector_id=inspector_id)

    def inspect(
        self,
        inspector_id: str,
        artifact: str | Path,
        *,
        capability: str,
        options: Mapping[str, JsonValue] | None = None,
    ) -> ExternalInspectionResult:
        """Negotiate a capability, then inspect one bounded regular file."""

        entry = self.allowlist.get(inspector_id)
        declared = self.capabilities(inspector_id)
        selected = declared.get(capability)
        if selected is None:
            choices = ", ".join(item.id for item in declared.capabilities) or "none"
            raise InspectorProtocolError(
                f"Inspector {inspector_id!r} does not declare capability {capability!r}; "
                f"available capabilities: {choices}."
            )

        path = _resolve_regular_file(Path(artifact), label="Inspector artifact")
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise InspectorResourceError(f"Could not stat inspector artifact: {exc}") from exc
        if size > self.limits.max_artifact_bytes:
            raise InspectorResourceError(
                f"Artifact has {size} bytes; limit is {self.limits.max_artifact_bytes} bytes."
            )
        if len(str(path)) > _MAX_PATH_LENGTH:
            raise InspectorResourceError("Artifact path exceeds the protocol path limit.")
        file_format = path.suffix.removeprefix(".").casefold()
        if "*" not in selected.formats and file_format not in selected.formats:
            choices = ", ".join(selected.formats)
            raise InspectorProtocolError(
                f"Capability {capability!r} does not support {file_format or 'extensionless'} "
                f"artifacts; supported formats: {choices}."
            )

        safe_options = dict(options or {})
        _validate_json_value(safe_options, label="Inspector options")
        artifact_digest = _sha256_file(path)
        response = self._exchange(
            entry,
            method="inspect",
            params={
                "capability": capability,
                "artifact": {
                    "path": str(path),
                    "format": file_format,
                    "size_bytes": size,
                    "sha256": artifact_digest,
                },
                "options": safe_options,
            },
        )
        observations = _parse_inspection_result(
            response,
            inspector_id=inspector_id,
            capability=selected,
            file_format=file_format,
        )
        return ExternalInspectionResult(
            inspector_id=inspector_id,
            inspector_version=declared.inspector_version,
            capability=selected,
            artifact=path,
            artifact_sha256=artifact_digest,
            observations=observations,
        )

    def _exchange(
        self,
        entry: InspectorExecutable,
        *,
        method: str,
        params: dict[str, JsonValue],
    ) -> dict[str, JsonValue]:
        _verify_entry(entry)
        request_id = _request_id(method, params)
        request: dict[str, JsonValue] = {
            "protocol": PROTOCOL_VERSION,
            "request_id": request_id,
            "method": method,
            "params": params,
        }
        encoded = _encode_json(request)
        if len(encoded) > self.limits.max_request_bytes:
            raise InspectorResourceError(
                f"Protocol request has {len(encoded)} bytes; limit is "
                f"{self.limits.max_request_bytes} bytes."
            )
        raw = _run_bounded(entry.command, encoded, self.limits)
        response = _decode_json_object(raw)
        _require_exact_keys(
            response,
            required={"protocol", "request_id", "status"},
            optional={"result", "error"},
            label="Protocol response",
        )
        if response["protocol"] != PROTOCOL_VERSION:
            raise InspectorResponseError(
                f"Inspector returned incompatible protocol {response['protocol']!r}; "
                f"expected {PROTOCOL_VERSION!r}."
            )
        if response["request_id"] != request_id:
            raise InspectorResponseError(
                "Inspector response request_id does not match the request."
            )
        status = response["status"]
        if status == "error":
            if "result" in response or "error" not in response:
                raise InspectorResponseError(
                    "Error responses must contain error and must not contain result."
                )
            _raise_remote_error(response["error"])
        if status != "ok":
            raise InspectorResponseError("Inspector response status must be 'ok' or 'error'.")
        if "error" in response or "result" not in response:
            raise InspectorResponseError(
                "Successful responses must contain result and must not contain error."
            )
        result = response["result"]
        if not isinstance(result, dict):
            raise InspectorResponseError("Inspector response result must be an object.")
        return result


def _resolve_regular_file(path: Path, *, label: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label} must use an absolute path.")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} is unavailable: {exc}") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {resolved}")
    return resolved


def _validate_identifier(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) > _MAX_IDENTIFIER_LENGTH
        or not _IDENTIFIER.fullmatch(value)
    ):
        raise ValueError(
            f"{label} must match {_IDENTIFIER.pattern!r} and be at most "
            f"{_MAX_IDENTIFIER_LENGTH} characters."
        )
    return value


def _normalize_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value.casefold()):
        raise ValueError(f"{label} must contain exactly 64 hexadecimal characters.")
    return value.casefold()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise InspectorIntegrityError(f"Could not hash {path}: {exc}") from exc
    return digest.hexdigest()


def _verify_entry(entry: InspectorExecutable) -> None:
    try:
        resolved = entry.executable.resolve(strict=True)
    except OSError as exc:
        raise InspectorIntegrityError(
            f"Allowlisted inspector {entry.id!r} is unavailable: {exc}"
        ) from exc
    if resolved != entry.executable or not resolved.is_file():
        raise InspectorIntegrityError(
            f"Allowlisted inspector {entry.id!r} no longer resolves to its approved file."
        )
    if entry.sha256 is not None and not _constant_digest_match(
        _sha256_file(resolved), entry.sha256
    ):
        raise InspectorIntegrityError(
            f"Allowlisted inspector {entry.id!r} does not match its SHA-256 pin."
        )
    for pin in entry.support_files:
        try:
            current = pin.path.resolve(strict=True)
        except OSError as exc:
            raise InspectorIntegrityError(
                f"Pinned inspector support file is unavailable: {exc}"
            ) from exc
        if current != pin.path or not current.is_file():
            raise InspectorIntegrityError(
                f"Pinned inspector support file no longer resolves to {pin.path}."
            )
        if not _constant_digest_match(_sha256_file(current), pin.sha256):
            raise InspectorIntegrityError(
                f"Pinned inspector support file does not match its SHA-256 pin: {pin.path}."
            )


def _constant_digest_match(observed: str, expected: str) -> bool:
    import hmac

    return hmac.compare_digest(observed, expected)


def _set_resource_limit(
    resource_module: Any,
    resource_kind: int,
    soft_limit: int,
    hard_limit: int,
) -> None:
    """Apply one POSIX limit without making an unsupported limit fatal."""
    try:
        _, current_hard = resource_module.getrlimit(resource_kind)
        infinity = resource_module.RLIM_INFINITY
        selected_hard = hard_limit if current_hard == infinity else min(hard_limit, current_hard)
        selected_soft = min(soft_limit, selected_hard)
        resource_module.setrlimit(resource_kind, (selected_soft, selected_hard))
    except (OSError, ValueError):
        # Darwin exposes RLIMIT_AS but rejects it for some hosted processes.
        # Parent-enforced wall-clock and output budgets remain active, while
        # every other supported POSIX limit is still applied independently.
        return


def _resource_limiter(limits: InspectorLimits) -> Any:
    if os.name != "posix":
        return None
    try:
        import resource
    except ImportError:  # pragma: no cover - resource exists on supported POSIX builds
        return None
    resource_module: Any = resource

    output_limit = max(limits.max_response_bytes, limits.max_stderr_bytes) + 4096

    def apply_limits() -> None:
        _set_resource_limit(
            resource_module,
            resource_module.RLIMIT_AS,
            limits.max_memory_bytes,
            limits.max_memory_bytes,
        )
        _set_resource_limit(
            resource_module,
            resource_module.RLIMIT_CPU,
            limits.max_cpu_seconds,
            limits.max_cpu_seconds + 1,
        )
        _set_resource_limit(
            resource_module,
            resource_module.RLIMIT_FSIZE,
            output_limit,
            output_limit,
        )
        if hasattr(resource_module, "RLIMIT_NOFILE"):
            _set_resource_limit(resource_module, resource_module.RLIMIT_NOFILE, 64, 64)

    return apply_limits


def _sanitized_environment() -> dict[str, str]:
    permitted = (
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
        "TMP",
        "TEMP",
        "TMPDIR",
        "LANG",
        "LC_ALL",
    )
    environment = {key: os.environ[key] for key in permitted if key in os.environ}
    environment.update(
        {
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    return environment


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            kill_process_group = cast(Any, os.__dict__["killpg"])
            kill_process_group(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=1.0)
    except (OSError, subprocess.TimeoutExpired):
        try:
            if os.name == "posix":
                kill_process_group = cast(Any, os.__dict__["killpg"])
                kill_process_group(process.pid, signal.__dict__["SIGKILL"])
            else:
                process.kill()
            process.wait(timeout=1.0)
        except (OSError, subprocess.TimeoutExpired):
            pass


def _run_bounded(command: tuple[str, ...], request: bytes, limits: InspectorLimits) -> bytes:
    with tempfile.TemporaryDirectory(prefix="researchplot-inspector-") as temporary:
        root = Path(temporary)
        input_path = root / "request.json"
        output_path = root / "response.json"
        error_path = root / "stderr.txt"
        input_path.write_bytes(request)
        creationflags = 0
        if os.name == "nt":
            creationflags = cast(int, getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            with (
                input_path.open("rb") as stdin,
                output_path.open("wb") as stdout,
                error_path.open("wb") as stderr,
            ):
                process = subprocess.Popen(
                    command,
                    stdin=stdin,
                    stdout=stdout,
                    stderr=stderr,
                    cwd=root,
                    env=_sanitized_environment(),
                    shell=False,
                    close_fds=True,
                    start_new_session=os.name == "posix",
                    preexec_fn=_resource_limiter(limits),
                    creationflags=creationflags,
                )
        except OSError as exc:
            raise InspectorExecutionError(f"Could not launch allowlisted inspector: {exc}") from exc

        started = time.monotonic()
        while process.poll() is None:
            elapsed = time.monotonic() - started
            if elapsed > limits.timeout_seconds:
                _terminate_process(process)
                raise InspectorTimeoutError(
                    f"Inspector exceeded the {limits.timeout_seconds:g}-second wall-clock budget."
                )
            try:
                output_size = output_path.stat().st_size
                error_size = error_path.stat().st_size
            except OSError as exc:
                _terminate_process(process)
                raise InspectorResourceError(f"Could not monitor inspector output: {exc}") from exc
            if output_size > limits.max_response_bytes:
                _terminate_process(process)
                raise InspectorResourceError(
                    f"Inspector stdout exceeded {limits.max_response_bytes} bytes."
                )
            if error_size > limits.max_stderr_bytes:
                _terminate_process(process)
                raise InspectorResourceError(
                    f"Inspector stderr exceeded {limits.max_stderr_bytes} bytes."
                )
            time.sleep(limits.poll_interval_seconds)

        try:
            output_size = output_path.stat().st_size
            error_size = error_path.stat().st_size
            response = output_path.read_bytes()
            stderr_data = error_path.read_bytes()
        except OSError as exc:
            raise InspectorExecutionError(f"Could not read inspector output: {exc}") from exc
        if output_size > limits.max_response_bytes:
            raise InspectorResourceError(
                f"Inspector stdout exceeded {limits.max_response_bytes} bytes."
            )
        if error_size > limits.max_stderr_bytes:
            raise InspectorResourceError(
                f"Inspector stderr exceeded {limits.max_stderr_bytes} bytes."
            )
        if process.returncode != 0:
            diagnostic = stderr_data.decode("utf-8", errors="replace").strip()
            if len(diagnostic) > 512:
                diagnostic = diagnostic[:509] + "..."
            suffix = f": {diagnostic}" if diagnostic else "."
            raise InspectorExecutionError(
                f"Inspector exited with status {process.returncode}{suffix}"
            )
        if stderr_data.strip():
            raise InspectorResponseError(
                "Inspector wrote to stderr despite returning a successful process status."
            )
        return response


def _request_id(method: str, params: dict[str, JsonValue]) -> str:
    payload = _encode_json({"method": method, "params": params})
    return hashlib.sha256(payload).hexdigest()[:32]


def _encode_json(value: JsonValue) -> bytes:
    _validate_json_value(value, label="Protocol data")
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise InspectorProtocolError(f"Protocol data is not canonical JSON: {exc}") from exc


def _duplicate_rejecting_object(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise InspectorResponseError(f"Inspector response repeats JSON key {key!r}.")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise InspectorResponseError(f"Inspector response contains non-finite number {value}.")


def _decode_json_object(raw: bytes) -> dict[str, JsonValue]:
    if not raw:
        raise InspectorResponseError("Inspector returned an empty response.")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise InspectorResponseError("Inspector response is not valid UTF-8.") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=_reject_constant,
        )
    except InspectorResponseError:
        raise
    except json.JSONDecodeError as exc:
        raise InspectorResponseError(f"Inspector response is not one JSON document: {exc}") from exc
    _validate_json_value(value, label="Inspector response", response=True)
    if not isinstance(value, dict):
        raise InspectorResponseError("Inspector response must be a JSON object.")
    return value


def _validate_json_value(
    value: object,
    *,
    label: str,
    response: bool = False,
    depth: int = 0,
    counter: list[int] | None = None,
) -> None:
    if counter is None:
        counter = [0]
    counter[0] += 1
    error_type = InspectorResponseError if response else InspectorProtocolError
    if counter[0] > _MAX_JSON_ITEMS:
        raise error_type(f"{label} exceeds the {_MAX_JSON_ITEMS}-item JSON limit.")
    if depth > _MAX_JSON_DEPTH:
        raise error_type(f"{label} exceeds the {_MAX_JSON_DEPTH}-level JSON depth limit.")
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise error_type(f"{label} contains a non-finite number.")
        return
    if isinstance(value, str):
        if len(value) > _MAX_JSON_STRING_LENGTH:
            raise error_type(
                f"{label} contains a string longer than {_MAX_JSON_STRING_LENGTH} characters."
            )
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_value(
                item,
                label=label,
                response=response,
                depth=depth + 1,
                counter=counter,
            )
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise error_type(f"{label} object keys must be strings.")
            if len(key) > _MAX_JSON_STRING_LENGTH:
                raise error_type(f"{label} contains an overlong object key.")
            _validate_json_value(
                item,
                label=label,
                response=response,
                depth=depth + 1,
                counter=counter,
            )
        return
    raise error_type(f"{label} contains unsupported value type {type(value).__name__}.")


def _require_exact_keys(
    value: Mapping[str, JsonValue],
    *,
    required: set[str],
    optional: set[str],
    label: str,
) -> None:
    keys = set(value)
    missing = sorted(required - keys)
    unknown = sorted(keys - required - optional)
    if missing:
        raise InspectorResponseError(f"{label} is missing keys: {', '.join(missing)}.")
    if unknown:
        raise InspectorResponseError(f"{label} has unknown keys: {', '.join(unknown)}.")


def _require_object(value: JsonValue, *, label: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise InspectorResponseError(f"{label} must be an object.")
    return value


def _require_string(
    value: JsonValue,
    *,
    label: str,
    maximum: int = _MAX_JSON_STRING_LENGTH,
) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise InspectorResponseError(
            f"{label} must be a non-empty string of at most {maximum} characters."
        )
    return value


def _require_string_array(value: JsonValue, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise InspectorResponseError(f"{label} must be a non-empty array of strings.")
    result = tuple(
        _require_string(item, label=f"{label} item", maximum=_MAX_IDENTIFIER_LENGTH)
        for item in value
    )
    if len(result) != len(set(result)):
        raise InspectorResponseError(f"{label} must not contain duplicates.")
    return result


def _raise_remote_error(value: JsonValue) -> None:
    error = _require_object(value, label="Protocol error")
    _require_exact_keys(error, required={"code", "message"}, optional=set(), label="Protocol error")
    code = _require_string(error["code"], label="Protocol error code", maximum=128)
    try:
        _validate_identifier(code, label="Protocol error code")
    except ValueError as exc:
        raise InspectorResponseError(str(exc)) from exc
    message = _require_string(error["message"], label="Protocol error message", maximum=4096)
    raise InspectorRemoteError(code, message)


def _parse_capabilities(
    result: dict[str, JsonValue], *, expected_inspector_id: str
) -> InspectorCapabilities:
    _require_exact_keys(
        result,
        required={"inspector", "capabilities"},
        optional=set(),
        label="Capabilities result",
    )
    identity = _require_object(result["inspector"], label="Inspector identity")
    _require_exact_keys(
        identity,
        required={"id", "version"},
        optional=set(),
        label="Inspector identity",
    )
    inspector_id = _require_string(identity["id"], label="Inspector identity ID", maximum=128)
    try:
        _validate_identifier(inspector_id, label="Inspector identity ID")
    except ValueError as exc:
        raise InspectorResponseError(str(exc)) from exc
    if inspector_id != expected_inspector_id:
        raise InspectorResponseError(
            f"Inspector identified itself as {inspector_id!r}, expected {expected_inspector_id!r}."
        )
    version = _require_string(
        identity["version"], label="Inspector version", maximum=_MAX_VERSION_LENGTH
    )
    raw_capabilities = result["capabilities"]
    if not isinstance(raw_capabilities, list):
        raise InspectorResponseError("Capabilities must be an array.")
    capabilities: list[InspectorCapability] = []
    for index, raw_capability in enumerate(raw_capabilities):
        item = _require_object(raw_capability, label=f"Capability {index}")
        _require_exact_keys(
            item,
            required={"id", "formats", "phases", "probes", "confidence"},
            optional=set(),
            label=f"Capability {index}",
        )
        identifier = _require_string(item["id"], label=f"Capability {index} ID", maximum=128)
        try:
            _validate_identifier(identifier, label=f"Capability {index} ID")
        except ValueError as exc:
            raise InspectorResponseError(str(exc)) from exc
        formats = tuple(
            value.casefold()
            for value in _require_string_array(item["formats"], label=f"Capability {index} formats")
        )
        if any(not _FORMAT.fullmatch(value) for value in formats):
            raise InspectorResponseError(
                f"Capability {identifier!r} contains an invalid artifact format."
            )
        phase_values = _require_string_array(item["phases"], label=f"Capability {index} phases")
        try:
            phases = tuple(EvidencePhase(value) for value in phase_values)
        except ValueError as exc:
            choices = ", ".join(value.value for value in EvidencePhase)
            raise InspectorResponseError(
                f"Capability {identifier!r} has an invalid phase; choose from: {choices}."
            ) from exc
        probes = _require_string_array(item["probes"], label=f"Capability {index} probes")
        if any(not _IDENTIFIER.fullmatch(probe) for probe in probes):
            raise InspectorResponseError(
                f"Capability {identifier!r} contains an invalid probe identifier."
            )
        confidence_text = _require_string(
            item["confidence"], label=f"Capability {index} confidence", maximum=32
        )
        try:
            confidence = EvidenceConfidence(confidence_text)
        except ValueError as exc:
            choices = ", ".join(value.value for value in EvidenceConfidence)
            raise InspectorResponseError(
                f"Capability {identifier!r} has invalid confidence; choose from: {choices}."
            ) from exc
        capabilities.append(InspectorCapability(identifier, formats, phases, probes, confidence))
    identifiers = [item.id for item in capabilities]
    if len(identifiers) != len(set(identifiers)):
        raise InspectorResponseError("Inspector capability IDs must be unique.")
    return InspectorCapabilities(inspector_id, version, tuple(capabilities))


def _parse_inspection_result(
    result: dict[str, JsonValue],
    *,
    inspector_id: str,
    capability: InspectorCapability,
    file_format: str,
) -> ObservationSet:
    _require_exact_keys(
        result,
        required={"capability", "observations"},
        optional=set(),
        label="Inspection result",
    )
    returned_capability = _require_string(
        result["capability"], label="Inspection result capability", maximum=128
    )
    if returned_capability != capability.id:
        raise InspectorResponseError("Inspection result capability does not match the request.")
    raw_observations = result["observations"]
    if not isinstance(raw_observations, list):
        raise InspectorResponseError("Inspection observations must be an array.")
    observations: list[Observation] = []
    for index, raw_observation in enumerate(raw_observations):
        item = _require_object(raw_observation, label=f"Observation {index}")
        _require_exact_keys(
            item,
            required={"probe", "value", "available", "phase", "detail", "unit"},
            optional=set(),
            label=f"Observation {index}",
        )
        probe = _require_string(item["probe"], label=f"Observation {index} probe", maximum=128)
        if probe not in capability.probes:
            raise InspectorResponseError(
                f"Observation probe {probe!r} was not declared by capability {capability.id!r}."
            )
        if not isinstance(item["available"], bool):
            raise InspectorResponseError(f"Observation {index} available must be a boolean.")
        available = item["available"]
        detail_value = item["detail"]
        if detail_value is not None and not isinstance(detail_value, str):
            raise InspectorResponseError(f"Observation {index} detail must be a string or null.")
        if isinstance(detail_value, str) and len(detail_value) > 4096:
            raise InspectorResponseError(f"Observation {index} detail exceeds 4096 characters.")
        if not available and (item["value"] is not None or not detail_value):
            raise InspectorResponseError(
                f"Unavailable observation {index} must have a null value and explanatory detail."
            )
        phase_text = _require_string(item["phase"], label=f"Observation {index} phase", maximum=32)
        try:
            phase = EvidencePhase(phase_text)
        except ValueError as exc:
            raise InspectorResponseError(
                f"Observation {index} has an invalid evidence phase."
            ) from exc
        if phase not in capability.phases:
            raise InspectorResponseError(
                f"Observation {index} uses phase {phase.value!r}, which the capability did not declare."
            )
        unit_value = item["unit"]
        if unit_value is not None and not isinstance(unit_value, str):
            raise InspectorResponseError(f"Observation {index} unit must be a string or null.")
        if isinstance(unit_value, str) and (not unit_value or len(unit_value) > 64):
            raise InspectorResponseError(f"Observation {index} unit is invalid.")
        try:
            observation = Observation(
                probe=probe,
                value=item["value"],
                available=available,
                phase=phase.value,
                detail=detail_value,
                unit=unit_value,
                confidence=capability.confidence,
                producer=f"external:{inspector_id}",
                supported_phases=capability.phases,
                supported_formats=capability.formats,
            )
        except ValueError as exc:
            raise InspectorResponseError(
                f"Observation {index} does not satisfy the typed observation contract: {exc}"
            ) from exc
        if "*" not in capability.formats and file_format not in observation.supported_formats:
            raise InspectorResponseError(
                f"Observation {index} is incompatible with artifact format {file_format!r}."
            )
        observations.append(observation)
    try:
        return ObservationSet(observations)
    except ValueError as exc:
        raise InspectorResponseError(str(exc)) from exc


__all__ = [
    "PROTOCOL_VERSION",
    "ExternalInspectionResult",
    "InspectorAllowlist",
    "InspectorAllowlistError",
    "InspectorCapabilities",
    "InspectorCapability",
    "InspectorDiagnostic",
    "InspectorExecutable",
    "InspectorExecutionError",
    "InspectorFilePin",
    "InspectorIntegrityError",
    "InspectorLimits",
    "InspectorProtocolClient",
    "InspectorProtocolError",
    "InspectorRemoteError",
    "InspectorResourceError",
    "InspectorResponseError",
    "InspectorTimeoutError",
]
