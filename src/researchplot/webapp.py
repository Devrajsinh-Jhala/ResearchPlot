"""Privacy-preserving local browser workspace for ResearchPlot.

The server deliberately uses only the standard library and binds to the IPv4
loopback interface.  Browser uploads are written to a private temporary
directory, inspected by the same engine as the Python and CLI APIs, and removed
before the response is returned.
"""

from __future__ import annotations

import json
import mimetypes
import os
import secrets
import shlex
import subprocess
import tempfile
import threading
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import cast
from urllib.parse import parse_qs, quote, urlsplit

from .registry import list_profiles
from .target import target

_MAX_UPLOAD_BYTES = 128 * 1024 * 1024
_SUPPORTED_SUFFIXES = {".eps", ".jpeg", ".jpg", ".pdf", ".png", ".svg", ".tif", ".tiff"}


def _command(argv: list[str]) -> str:
    return subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)


class LocalWebError(RuntimeError):
    """Raised when the local preflight server cannot be started safely."""


@dataclass(frozen=True, slots=True)
class ServerInfo:
    """Connection details for a running local workspace."""

    url: str
    host: str
    port: int
    token: str


class _WorkspaceServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, address: tuple[str, int], token: str) -> None:
        super().__init__(address, _WorkspaceHandler)
        self.token = token


class _WorkspaceHandler(BaseHTTPRequestHandler):
    server_version = "ResearchPlotLocal/2"

    @property
    def workspace_server(self) -> _WorkspaceServer:
        return cast(_WorkspaceServer, self.server)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        # Avoid printing artifact names or access tokens to shared terminal logs.
        return

    def _query(self) -> dict[str, list[str]]:
        return parse_qs(urlsplit(self.path).query, keep_blank_values=True)

    def _authorized(self) -> bool:
        supplied = self.headers.get("X-ResearchPlot-Token")
        if supplied is None:
            supplied = self._query().get("token", [""])[0]
        return secrets.compare_digest(supplied, self.workspace_server.token)

    def _origin_allowed(self) -> bool:
        origin = self.headers.get("Origin")
        if origin is None:
            return True
        address = cast(tuple[str, int], self.workspace_server.server_address)
        host, port = address
        return origin in {f"http://{host}:{port}", f"http://localhost:{port}"}

    def _headers(self, status: HTTPStatus, content_type: str, length: int) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' blob: data:; style-src 'self'; "
            "script-src 'self'; connect-src 'self'; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'",
        )
        self.end_headers()

    def _send_bytes(
        self, payload: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK
    ) -> None:
        self._headers(status, content_type, len(payload))
        self.wfile.write(payload)

    def _send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8", status)

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message, "status": status.value}, status)

    def _asset(self, name: str) -> None:
        if name not in {"index.html", "app.js", "styles.css"}:
            self._error(HTTPStatus.NOT_FOUND, "Asset not found.")
            return
        resource = files("researchplot").joinpath("web_assets", name)
        try:
            body = resource.read_bytes()
        except FileNotFoundError:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "Web assets are missing.")
            return
        content_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
        if name.endswith((".html", ".js", ".css")):
            content_type += "; charset=utf-8"
        self._send_bytes(body, content_type)

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path == "/":
            if not self._authorized():
                self._error(HTTPStatus.UNAUTHORIZED, "A valid local session token is required.")
                return
            self._asset("index.html")
            return
        if path in {"/app.js", "/styles.css"}:
            self._asset(path.removeprefix("/"))
            return
        if not self._authorized():
            self._error(HTTPStatus.UNAUTHORIZED, "A valid local session token is required.")
            return
        if path == "/api/health":
            self._send_json({"status": "ok", "network": "loopback-only"})
            return
        if path == "/api/profiles":
            self._send_json(
                {
                    "profiles": [
                        {
                            "id": profile.id,
                            "coordinate": profile.coordinate,
                            "name": profile.name,
                            "kind": profile.kind.value,
                            "year": profile.year,
                            "widths": list(profile.width_options),
                            "default_width": profile.default_width,
                            "verified_on": profile.verified_on,
                            "caveats": list(profile.caveats),
                            "sources": [source.to_dict() for source in profile.sources],
                        }
                        for profile in list_profiles()
                    ]
                }
            )
            return
        self._error(HTTPStatus.NOT_FOUND, "Endpoint not found.")

    def do_POST(self) -> None:  # noqa: N802
        if not self._authorized():
            self._error(HTTPStatus.UNAUTHORIZED, "A valid local session token is required.")
            return
        if not self._origin_allowed():
            self._error(HTTPStatus.FORBIDDEN, "Cross-origin requests are not permitted.")
            return
        if urlsplit(self.path).path != "/api/audit":
            self._error(HTTPStatus.NOT_FOUND, "Endpoint not found.")
            return
        raw_length = self.headers.get("Content-Length")
        try:
            length = int(raw_length or "")
        except ValueError:
            self._error(HTTPStatus.LENGTH_REQUIRED, "A valid Content-Length is required.")
            return
        if length <= 0 or length > _MAX_UPLOAD_BYTES:
            self._error(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                f"Artifacts must be between 1 byte and {_MAX_UPLOAD_BYTES} bytes.",
            )
            return
        query = self._query()
        profile = query.get("profile", [""])[0]
        filename = Path(query.get("filename", ["artifact"])[0]).name
        suffix = Path(filename).suffix.casefold()
        if not profile:
            self._error(HTTPStatus.BAD_REQUEST, "A profile coordinate is required.")
            return
        if suffix not in _SUPPORTED_SUFFIXES:
            self._error(HTTPStatus.BAD_REQUEST, f"Unsupported artifact extension {suffix!r}.")
            return
        width = query.get("width", [""])[0] or None
        role = query.get("role", ["main"])[0]
        content = query.get("content", ["data-visualization"])[0]
        body = self.rfile.read(length)
        if len(body) != length:
            self._error(HTTPStatus.BAD_REQUEST, "The artifact upload was truncated.")
            return
        try:
            with tempfile.TemporaryDirectory(prefix="researchplot-web-") as directory:
                artifact = Path(directory) / f"artifact{suffix}"
                artifact.write_bytes(body)
                report = target(profile, role=role, width=width, content=content).audit(artifact)
            report_payload = report.to_dict()
            findings = report_payload.get("findings")
            if isinstance(findings, list):
                for finding in findings:
                    if isinstance(finding, dict) and finding.get("artifact") is not None:
                        finding["artifact"] = filename
            command_argv = [
                "researchplot",
                "audit",
                filename,
                "--profile",
                profile,
            ]
            if width:
                command_argv.extend(("--width", width))
            command_argv.extend(("--role", role, "--content", content))
            self._send_json(
                {
                    "filename": filename,
                    "bytes": length,
                    "report": report_payload,
                    "command": _command(command_argv),
                    "command_argv": command_argv,
                }
            )
        except (OSError, RuntimeError, ValueError) as exc:
            self._error(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))


def create_server(*, host: str = "127.0.0.1", port: int = 0) -> tuple[_WorkspaceServer, ServerInfo]:
    """Create a loopback-only local workspace server without starting it."""

    if host != "127.0.0.1":
        raise LocalWebError("ResearchPlot 2.0 web workspaces bind only to 127.0.0.1.")
    if not isinstance(port, int) or isinstance(port, bool) or not 0 <= port <= 65535:
        raise LocalWebError("Web workspace port must be an integer from 0 to 65535.")
    token = secrets.token_urlsafe(32)
    server = _WorkspaceServer((host, port), token)
    selected_port = int(server.server_address[1])
    url = f"http://{host}:{selected_port}/?token={quote(token)}"
    return server, ServerInfo(url=url, host=host, port=selected_port, token=token)


def serve(*, port: int = 0, open_browser: bool = True) -> ServerInfo:
    """Run the local preflight workspace until interrupted.

    The returned :class:`ServerInfo` is mainly useful to callers that stop the
    server from another thread.  CLI use blocks until Ctrl+C.
    """

    server, info = create_server(port=port)
    if open_browser:
        threading.Timer(0.25, webbrowser.open, args=(info.url,)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
    return info


__all__ = ["LocalWebError", "ServerInfo", "create_server", "serve"]
