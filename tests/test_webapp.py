from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from io import BytesIO

import pytest
from PIL import Image

from researchplot.webapp import LocalWebError, create_server


def _request(url: str, token: str, *, data: bytes | None = None) -> urllib.request.Request:
    return urllib.request.Request(
        url,
        data=data,
        headers={
            "X-ResearchPlot-Token": token,
            "Content-Type": "application/octet-stream",
        },
        method="POST" if data is not None else "GET",
    )


def test_local_server_rejects_remote_binding() -> None:
    with pytest.raises(LocalWebError, match="127.0.0.1"):
        create_server(host="0.0.0.0")


def test_local_server_requires_token_and_lists_profiles() -> None:
    server, info = create_server()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(urllib.error.HTTPError) as denied:
            urllib.request.urlopen(f"http://{info.host}:{info.port}/api/profiles")
        assert denied.value.code == 401

        request = _request(f"http://{info.host}:{info.port}/api/profiles", info.token)
        with urllib.request.urlopen(request) as response:
            payload = json.load(response)
        assert any(profile["id"] == "nature" for profile in payload["profiles"])
        assert response.headers["Cache-Control"] == "no-store"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_local_server_audits_uploaded_artifact() -> None:
    image = Image.new("RGB", (100, 100), "white")
    buffer = BytesIO()
    image.save(buffer, format="PNG", dpi=(300, 300))
    server, info = create_server()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = (
            f"http://{info.host}:{info.port}/api/audit?"
            "profile=nature%402026.08.0&width=single&role=main&"
            "content=data-visualization&filename=figure.png"
        )
        request = _request(url, info.token, data=buffer.getvalue())
        with urllib.request.urlopen(request) as response:
            payload = json.load(response)
        assert payload["filename"] == "figure.png"
        assert payload["command_argv"][:3] == ["researchplot", "audit", "figure.png"]
        assert payload["report"]["profile"] == "nature@2026.08.0"
        assert "researchplot-web-" not in json.dumps(payload)
        assert payload["report"]["verdict"] in {
            "compliant",
            "non_compliant",
            "indeterminate",
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
