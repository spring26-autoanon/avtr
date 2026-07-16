"""
Smoke test for demo/server.py.

Starts the FastAPI app in-process (via TestClient — no real socket, no
subprocess), connects a WebSocket client, sends one short raw-PCM audio
chunk, and confirms audio comes back. Uses the "adapter: stub" config escape
hatch (same one evals/runner.py's own tests use) so this runs without a GPU
or a real MoshiRAG checkpoint. No sounddevice anywhere — the client here is
Starlette's WebSocket test client, not a real microphone/speaker.
"""

from pathlib import Path

import pytest
from starlette.testclient import TestClient, WebSocketDenialResponse

import demo.server as demo_server


def _write_stub_config(tmp_path: Path) -> str:
    cfg = tmp_path / "stub.yaml"
    cfg.write_text(
        "model:\n"
        "  checkpoint: stub\n"
        "  adapter: stub\n"
        "  retrieval:\n"
        "    enabled: false\n"
        "evals: []\n"
        "output_dir: ./evals/results/\n"
    )
    return str(cfg)


def test_websocket_roundtrip_returns_audio(tmp_path):
    demo_server._session_active = False
    config_path = _write_stub_config(tmp_path)
    demo_server._configure_app(config_path)

    with TestClient(demo_server.app) as client:
        with client.websocket_connect(f"/ws?config={config_path}") as ws:
            ws.send_bytes(b"\x00\x01" * 100)  # one short fake PCM chunk
            received = ws.receive_bytes()

    assert isinstance(received, bytes)
    assert len(received) > 0


def test_second_connection_rejected_with_503(tmp_path):
    demo_server._session_active = False
    config_path = _write_stub_config(tmp_path)
    demo_server._configure_app(config_path)

    with TestClient(demo_server.app) as client:
        with client.websocket_connect(f"/ws?config={config_path}") as ws1:
            ws1.send_bytes(b"\x00\x01" * 100)
            ws1.receive_bytes()

            with pytest.raises(WebSocketDenialResponse) as exc_info:
                with client.websocket_connect(f"/ws?config={config_path}"):
                    pass
            assert exc_info.value.status_code == 503
