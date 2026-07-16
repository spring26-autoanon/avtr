#!/usr/bin/env python3
"""
MoshiRAG browser demo server.

Usage:
  uv run demo/server.py --config configs/baseline_with_retrieval.yaml

Access from any browser via an SSH tunnel:
  ssh -L 8998:localhost:8998 user@vertex-vm
  open http://localhost:8998
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles

from core.config import load_config
from core.retrieval_backend import GeminiAPIBackend, NullBackend
from evals.runner import build_model

CLIENT_DIR = Path(__file__).parent / "client"

app = FastAPI()
_session_active = False


def _build_retrieval_backend(cfg: dict):
    retrieval_cfg = cfg.get("model", {}).get("retrieval", {})
    if retrieval_cfg.get("enabled"):
        return GeminiAPIBackend(
            model=retrieval_cfg.get("model", "gemini-3.5-flash"),
            latency_gate_ms=retrieval_cfg["latency_gate_ms"],
        )
    return NullBackend()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    global _session_active

    if _session_active:
        await websocket.send_denial_response(
            PlainTextResponse("one active session at a time", status_code=503)
        )
        return

    await websocket.accept()
    _session_active = True

    try:
        # Retrieval on/off is config-driven: the client toggles by reconnecting
        # with a different ?config= value. Swapping the retrieval backend on
        # the already-loaded model is enough — no model reload needed.
        config_path = websocket.query_params.get("config", app.state.default_config_path)
        if config_path != app.state.current_config_path:
            cfg = load_config(config_path)
            app.state.model.retrieval_backend = _build_retrieval_backend(cfg)
            app.state.current_config_path = config_path

        async def audio_chunks():
            try:
                while True:
                    yield await websocket.receive_bytes()
            except WebSocketDisconnect:
                return

        async for audio_out, _text_out, _metadata in app.state.model.respond_stream(audio_chunks()):
            await websocket.send_bytes(audio_out)
    except WebSocketDisconnect:
        pass
    finally:
        _session_active = False


app.mount("/", StaticFiles(directory=str(CLIENT_DIR), html=True), name="client")


def _configure_app(config_path: str) -> None:
    cfg = load_config(config_path)
    app.state.model = build_model(cfg)
    app.state.default_config_path = config_path
    app.state.current_config_path = config_path


def main() -> None:
    parser = argparse.ArgumentParser(description="MoshiRAG browser demo server")
    parser.add_argument("--config", required=True, help="YAML config path")
    args = parser.parse_args()

    # Without this, core/model_interface.py's logger.info() calls (checkpoint,
    # ARC-Encoder, warmup progress) are silently discarded — Python's logging
    # module does nothing until configured, so loading looked like a hang.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    print(f"Loading model from {args.config} — checkpoint + ARC-Encoder + warmup, can take a few minutes...")
    _configure_app(args.config)
    print("Model ready. Starting server...")
    uvicorn.run(app, host="127.0.0.1", port=8998)


if __name__ == "__main__":
    main()
