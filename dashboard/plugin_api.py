"""hermes-dictate supervisor: backend plugin API mounted at /api/plugins/hermes-dictate/.

One job: make "python3 sidecar.py" disappear from the user's workflow. The desktop
plugin calls POST /start before connecting its WebSocket; this router spawns the
sidecar subprocess (using the plugin's own .venv if present), waits for its port
to answer, and reports status. The child is started detached-from-parent but in
the same session; it stays up for the app's lifetime and is reused across
dictation toggles. GET /status reports whether the port answers; POST /stop kills
a known child (leaves foreign sidecars alone).

No config, no persistence — the sidecar itself is stateless between utterances.
"""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

_PLUGIN_DIR = Path(__file__).resolve().parent.parent        # ~/.hermes/plugins/hermes-dictate/
# The sidecar + models live with the DESKTOP plugin (desktop-plugins/hermes-dictate),
# which is where the user installed them; support both layouts.
_CANDIDATE_DIRS = [
    _PLUGIN_DIR,
    _PLUGIN_DIR.parent.parent / "desktop-plugins" / "hermes-dictate",
]
_PORT = 8765
_START_TIMEOUT_S = 120.0        # model load can take ~30-60s cold
_POLL_S = 0.5

_child: subprocess.Popen | None = None
_child_dir: str | None = None


def _port_open(port: int = _PORT) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.4):
            return True
    except OSError:
        return False


def _auto_model_dir(d: Path) -> str | None:
    """Best available model next to the sidecar, in preference order:
    zip hybrid (zip + offline parakeet) > offline-only parakeet > zip alone >
    whatever 'model' holds. Falls back to the sidecar default."""
    has = lambda n: (d / n / "encoder.int8.onnx").exists() or (d / n).is_dir() and any((d / n).glob("encoder*.onnx"))
    if has("model-zip") and has("model-parakeet-off"):
        return "model-zip"          # hybrid: streaming zip + offline final via env below
    if has("model-parakeet-off"):
        return "model-parakeet-off"
    if has("model-zip"):
        return "model-zip"
    return None


def _sidecar_dir() -> tuple[Path | None, str]:
    """Locate the sidecar directory + python to run it with."""
    for d in _CANDIDATE_DIRS:
        if (d / "sidecar.py").exists():
            # prefer the plugin's own venv, then ANY python that can import sherpa_onnx
            venv = d / ".venv" / "bin" / "python"
            if venv.exists():
                return d, str(venv)
            for py in ("python3", sys.executable):
                if py and _has_sherpa(py):
                    return d, py
            return d, "python3"
    return None, "python3"


def _has_sherpa(py: str) -> bool:
    try:
        return subprocess.run([py, "-c", "import sherpa_onnx"], capture_output=True, timeout=30).returncode == 0
    except Exception:
        return False


class StartReq(BaseModel):
    model_dir: str | None = None
    final_model_dir: str | None = None
    threads: int = 0
    port: int = _PORT


@router.get("/status")
def status() -> dict:
    d, py = _sidecar_dir()
    return {
        "running": _port_open(),
        "port": _PORT,
        "sidecar_dir": str(d) if d else None,
        "python": py,
        "child_pid": _child.pid if _child and _child.poll() is None else None,
    }


@router.post("/start")
def start(req: StartReq) -> dict:
    global _child, _child_dir
    if _port_open():
        return {"ok": True, "already_running": True}
    d, py = _sidecar_dir()
    if d is None:
        return {"ok": False, "error": "sidecar.py not found in ~/.hermes/plugins/hermes-dictate or ~/.hermes/desktop-plugins/hermes-dictate"}
    model_dir = req.model_dir or os.environ.get("DICTATE_MODEL_DIR") or _auto_model_dir(d)
    final_model_dir = req.final_model_dir or os.environ.get("DICTATE_FINAL_MODEL_DIR")
    cmd = [py, "-u", str(d / "sidecar.py"), "--port", str(req.port)]
    if model_dir:
        cmd += ["--model-dir", model_dir]
    if final_model_dir:
        cmd += ["--final-model-dir", final_model_dir]
    elif model_dir == "model-zip" and (d / "model-parakeet-off").is_dir():
        cmd += ["--final-model-dir", "model-parakeet-off"]
    if req.threads:
        cmd += ["--threads", str(req.threads)]
    log = open(d / "sidecar-supervisor.log", "ab")
    try:
        _child = subprocess.Popen(
            cmd, cwd=str(d), stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,   # survives backend restarts; killed via /stop
        )
    except Exception as e:
        return {"ok": False, "error": f"spawn failed: {e}"}
    _child_dir = str(d)
    t0 = time.time()
    while time.time() - t0 < _START_TIMEOUT_S:
        if _port_open(req.port):
            return {"ok": True, "already_running": False, "pid": _child.pid}
        if _child.poll() is not None:
            tail = ""
            try:
                tail = (d / "sidecar-supervisor.log").read_text(encoding="utf-8", errors="replace")[-400:]
            except OSError:
                pass
            return {"ok": False, "error": f"sidecar exited rc={_child.returncode}", "log_tail": tail}
        time.sleep(_POLL_S)
    return {"ok": False, "error": f"sidecar did not listen on {req.port} within {_START_TIMEOUT_S:.0f}s (model load too slow?)"}


@router.post("/stop")
def stop() -> dict:
    global _child
    killed = False
    if _child and _child.poll() is None:
        try:
            os.killpg(os.getpgid(_child.pid), signal.SIGTERM)
            killed = True
        except (ProcessLookupError, PermissionError):
            pass
        _child = None
    return {"ok": True, "killed": killed, "port_open": _port_open()}
