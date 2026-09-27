# hermes-dictate

Local-first, WisprFlow-style voice dictation for the **Hermes Desktop** app.
Speak → text appears live in the composer as a draft → it NEVER sends. Enter is
always yours.

- **100% on-device ASR** (sherpa-onnx + NeMo parakeet / silero VAD) — no audio
  ever leaves your machine
- Live draft in the composer with per-word streaming (hybrid mode) or clean
  punctuated finals on each pause (offline-only mode)
- Deterministic cleanup: removes um/uh/erm and stutters, keeps every real word
  you said (including "So", "Well", "you know")
- Your edits always win: delete words mid-dictation and they stay deleted
- ~0.5s end-of-utterance cut; adaptive noise-floor gating + silero VAD

## Requirements

- Linux (systemd user services) or macOS, Python 3.10+
- Hermes Desktop app
- ~250MB model download, ~1GB RAM (offline mode) / ~1.6GB (hybrid)

## Install

From this repo (or after installing it as a desktop plugin via the app's
plugin installer):

```bash
cd ~/.hermes/desktop-plugins/hermes-dictate

# 1. python env (PEP 668-safe)
python3 -m venv .venv
.venv/bin/pip install sherpa-onnx websockets numpy onnxruntime

# 2. model (offline parakeet — the default, best-quality mode)
VARIANT=parakeet-off ./setup-model.sh

# 3. autostart the sidecar at login (recommended)
mkdir -p ~/.config/systemd/user
cp hermes-dictate.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now hermes-dictate
```

Then open Hermes Desktop, press the mic chip (or `Ctrl/Cmd+Shift+D`) and speak.

No systemd? Run the sidecar manually instead (step 3):

```bash
.venv/bin/python sidecar.py --model-dir=model-parakeet-off
```

## Modes

| Mode | Flags | Behavior | RAM |
|---|---|---|---|
| **offline-only** (default, recommended) | `--model-dir=model-parakeet-off` | Punctuated parakeet-quality final ~0.5-1s after each pause. No live words. | ~1GB |
| hybrid | `--model-dir=model-560 --final-model-dir=model-parakeet-off` | Live draft while speaking; parakeet final replaces it on pause. Needs a strong CPU (RTF < 1) — run `bench.py model-560` first. | ~1.8GB |

If `bench.py` shows RTF > 1 for your CPU, stay on offline-only — it's the quality mode anyway.

Other variants (240/560/1120ms streaming parakeet) exist but need a strong CPU;
run `bench.py <model-dir>` to check RTF (must be < 1 for streaming modes).

## Model selection

`setup-model.sh VARIANT=` : `parakeet-off` (default) | `560` | `240` | `1120`

(The lightweight zipformer streaming model also exists — `VARIANT=zip` — but its
accuracy is far below parakeet; not recommended.)

Edit `~/.config/systemd/user/hermes-dictate.service` → `systemctl --user restart hermes-dictate`.

## Tuning

Sidecar flags: `--vad-threshold` (0.5; lower = more sensitive), `--threads`,
`--port` (8765), `--polish` (LLM rewrite of long finals via the Hermes provider
env; off by default, regex-only fallback).

## Troubleshooting

- **"backend supervisor did not start it"** — the sidecar isn't running and
  your desktop's backend is remote. `systemctl --user status hermes-dictate`.
- **Nothing transcribes** — check the mic is captured by the app (chip shows a
  level %), then `journalctl --user -u hermes-dictate -f` while speaking; you
  should see `noise floor calibrated` then `decode batch` lines.
- **Remote-backend desktops**: the sidecar MUST run on the machine with the
  mic (the plugin probes `ws://127.0.0.1:8765` locally first; a remote
  supervisor can't help and is skipped).

## Files

- `plugin.js` — desktop plugin (mic capture, live draft, chip UI, keybind)
- `sidecar.py` — local ASR sidecar (WS on 127.0.0.1:8765)
- `setup-model.sh` — model downloader
- `bench.py` — decode-speed benchmark (thread sweep)
- `hermes-dictate.service` — systemd user unit (offline-only default)

## Optional: backend supervisor half

If your Hermes backend runs **locally**, you can also install the
`dashboard/plugin_api.py` supervisor (mounts `/api/plugins/hermes-dictate/*`,
lets the backend spawn the sidecar for you). Not needed with the systemd unit;
skip it on remote-backend setups. Requires `plugins.enabled: [hermes-dictate]`
in config.yaml. Copy `dashboard/` to `~/.hermes/plugins/hermes-dictate/dashboard/`.

## License

MIT
