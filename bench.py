"""Pure decode benchmark — no WebSocket, no plugin, no UI.
Measures what a parakeet streaming model costs on THIS cpu, sweeping thread
counts (1/2/4/all) to find the fastest config.

Run:  .venv/bin/python bench.py [model-dir]
      (default ./model; try ./model-560, ./model-1120)
Reads: the config with the lowest RTF. <1.0 = streaming viable.
"""
import glob, os, sys, time, wave
import numpy as np
import sherpa_onnx

MODEL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

def find(role):
    import glob as g
    i8 = sorted(g.glob(os.path.join(MODEL, f"*{role}*.int8.onnx")))
    fp = [p for p in sorted(g.glob(os.path.join(MODEL, f"*{role}*.onnx"))) if not p.endswith(".int8.onnx")]
    pick = i8 or fp
    return pick[0] if pick else None

enc, dec, joi = find("encoder"), find("decoder"), find("joiner")
tok = sorted(__import__("glob").glob(os.path.join(MODEL, "tokens*.txt")))
tok = tok[0] if tok else None
for p, r in ((enc, "encoder"), (dec, "decoder"), (joi, "joiner"), (tok, "tokens")):
    if not p:
        raise SystemExit(f"missing {r} in {MODEL}")

wavs = sorted(glob.glob(os.path.join(MODEL, "test_wavs", "*.wav")))
if wavs:
    w = wave.open(wavs[0])
    audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
else:
    # no test wavs (old extracts excluded them): synthesize speech-level noise —
    # decode cost is duration-driven, so RTF is measured just the same
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(16000 * 5) * 0.05).astype(np.float32)
    print("(no test_wavs — using 5s synthetic audio; RTF measurement unaffected)")
audio = audio[:16000 * 5]                       # first 5s is plenty
dur = len(audio) / 16000

# detect dialect SAFELY: streaming encoders carry chunking metadata
# (chunk_encoder_frames / buffered_streaming); offline ones don't. Probing with
# the streaming loader hard-exits the process on some sherpa builds.
def _is_streaming_model(encoder_path):
    try:
        import onnxruntime as _ort
        meta = _ort.InferenceSession(encoder_path, providers=["CPUExecutionProvider"]).get_modelmeta()
        m = meta.custom_metadata_map or {}
        return "chunk_encoder_frames" in m or "buffered_streaming" in m
    except Exception:
        return True   # can't inspect: assume streaming (previous behavior)

IS_OFFLINE = not _is_streaming_model(enc)
if IS_OFFLINE:
    print("(offline model — using OfflineRecognizer)")

ALL = os.cpu_count() or 4
best = (None, 1e9)
for threads in sorted(set(t for t in (1, 2, 4, ALL) if t <= ALL)):
    t0 = time.time()
    if IS_OFFLINE:
        rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=enc, decoder=dec, joiner=joi, tokens=tok, num_threads=threads,
            model_type="nemo_transducer")
    else:
        rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=enc, decoder=dec, joiner=joi, tokens=tok, num_threads=threads)
    load = time.time() - t0
    if IS_OFFLINE:
        t0 = time.time()
        stream = rec.create_stream()
        stream.accept_waveform(16000, audio)
        rec.decode_stream(stream)
        wall = time.time() - t0
        sample = stream.result.text
    else:
        stream = rec.create_stream()
        t0 = time.time()
        for i in range(0, len(audio), 1280):    # 80ms chunks, live-path cadence
            stream.accept_waveform(16000, audio[i:i+1280])
            while rec.is_ready(stream):
                rec.decode_stream(stream)
        wall = time.time() - t0
        sample = rec.get_result(stream)
    rtf = wall / dur
    tag = ""
    if rtf < best[1]:
        best = (threads, rtf); tag = "  <-- best"
    print(f"threads={threads}: load {load:.1f}s | decode {wall:.1f}s for {dur:.1f}s audio | RTF {rtf:.2f}{tag}")
print(f"\nRESULT: {os.path.basename(MODEL)} best RTF {best[1]:.2f} at threads={best[0]}")
print("final text sample:", repr(sample)[:100])
