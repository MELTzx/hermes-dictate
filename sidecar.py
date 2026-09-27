#!/usr/bin/env python3
"""hermes-dictate sidecar: local-first streaming dictation for the Hermes desktop app.

WebSocket server on 127.0.0.1:8765. The renderer connects, sends 16kHz float32 PCM
(binary frames), receives JSON: {"type":"partial","text":...} during speech and
{"type":"final","text":...,"clean":...} on utterance boundaries (silero VAD).

Cleanup is deterministic (fillers, stutters, false starts). Optional LLM polish
(--polish) sends finals >15 words to glm-5.3-flash via the Hermes provider env;
on any failure it returns the regex-clean text. Nothing leaves the machine unless
--polish is on.

Usage: python3 sidecar.py [--model-dir DIR] [--port 8765] [--polish]
Model dir must contain encoder/decoder/joiner .int8.onnx + tokens.txt
(sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-streaming-240ms) + silero_vad.onnx.
"""
import argparse, asyncio, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor

MODEL_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

# ── deterministic cleanup ─────────────────────────────────────────────
FILLERS = r"(?:um+|uh+|erm|uhm+|hmm+|mhm+)"   # vocalized hesitations ONLY —
# discourse words (so, well, okay, you know, basically, like...) are REAL words
# the user said; stripping them loses content (user report: missing edge words)
COMMA_FILLER_RE = re.compile(rf",\s*{FILLERS}\s*,?\s*", re.I)   # ", uh," -> " "
FILLER_RE = re.compile(rf"(?<![\w'])\s*{FILLERS}\s*,?\s*", re.I) # " uh, " -> " "
LEADING_RE = re.compile(r"^\s*(?:um+|uh+|erm|uhm+|hmm+|mhm+)\b[\s,]*", re.I)
REPEAT_RE = re.compile(r"\b(\w+)(?:[,;]?\s+\1\b)+", re.I)           # "the the" / "the, the" -> "the"
FALSE_START_RE = re.compile(                                     # "I want to- I need to" -> "I need to"
    r"\b(\w+(?:\s+\w+){0,5}?)\s*-{1,2}\s+(\w+(?:\s+\w+){0,6})\b", re.I)
SPACE_RE = re.compile(r"\s{2,}")

def clean_text(t: str) -> str:
    t = t.strip()
    if not t:
        return t
    prev = None
    while prev != t:                       # repeat until stable (cleanup can expose more)
        prev = t
        t = FALSE_START_RE.sub(lambda m: m.group(2), t)
        t = COMMA_FILLER_RE.sub(" ", t)
        t = FILLER_RE.sub(" ", t)
        t = REPEAT_RE.sub(r"\1", t)
        t = LEADING_RE.sub("", t)
        t = re.sub(r"\s+([,.;:?!])", r"\1", t)   # "word ," -> "word,"
        t = re.sub(r",\s*,", ",", t)             # ",," -> ","
    t = SPACE_RE.sub(" ", t).strip()
    t = re.sub(r"^[\s,.;:?!-]+", "", t)          # leading orphan punctuation
    t = re.sub(r"\s+([,.;:?!])", r"\1", t)
    if t:
        t = t[0].upper() + t[1:]                 # sentence-start capital (zipformer emits lowercase)
        if not t[-1] in ".,;:?!":
            t = t + "."                          # terminal punctuation (zipformer emits none)
    return t

# ── optional LLM polish ────────────────────────────────────────────────
async def llm_polish(text: str) -> str:
    """Light rewrite for readability; returns input unchanged on any failure."""
    try:
        import litellm
    except Exception:
        return text
    key = os.environ.get("GLM_API_KEY", "")
    if not key:
        return text
    try:
        r = await litellm.acompletion(
            model="openai/glm-5.3-flash",
            api_base="https://api.z.ai/api/paas/v4",
            api_key=key,
            messages=[
                {"role": "system", "content": "Rewrite this dictation for clarity. Keep the speaker's "
                 "meaning and words; only fix grammar, punctuation and run-ons. Return ONLY the "
                 "rewritten text, no commentary."},
                {"role": "user", "content": text}],
            timeout=8, max_tokens=400)
        out = (r.choices[0].message.content or "").strip()
        return out if out else text
    except Exception:
        return text

# ── recognizer + VAD wrapper ───────────────────────────────────────────
def _find_model_file(model_dir: str, role: str):
    """Glob the model dir for encoder/decoder/joiner/tokens — works for both
    parakeet (encoder.int8.onnx) and zipformer (encoder-epoch-99-avg-1.onnx)
    layouts. Prefers int8 quantized when present."""
    import glob as _g
    i8 = sorted(_g.glob(os.path.join(model_dir, f"*{role}*.int8.onnx")))
    fp = [p for p in sorted(_g.glob(os.path.join(model_dir, f"*{role}*.onnx"))) if not p.endswith(".int8.onnx")]
    if role == "tokens":
        t = sorted(_g.glob(os.path.join(model_dir, "tokens*.txt")))
        return t[0] if t else None
    pick = i8 or fp
    return pick[0] if pick else None

def _is_streaming_encoder(encoder_path: str) -> bool:
    """Dialect detection WITHOUT loading via sherpa (a mismatched loader
    hard-exits the process). Rules verified against real exports:
      - NeMo streaming (parakeet-240/560): carries chunk_encoder_frames /
        buffered_streaming
      - k2 zipformer (streaming export): model_type starts with 'zipformer'
      - NeMo offline (parakeet non-streaming): model_type EncDecRNNTBPEModel,
        no chunk keys
    """
    try:
        import onnxruntime as _ort
    except ImportError:
        sys.exit("onnxruntime is required to auto-detect model type: "
                 ".venv/bin/pip install onnxruntime")
    try:
        meta = _ort.InferenceSession(encoder_path, providers=["CPUExecutionProvider"]).get_modelmeta()
        m = meta.custom_metadata_map or {}
        if "chunk_encoder_frames" in m or "buffered_streaming" in m:
            return True                                  # NeMo streaming
        mt = (m.get("model_type") or "").lower()
        if mt.startswith("zipformer"):
            return True                                  # k2 streaming zipformer export
        if "nemo" in mt or mt.startswith("encdec"):
            return False                                 # NeMo offline (nemo_transducer)
        sys.exit(f"cannot classify model: encoder metadata model_type={mt!r} "
                 f"keys={sorted(m.keys())}")
    except SystemExit:
        raise
    except Exception as e:
        sys.exit(f"could not inspect encoder metadata ({e}); refusing to guess model type")

def _build_vad(model_dir: str, threshold: float):
    """Silero VAD; falls back to sibling model dirs when this one ships no VAD
    (the offline parakeet tarball doesn't include one)."""
    import sherpa_onnx, glob as _g
    vad = os.path.join(model_dir, "silero_vad.onnx")
    if not os.path.exists(vad):
        for cand in sorted(_g.glob(os.path.join(os.path.dirname(os.path.abspath(model_dir)), "*", "silero_vad.onnx"))):
            vad = cand
            break
    vcfg = sherpa_onnx.VadModelConfig()
    vcfg.silero_vad.model = vad
    vcfg.silero_vad.threshold = threshold
    vcfg.silero_vad.min_speech_duration = 0.2
    vcfg.silero_vad.min_silence_duration = 0.4
    vcfg.sample_rate = 16000
    return sherpa_onnx.VoiceActivityDetector(vcfg, buffer_size_in_seconds=60) if os.path.exists(vad) else None

def build_recognizer(model_dir: str, threads: int, threshold: float = 0.6):
    """Returns (recognizer, vad, is_offline). Offline models are accepted here
    for OFFLINE-ONLY mode: no partials, finals decode on utterance end."""
    import sherpa_onnx, os as _os
    if threads <= 0:
        threads = max(1, _os.cpu_count() or 1)   # default: use every core
    enc = _find_model_file(model_dir, "encoder")
    dec = _find_model_file(model_dir, "decoder")
    joi = _find_model_file(model_dir, "joiner")
    tok = _find_model_file(model_dir, "tokens")
    for p, r in ((enc, "encoder"), (dec, "decoder"), (joi, "joiner"), (tok, "tokens")):
        if not p:
            sys.exit(f"missing {r} model file in {model_dir}")
    if not _is_streaming_encoder(enc):
        # OFFLINE-ONLY mode: parakeet int8 non-streaming as the sole model
        rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=enc, decoder=dec, joiner=joi, tokens=tok, num_threads=threads,
            model_type="nemo_transducer")
        vad_obj = _build_vad(model_dir, threshold)
        return rec, vad_obj, True
    rec = sherpa_onnx.OnlineRecognizer.from_transducer(
        encoder=enc, decoder=dec, joiner=joi, tokens=tok, num_threads=threads)
    vad_obj = _build_vad(model_dir, threshold)
    return rec, vad_obj, False

SAMPLE_RATE = 16000

class Session:
    """One websocket client = one dictation session. VAD segments utterances;
    each utterance is decoded streaming (partials), finalized on silence."""
    SILENCE_MS = 500            # end-of-utterance silence
    MAX_UTT_S = 30              # hard cap per utterance
    VAD_WINDOW = 512            # silero window size in samples (32ms @16k)
    PREROLL_S = 0.5             # ring of pre-speech audio so onsets aren't clipped
    NOISE_CAL_S = 2.0           # seconds of room noise sampled to set the adaptive floor
    RMS_MARGIN = 2.5            # open utterance at 2.5x the learned noise floor (slightly more sensitive)

    def __init__(self, rec, vad, polish, off=None):
        self.rec, self.vad, self.polish, self.off = rec, vad, polish, off
        self.offline_only = off == "offline_only"
        if self.offline_only:
            self.off = None               # rec IS the offline model
        self.stream = None if self.offline_only else rec.create_stream()
        self.in_utt = False
        self.silence_ms = 0
        self.utt_ms = 0
        self._vad_buf = []
        self._last_partial = ""
        self._preroll = []
        self._utt_audio = []                        # captured audio for offline re-decode
        self._noise_rms = 0.003                 # provisional floor until calibrated
        self._cal_rms = []                      # per-chunk rms during calibration window
        self._calibrated = False
        self._last_level = 0.0                  # for the idle-level diagnostics line

    def _voiced(self) -> bool:
        """is_speech_detected is a METHOD in sherpa-onnx 1.13.x — call it."""
        v = self.vad.is_speech_detected
        return bool(v() if callable(v) else v)

    def _feed_vad(self, pcm) -> bool:
        """Buffer audio, feed silero in 512-sample windows; return latest voiced state."""
        if self.vad is None:
            return True
        self._vad_buf.extend(pcm.tolist())
        n = len(self._vad_buf) // self.VAD_WINDOW * self.VAD_WINDOW
        if n:
            import numpy as np
            self.vad.accept_waveform(np.array(self._vad_buf[:n], dtype=np.float32))
            self._vad_buf = self._vad_buf[n:]
        return self._voiced()

    @staticmethod
    def _rms(pcm) -> float:
        import numpy as np
        return float(np.sqrt(np.mean(np.square(pcm)))) if len(pcm) else 0.0

    def feed(self, pcm) -> list:
        """Feed float32 chunk (any size); returns list of messages to send.
        CPU-heavy and synchronous — callers should run it in an executor.

        Silence costs (almost) nothing: the ASR stream is only fed inside an
        utterance (plus a preroll so onsets aren't clipped). VAD (cheap, tiny
        model) runs always. The energy gate is ADAPTIVE: the first ~2s of
        quiet audio calibrates the room's noise floor; speech must exceed
        floor * 3.5 AND satisfy silero."""
        out = []
        import numpy as np
        ms = int(1000 * len(pcm) / SAMPLE_RATE)
        voiced = self._feed_vad(pcm) if len(pcm) > 0 else self.in_utt
        rms = self._rms(pcm)
        self._last_level = rms
        if not self.in_utt and not self._calibrated:
            # calibration window: collect per-chunk levels; use a LOW percentile so
            # speech during calibration (VAD lags onsets ~300ms) cannot poison the
            # floor, and cap the gate at 0.06 abs so quiet rooms can't self-block
            self._cal_rms.append(rms)
            if len(self._cal_rms) * (ms if ms else 80) >= self.NOISE_CAL_S * 1000:
                arr = sorted(self._cal_rms)
                self._noise_rms = max(0.0008, min(arr[len(arr) // 4], 0.02))
                self._calibrated = True
                print(f"[hermes-dictate] noise floor calibrated: rms={self._noise_rms:.5f} "
                      f"(speech gate = {self._noise_rms * self.RMS_MARGIN:.5f})", flush=True)
        loud = rms >= self._noise_rms * self.RMS_MARGIN
        # speech requires BOTH silero agreement and energy over the adaptive floor
        # — opening AND continuing. Vetoing only on open let silero alone hold an
        # utterance open through a pause (breath/room noise reads as speech at
        # 0.5), so silence_ms never accumulated and the cut never fired.
        if voiced and not loud:
            voiced = False
        if voiced and not self.in_utt:
            # utterance opens: flush preroll into the ASR stream, then this chunk
            pre = np.concatenate(self._preroll) if self._preroll else np.zeros(0, dtype=np.float32)
            if len(pre):
                if self.stream is not None:
                    self.stream.accept_waveform(SAMPLE_RATE, pre)
                self._utt_audio.append(pre)
            if self.stream is not None:
                self.stream.accept_waveform(SAMPLE_RATE, pcm)
            self._utt_audio.append(pcm)
            self.in_utt = True
            self.silence_ms = 0
            self._preroll = []
        elif self.in_utt:
            if voiced:
                if self.stream is not None:
                    self.stream.accept_waveform(SAMPLE_RATE, pcm)
                self._utt_audio.append(pcm)
                self.silence_ms = 0
            else:
                self.silence_ms += ms
                if self.silence_ms < self.SILENCE_MS:
                    if self.stream is not None:
                        self.stream.accept_waveform(SAMPLE_RATE, pcm)   # trailing silence up to the cut
                    self._utt_audio.append(pcm)
                if self.silence_ms >= self.SILENCE_MS:
                    out.extend(self._finalize())
                    self._last_partial = ""
        else:
            # idle: remember last PREROLL_S of audio, no decode at all
            self._preroll.append(pcm)
            total = sum(len(p) for p in self._preroll)
            keep = self.PREROLL_S * SAMPLE_RATE
            i = 0
            while total > keep and i < len(self._preroll):
                total -= len(self._preroll[i]); i += 1
            if i:
                self._preroll = self._preroll[i:]
            # speak-level diagnostics while gated (throttled: only on notable change)
            lv = rms * 1000
            if abs(lv - getattr(self, "_last_diag_lv", -1)) > 1.5:
                vad_s = "VAD+" if voiced else "vad-"
                gate = self._noise_rms * self.RMS_MARGIN * 1000
                print(f"[hermes-dictate] idle: level {lv:.1f} (gate {gate:.1f}) {vad_s}"
                      f"{' [calibrating]' if not self._calibrated else ''}", flush=True)
                self._last_diag_lv = lv
            return out                          # idle: nothing to emit
        # decode whatever is now pending (utterance audio only; offline-only has no stream)
        if self.stream is None:
            return out
        steps = 0
        last_text = self._last_partial
        while self.rec.is_ready(self.stream):
            self.rec.decode_stream(self.stream)
            steps += 1
            if steps % 2 == 0:
                t = (self.rec.get_result(self.stream) or "").strip()
                if t and t != last_text:
                    out.append({"type": "partial", "text": t})
                    last_text = t
        partial = (self.rec.get_result(self.stream) or "").strip()
        if partial and partial != last_text:
            out.append({"type": "partial", "text": partial})
            last_text = partial
        self._last_partial = last_text
        if self.in_utt:
            self.utt_ms += ms
            if self.utt_ms >= self.MAX_UTT_S * 1000:
                out.extend(self._finalize())
                self._last_partial = ""
        return out

    def _finalize(self):
        final = ""
        if self.offline_only:
            # sole offline model: decode the captured utterance directly
            if self._utt_audio:
                try:
                    import numpy as np
                    audio = np.concatenate(self._utt_audio)
                    ostream = self.rec.create_stream()
                    ostream.accept_waveform(SAMPLE_RATE, audio)
                    self.rec.decode_stream(ostream)
                    final = (getattr(ostream.result, "text", "") or "").strip()
                except Exception as e:
                    print(f"[hermes-dictate] offline decode failed ({e})", flush=True)
        else:
            final = self.rec.get_result(self.stream)
            # hybrid: re-decode the captured utterance with the offline model for
            # higher quality + punctuation; falls back to the streaming text
            if self.off is not None and self._utt_audio:
                try:
                    import numpy as np
                    audio = np.concatenate(self._utt_audio)
                    ostream = self.off.create_stream()
                    ostream.accept_waveform(SAMPLE_RATE, audio)
                    self.off.decode_stream(ostream)
                    hi = (getattr(ostream.result, "text", "") or "").strip()
                    if hi:
                        final = hi
                except Exception as e:
                    print(f"[hermes-dictate] offline re-decode failed ({e}); using streaming text", flush=True)
        self._utt_audio = []
        # drain silero's completed-segment queue (else it grows for process lifetime)
        if self.vad is not None:
            try:
                while not self.vad.empty():
                    self.vad.pop()
            except Exception:
                pass
        # reset stream for next utterance
        if not self.offline_only:
            self.stream = self.rec.create_stream()
        self.in_utt, self.silence_ms, self.utt_ms = False, 0, 0
        text = (final or "").strip()
        if not text:
            return []
        clean = clean_text(text)
        return [{"type": "final", "text": text, "clean": clean}]

# ── optional second-pass recognizer (hybrid mode) ──────────────────────
def build_offline_recognizer(model_dir: str, threads: int):
    """Offline parakeet (int8 non-streaming) for high-quality finals.
    Returns None if the dir doesn't exist — hybrid silently off."""
    import sherpa_onnx
    if not os.path.isdir(model_dir):
        return None
    enc = _find_model_file(model_dir, "encoder")
    dec = _find_model_file(model_dir, "decoder")
    joi = _find_model_file(model_dir, "joiner")
    tok = _find_model_file(model_dir, "tokens")
    if not (enc and dec and joi and tok):
        print(f"[hermes-dictate] offline model incomplete in {model_dir}; hybrid off", flush=True)
        return None
    t = threads if threads > 0 else max(1, os.cpu_count() or 1)
    try:
        off = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=enc, decoder=dec, joiner=joi, tokens=tok, num_threads=t,
            model_type="nemo_transducer")
        return off
    except Exception as e:
        print(f"[hermes-dictate] offline model load failed ({e}); hybrid off", flush=True)
        return None

# ── server ─────────────────────────────────────────────────────────────
async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=MODEL_DIR_DEFAULT)
    ap.add_argument("--final-model-dir", default=None,
                    help="offline model dir for high-quality finals (hybrid mode); "
                         "e.g. model-parakeet-off. Skipped if missing.")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--threads", type=int, default=0, help="decode threads; 0 = auto (all cores)")
    ap.add_argument("--vad-threshold", type=float, default=0.5, help="silero threshold 0..1 (lower=more sensitive)")
    ap.add_argument("--polish", action="store_true", help="LLM polish long finals via glm-5.3-flash")
    args = ap.parse_args()

    t0 = time.time()
    rec, vad, is_offline = build_recognizer(args.model_dir, args.threads, args.vad_threshold)
    t_rec = time.time() - t0
    off = build_offline_recognizer(args.final_model_dir, args.threads) if args.final_model_dir else None
    mode = "offline-only" if is_offline else ("hybrid" if off else "streaming")
    print(f"[hermes-dictate] model loaded in {t_rec:.1f}s (mode={mode}, polish={'on' if args.polish else 'off'})", flush=True)

    import websockets

    async def handler(ws):
        sess = Session(rec, vad, args.polish, off=("offline_only" if is_offline else off))
        loop = asyncio.get_running_loop()
        print("[hermes-dictate] client connected", flush=True)
        decode_pool = ThreadPoolExecutor(max_workers=1)  # serialize decode; keeps order
        inbox = asyncio.Queue(maxsize=256)               # reader -> decoder; bounded backpressure
        closed = asyncio.Event()

        async def decoder():
            """Drain audio, decode, emit messages. Coalesces queued chunks when
            behind (concat, capped so partial cadence stays bounded), never
            processes more than ~600ms per batch."""
            import numpy as np
            last_partial = ""
            MAX_BATCH = 9600        # samples: 600ms — bounds partial latency when behind
            while True:
                item = await inbox.get()
                if item is None:
                    break                               # stop/flush
                pcm = item
                # coalesce anything already queued, but stop at the cap
                stop = False
                while not inbox.empty() and len(pcm) < MAX_BATCH:
                    nxt = inbox.get_nowait()
                    if nxt is None:
                        stop = True
                        break
                    pcm = np.concatenate([pcm, nxt])
                # if the stop sentinel was consumed above it stays in stop;
                # leftover chunks beyond the cap remain queued for the next loop
                t_dec = time.time()
                try:
                    for m in await loop.run_in_executor(decode_pool, sess.feed, pcm):
                        await ws.send(json.dumps(m))
                        if m["type"] == "partial":
                            last_partial = m["text"]
                except Exception as e:
                    print(f"[hermes-dictate] decode error: {e}", flush=True)
                    try:
                        await ws.send(json.dumps({"type": "status", "error": str(e)}))
                    except Exception:
                        pass
                dt = time.time() - t_dec
                audio_ms = len(pcm) / 16
                rtf = dt / (audio_ms / 1000) if audio_ms else 0
                if dt > 1.0:
                    print(f"[hermes-dictate] decode batch: {audio_ms:.0f}ms audio took {dt:.2f}s (RTF {rtf:.1f})", flush=True)
                    if rtf > 2.5:
                        print(f"[hermes-dictate] !! MODEL TOO HEAVY for this CPU (RTF {rtf:.1f})."
                              f" Try VARIANT=zip ./setup-model.sh && sidecar.py --model-dir model-zip (lighter model)", flush=True)
                if stop:
                    break
            try:
                for m in await loop.run_in_executor(decode_pool, sess._finalize):
                    await ws.send(json.dumps(m))
            except Exception as e:
                print(f"[hermes-dictate] flush error: {e}", flush=True)

        dec_task = asyncio.ensure_future(decoder())
        dropped = 0
        last_drop_log = time.time()

        def drop_quietest():
            """Under congestion, evict the quietest queued chunk (never the stop
            sentinel) so SPEECH survives backlog; silence is cheap to lose."""
            import numpy as np
            q = inbox._queue                      # deque of np arrays (+ possible None)
            best_i, best_e = None, None
            for i in range(0, len(q), 4):         # stride 4: cheap scan
                item = q[i]
                if item is None:
                    continue
                e = float(np.sum(np.square(item[::4])))
                if best_e is None or e < best_e:
                    best_i, best_e = i, e
            if best_i is not None:
                del q[best_i]
                return True
            return False

        try:
            async for msg in ws:
                if isinstance(msg, bytes):
                    if inbox.qsize() >= 250:
                        # decoder behind: shed the quietest audio, keep speech
                        if drop_quietest():
                            dropped += 1
                        if time.time() - last_drop_log > 1.0:
                            print(f"[hermes-dictate] decoder behind: {dropped} chunks dropped (quiet ones first; decode slower than realtime?)", flush=True)
                            last_drop_log = time.time()
                    await inbox.put(np.frombuffer(msg, dtype=np.float32))
                elif isinstance(msg, str):
                    cmd = json.loads(msg)
                    if cmd.get("type") == "stop":
                        await inbox.put(None)
                        await dec_task
                        break
        except Exception as e:
            import traceback
            print(f"[hermes-dictate] client error: {e}", flush=True)
            traceback.print_exc()          # full trace — names the real failure mode
        finally:
            dec_task.cancel()
            decode_pool.shutdown(wait=False)

    # np is module-level-imported by build_recognizer's caller path; ensure here too
    import numpy as np  # noqa: F811 — used by handler's reader loop

    async with websockets.serve(handler, "127.0.0.1", args.port, max_size=2**22, ping_interval=None):
        print(f"[hermes-dictate] listening on ws://127.0.0.1:{args.port}", flush=True)
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
