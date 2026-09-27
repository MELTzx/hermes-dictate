#!/usr/bin/env bash
# Downloads the parakeet-unified-en-0.6b int8 streaming model + silero VAD.
# VARIANT=240 (default) | 560 | 1120  — streaming chunk size in ms.
# Bigger chunk = much less CPU (RTF), slightly higher latency + WER.
#   240ms: lowest latency, heaviest CPU (RTF ~5 on a ThinkPad X1 — too slow there)
#   560ms: ~2-3x lighter, WER actually BETTER (6.99 vs 7.35), 0.56s latency — dictation sweet spot
#   1120ms: lightest, 1.12s latency — fallback for very weak CPUs
# Extracts into model/ (240) or model-<VARIANT>/; pass the dir to sidecar.py --model-dir.
set -euo pipefail
cd "$(dirname "$0")"
V="${VARIANT:-240}"
case "$V" in
  240|560|1120)
    DIR="model"; [ "$V" = "240" ] || DIR="model-$V"
    URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-streaming-${V}ms.tar.bz2"
    TARB="parakeet-${V}ms.tar.bz2"
    ;;
  zip)
    DIR="model-zip"
    URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2"
    TARB="zipformer-en.tar.bz2"
    ;;
  parakeet-off)
    DIR="model-parakeet-off"
    URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming.tar.bz2"
    TARB="parakeet-off.tar.bz2"
    ;;
  *)
    echo "VARIANT must be 240|560|1120|zip|parakeet-off" >&2; exit 1 ;;
esac

echo ">> variant $V -> $DIR"
# presence check works for both layouts (parakeet: encoder.int8.onnx; zipformer: encoder-epoch-*.onnx)
ls "$DIR"/encoder*.onnx >/dev/null 2>&1 && { echo ">> already present, skipping download"; }
ls "$DIR"/encoder*.onnx >/dev/null 2>&1 || {
  curl -sL --retry 5 --retry-delay 3 -C - -o "$TARB" "$URL"
  echo ">> size: $(stat -c%s "$TARB")"
  bzip2 -t "$TARB" && echo ">> bzip2 OK"
  rm -rf "$DIR" && mkdir -p "$DIR"
  tar xf "$TARB" -C "$DIR" --strip-components=1 --exclude='*.md'
  rm -f "$TARB"
}
[ -f "$DIR/silero_vad.onnx" ] || curl -sL --retry 5 -o "$DIR/silero_vad.onnx" \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
echo ">> contents:"
ls -la "$DIR" | awk 'NR>1 {printf "   %10s  %s\n", $5, $9}'
echo ">> done. Start with:  .venv/bin/python sidecar.py --model-dir $DIR"
echo ">> or bench first:    .venv/bin/python bench.py $DIR"
