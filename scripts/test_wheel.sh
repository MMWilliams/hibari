#!/usr/bin/env bash
# Install the built wheel into a fresh virtual environment and run the test suite and the CLI there.
# Works in Git Bash on Windows and in Linux/macOS shells.
#   bash scripts/test_wheel.sh [python-executable] [extra pip args]
set -euo pipefail
PY=${1:-python3}
shift || true
ROOT=$(cd "$(dirname "$0")/.." && pwd)
WHL=$(ls -t "$ROOT"/dist/hibari*.whl | head -1)
TMP=$(mktemp -d)
echo "venv in $TMP with $($PY --version 2>&1)"
"$PY" -m venv "$TMP/venv"
if [ -x "$TMP/venv/bin/python" ]; then VPY="$TMP/venv/bin/python"; else VPY="$TMP/venv/Scripts/python.exe"; fi
"$VPY" -m pip install -q --upgrade pip >/dev/null
"$VPY" -m pip install -q "$WHL" pytest "$@"
cp -r "$ROOT/tests" "$TMP/tests"
cd "$TMP"
"$VPY" -c "import hibari, onnxruntime, numpy, platform; print('hibari', hibari.__version__, '| onnxruntime', onnxruntime.__version__, '| numpy', numpy.__version__, '|', platform.platform(), platform.machine())"
"$VPY" -m pytest tests -q
"$VPY" -m hibari.cli "Hello from a fresh install on $(uname -s)." -o fresh.wav && ls -la fresh.wav
"$VPY" -m hibari.cli "Hello there." --phonemes
"$VPY" -m hibari.cli --bench
"$VPY" -m hibari.cli --bench --threads 4
echo "hibari console script: $("$TMP/venv/bin/hibari" --version 2>/dev/null || "$TMP/venv/Scripts/hibari.exe" --version)"
