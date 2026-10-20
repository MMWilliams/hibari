#!/usr/bin/env bash
# Fresh check of a published release: new venv, pip install hibari-tts==<version> from PyPI, fresh HIBARI_HOME so every
# voice downloads from Hugging Face, synthesize with each voice, --list-voices, and the phonemes of "Dr. Smith" + a date.
#   bash scripts/check_release.sh 0.2.0 [python-executable]
set -euo pipefail
VER=$1
PY=${2:-python3}
TMP=$(mktemp -d)
echo "venv in $TMP with $($PY --version 2>&1)"
"$PY" -m venv "$TMP/venv"
if [ -x "$TMP/venv/bin/python" ]; then VPY="$TMP/venv/bin/python"; else VPY="$TMP/venv/Scripts/python.exe"; fi
"$VPY" -m pip install -q --upgrade pip >/dev/null
"$VPY" -m pip install -q "hibari-tts==$VER"
export HIBARI_HOME="$TMP/hibari_home"
cd "$TMP"
"$VPY" -c "import hibari; assert hibari.__version__ == '$VER', hibari.__version__; print('hibari', hibari.__version__)"
"$VPY" -m hibari.cli --list-voices
for v in $("$VPY" -c "from hibari.voices import VOICES; print(' '.join(VOICES))"); do
  "$VPY" -m hibari.cli "Hello from the $v voice, installed fresh from PyPI." --voice "$v" -o "$v.wav"
  ls -la "$v.wav"
done
ls -la "$HIBARI_HOME"
"$VPY" -m hibari.cli "Dr. Smith arrived on 10/12/2019 at 9:30am." --phonemes
"$VPY" -m hibari.cli "Dr. Smith arrived on 10/12/2019 at 9:30am." --phonemes --voice emma
"$VPY" -m hibari.cli --list-voices
echo "release check OK: $TMP"
