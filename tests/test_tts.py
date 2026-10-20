import os, subprocess, sys, wave

import numpy as np
import pytest

from hibari import Hibari, SAMPLE_RATE
from hibari.voices import VOICES, cache_dir


@pytest.fixture(scope="module")
def tts():
    return Hibari()


def test_call_shape_and_level(tts):
    y = tts("Hello there, this is a test.", seed=0)
    assert y.dtype == np.float32 and y.ndim == 1
    assert 1.0 < len(y) / SAMPLE_RATE < 4.0
    assert 0.05 < np.abs(y).max() <= 1.0


def test_seed_reproducible(tts):
    a = tts("Same words twice.", seed=1)
    b = tts("Same words twice.", seed=1)
    c = tts("Same words twice.", seed=2)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_speed(tts):
    slow = tts("Speed changes the length of the audio.", speed=0.8, seed=0)
    fast = tts("Speed changes the length of the audio.", speed=1.25, seed=0)
    assert len(slow) > 1.25 * len(fast)


def test_stream_matches_call(tts):
    text = "First sentence here. Second one follows! Third asks a question?"
    chunks = list(tts.stream(text, seed=0))
    assert len(chunks) == 3
    assert np.array_equal(np.concatenate(chunks), tts(text, seed=0))


def test_split_sentences_keeps_titles_and_initials():
    assert Hibari.split_sentences("Dr. Smith met J. K. Rowling at 5 p.m. today. She said \"Hi.\" Then left!") == [
        "Dr. Smith met J. K. Rowling at 5 p.m. today.", "She said \"Hi.", "Then left!"]


def test_phonemize_and_from_phonemes(tts):
    ps = tts.phonemize("Hello there.")
    assert ps == ["həlˈO ðˈɛɹ."]
    y = tts.from_phonemes(ps[0], seed=0)
    assert np.array_equal(y, tts("Hello there.", seed=0))


def test_save(tts, tmp_path):
    p = tts.save("Saved to disk.", str(tmp_path / "x.wav"))
    with wave.open(p) as w:
        assert w.getframerate() == SAMPLE_RATE and w.getnchannels() == 1 and w.getnframes() > SAMPLE_RATE


def test_cli(tmp_path):
    out = tmp_path / "cli.wav"
    r = subprocess.run([sys.executable, "-m", "hibari.cli", "Hello from the command line.", "-o", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert out.exists() and os.path.getsize(out) > 20000
    r = subprocess.run([sys.executable, "-m", "hibari.cli", "Hello.", "--phonemes"], capture_output=True, text=True,
                       encoding="utf-8")
    assert r.returncode == 0 and r.stdout.strip() == "həlˈO."
    r = subprocess.run([sys.executable, "-m", "hibari.cli", "-o", str(out)], input="From stdin.", capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("name", [n for n, v in VOICES.items() if not v["bundled"]])
def test_other_voices_when_cached(name):
    """Runs only where the voice file is already in the cache (HIBARI_HOME); no download in tests."""
    if not os.path.exists(os.path.join(cache_dir(), VOICES[name]["file"])):
        pytest.skip(f"{name} not cached")
    tts = Hibari(voice=name)
    y = tts("Hello there, this is a test.", seed=0)
    assert 1.0 < len(y) / SAMPLE_RATE < 4.0 and 0.05 < np.abs(y).max() <= 1.0
    ps = tts.phonemize("Better water.")[0]
    assert ps == ("bˈɛtə wˈɔːtə." if VOICES[name]["british"] else "bˈɛTəɹ wˈɔTəɹ.")
