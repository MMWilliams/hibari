import os, subprocess, sys

import pytest

from hibari import voices
from hibari.voices import VOICES, cache_dir, download, sha256_of, voice_path


def test_registry_entries():
    assert "heart" in VOICES and VOICES["heart"]["bundled"]
    for name, v in VOICES.items():
        for k in ("kokoro", "language", "accent", "gender", "file", "bytes", "sha256", "british"):
            assert k in v, (name, k)
        assert v["british"] == v["language"].endswith("GB")
        if v["bundled"]:
            assert os.path.exists(voice_path(name, download_missing=False))
        else:
            assert len(v["sha256"]) == 64 and v["bytes"] > 1_000_000


def test_cache_dir_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HIBARI_HOME", str(tmp_path))
    assert cache_dir() == str(tmp_path)
    monkeypatch.delenv("HIBARI_HOME")
    assert cache_dir().endswith("hibari")


def test_download_verifies_and_renames(tmp_path):
    src = tmp_path / "model.onnx"
    src.write_bytes(os.urandom(300_000))
    dest = tmp_path / "cache" / "sub" / "model.onnx"
    assert download(src.as_uri(), str(dest), sha256_of(str(src))) == str(dest)
    assert dest.read_bytes() == src.read_bytes()
    assert [p.name for p in dest.parent.iterdir()] == ["model.onnx"]          # no .part left behind
    with pytest.raises(RuntimeError, match="sha256"):
        download(src.as_uri(), str(dest.with_name("bad.onnx")), "0" * 64)
    assert not dest.with_name("bad.onnx").exists()
    with pytest.raises(RuntimeError, match="could not download"):
        download((tmp_path / "missing.onnx").as_uri(), str(dest.with_name("x.onnx")), "0" * 64)


def test_voice_path_downloads_once(monkeypatch, tmp_path):
    src_dir = tmp_path / "remote"; src_dir.mkdir()
    (src_dir / "fake.onnx").write_bytes(b"\x08" * 50_000)
    monkeypatch.setattr(voices, "BASE_URL", src_dir.as_uri() + "/")
    monkeypatch.setenv("HIBARI_HOME", str(tmp_path / "home"))
    monkeypatch.setitem(VOICES, "fake", {"kokoro": "xx_fake", "language": "en-US", "accent": "American", "gender": "female",
                                         "file": "fake.onnx", "bytes": 50_000, "sha256": sha256_of(str(src_dir / "fake.onnx")),
                                         "british": False, "bundled": False})
    with pytest.raises(FileNotFoundError):
        voice_path("fake", download_missing=False)
    p = voice_path("fake")
    assert p == os.path.join(str(tmp_path / "home"), "fake.onnx") and os.path.getsize(p) == 50_000
    (src_dir / "fake.onnx").unlink()
    assert voice_path("fake") == p                                             # cached: no second download
    with pytest.raises(ValueError, match="unknown voice"):
        voice_path("nobody")


def test_cli_list_voices():
    r = subprocess.run([sys.executable, "-m", "hibari.cli", "--list-voices"], capture_output=True, text=True)
    assert r.returncode == 0 and "heart" in r.stdout and "in the package" in r.stdout
