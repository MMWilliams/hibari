"""Voices: the registry (assets/voices.json) and the on-demand download of voice models.

Every voice is a complete model with the same architecture, size and speed; "heart" ships inside the
package, the others are fetched once from Hugging Face into a cache directory (HIBARI_HOME, else the
OS user cache dir), checked against their sha256 and renamed into place atomically.

    from hibari.voices import VOICES, voice_path, cache_dir
"""
import hashlib, json, os, sys, tempfile, urllib.error, urllib.request

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
BASE_URL = "https://huggingface.co/maureesewilliams/hibari/resolve/main/voices/"
with open(os.path.join(ASSETS, "voices.json"), encoding="utf-8") as _f:
    VOICES = json.load(_f)                 # name -> {kokoro, language, accent, gender, file, bytes, sha256, british}
DEFAULT = "heart"


def cache_dir():
    """HIBARI_HOME, else %LOCALAPPDATA%\\hibari, ~/Library/Caches/hibari or $XDG_CACHE_HOME/hibari (~/.cache)."""
    d = os.environ.get("HIBARI_HOME")
    if d:
        return d
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Local")
    elif sys.platform == "darwin":
        base = os.path.join(os.path.expanduser("~"), "Library", "Caches")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "hibari")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest, sha256):
    """Fetch url into dest (atomically: temp file in the same directory, verified, then renamed)."""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(dest) + ".", suffix=".part", dir=os.path.dirname(dest) or ".")
    h = hashlib.sha256()
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=60) as r:
            for chunk in iter(lambda: r.read(1 << 20), b""):
                out.write(chunk)
                h.update(chunk)
        if h.hexdigest() != sha256:
            raise RuntimeError(f"hibari: {url} downloaded but its sha256 {h.hexdigest()[:12]}... does not match the "
                               f"registry ({sha256[:12]}...); not using it")
        os.replace(tmp, dest)
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        raise RuntimeError(f"hibari: could not download {url} ({e}). Check the connection, or copy the file to "
                           f"{dest} by hand.") from e
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return dest


def voice_path(name, download_missing=True):
    """Path of a voice's ONNX model; downloads it on first use (unless download_missing is False)."""
    if name not in VOICES:
        raise ValueError(f"hibari: unknown voice {name!r}; available: {', '.join(VOICES)}")
    v = VOICES[name]
    if v.get("bundled"):
        return os.path.join(ASSETS, v["file"])
    dest = os.path.join(cache_dir(), v["file"])
    if os.path.exists(dest):
        return dest
    if not download_missing:
        raise FileNotFoundError(dest)
    return download(BASE_URL + v["file"], dest, v["sha256"])


def describe():
    """One line per voice for --list-voices."""
    lines = []
    for name, v in VOICES.items():
        where = "in the package" if v.get("bundled") else ("cached" if os.path.exists(os.path.join(cache_dir(), v["file"]))
                                                            else f"downloaded on first use ({v['bytes'] / 1e6:.1f} MB)")
        lines.append(f"{name:8s} {v['language']}  {v['gender']:6s} {v['accent']:9s} Kokoro {v['kokoro']:10s} {where}")
    return "\n".join(lines)
