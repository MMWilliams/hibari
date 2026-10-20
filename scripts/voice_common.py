"""Shared by voice_gap.py and voice_swap.py: the per-voice checkpoints, the trainer's held-out rows, teacher
audio and prosody from the dump, the teacher-forced student, UTMOS, frame energy."""
import glob, gzip, json, os, random, re
import numpy as np, torch
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "training"))
from tiny_model import TinyKokoro, ProsodyNet, to_wav, SR, HOP  # noqa: E402

VOICES = {
    "af_heart": dict(run="runs/tiny_v3", snap="40k", index="data/dist/shard*.jsonl", prosody="runs/prosody_v1/best.pt"),
    "af_kore": dict(run="runs/voices/af_kore/tiny", snap="25k", index="data/dist_af_kore/shard*.jsonl", prosody="runs/voices/af_kore/prosody/best.pt"),
    "am_fenrir": dict(run="runs/voices/am_fenrir/tiny", snap="30k", index="data/dist_am_fenrir/shard*.jsonl", prosody="runs/voices/am_fenrir/prosody/best.pt"),
    "bm_fable": dict(run="runs/voices/bm_fable/tiny", snap="25k", index="data/dist_bm_fable/shard*.jsonl", prosody="runs/voices/bm_fable/prosody/best.pt"),
    "bf_emma": dict(run="runs/voices/bf_emma/tiny", snap="40k", index="data/dist_bf_emma*/shard*.jsonl", prosody="runs/voices/bf_emma_v1/prosody/best.pt"),
}
DUMP_START = {"data/dist_bf_emma2": 40000}        # sentence offset of each dump (0 unless listed)
SIL_DB, QUIET_DB = -55.0, -50.0
dev = "cuda"
if os.environ.get("HIBARI_GPU_FRAC"):  # share of GPU memory this process may use (user limit)
    torch.cuda.set_per_process_memory_fraction(float(os.environ["HIBARI_GPU_FRAC"]))


def add_spec(spec):
    """--spec JSON: override or add VOICES entries; a key 'name@tag' inherits name's fields."""
    for k, v in json.loads(spec).items():
        VOICES[k] = {**VOICES.get(k.split("@")[0], {}), **v}


_lines = None


def text_of(r):
    global _lines
    if _lines is None:
        _lines = [l.strip() for l in gzip.open(os.path.join(ROOT, "data/sentences.txt.gz"), "rt", encoding="utf-8")]
        _lines = [l for l in _lines if 4 <= len(l.split()) <= 35]
        random.Random(0).shuffle(_lines)
    d = os.path.dirname(r["bin"]).replace("\\", "/")
    k = int(re.search(r"shard(\d+)", r["bin"]).group(1))
    return _lines[DUMP_START.get(d, 0) + k + 3 * r["sid"]]


def held_out(index, n):
    """The trainer's split (train_tiny.py): same filter and shuffle, the last 200 rows are held out."""
    rows = [json.loads(l) for f in sorted(glob.glob(index)) for l in open(f, encoding="utf-8")]
    rows = [r for r in rows if sum(r["dur"]) >= 66 and len(r["ids"]) <= 510]
    random.Random(1234).shuffle(rows)
    return rows[-200:][:n]


def teacher(r):
    w = np.memmap(r["bin"], dtype=np.int16, mode="r")[r["offset"]: r["offset"] + r["samples"]]
    T = 2 * sum(r["dur"])
    fn = np.memmap(r["f0_bin"], dtype=np.float16, mode="r")[r["f0_offset"]: r["f0_offset"] + 2 * T]
    return w.astype(np.float32) / 32767, np.asarray(fn[:T], np.float32), np.asarray(fn[T:], np.float32)


def load_voice(v):
    cfg = VOICES[v]
    c = torch.load(os.path.join(ROOT, cfg["run"], "snap", f"{cfg['snap']}.pt"), map_location="cpu", weights_only=False)
    m = TinyKokoro(**c["kw"]).to(dev).eval(); m.load_state_dict(c["model"])
    cp = torch.load(os.path.join(ROOT, cfg["prosody"]), map_location="cpu", weights_only=False)
    pn = ProsodyNet(**cp["kw"]).to(dev).eval(); pn.load_state_dict(cp["model"])
    return cfg, m, pn


def student_tf(m, r, f0, n, seed):
    """Student output with the teacher's durations, F0 and energy (only the decoder differs), fixed noise seed."""
    with torch.no_grad():
        ids = torch.tensor([r["ids"]], device=dev)
        f0g, ng = torch.from_numpy(f0)[None].to(dev), torch.from_numpy(n)[None].to(dev)
        h, _ = m.encode(ids, torch.zeros_like(ids, dtype=torch.bool))
        parts = m.decode(m.expand(h, torch.tensor([r["dur"]], device=dev)), f0g, ng)
        torch.manual_seed(seed)
        return parts, f0g, to_wav(parts, f0g)[0].cpu().numpy()


_utmos = None


def mos(y):
    global _utmos
    if _utmos is None:
        _utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(dev).eval()
    with torch.no_grad():
        return float(_utmos(torch.as_tensor(y, dtype=torch.float32)[None].to(dev), SR))


def frame_db(y, T):
    n = min(T, len(y) // HOP)
    rms = np.sqrt((y[: n * HOP].reshape(n, HOP) ** 2).mean(1) + 1e-14)
    return 20 * np.log10(rms + 1e-7)
