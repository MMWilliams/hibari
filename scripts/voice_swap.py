"""Oracle swaps: which part of the student's output costs the UTMOS? On the same held-out rows as voice_gap.py,
the teacher-forced student's STFT (n_fft 1200, hop 300, the model's own grid) gets the teacher's bins in one band,
or the teacher's frames of one kind (voiced, unvoiced, silent, phoneme boundaries, voicing transitions), or the
teacher's phase with the student's magnitude (and the reverse), and is scored again. The share of the gap each
swap recovers says where the loss is. Also a periodicity measure per band that does not depend on the F0
curve lining up: the normalised autocorrelation peak near the F0 period, teacher vs student.

    python scripts/voice_swap.py --out runs/voice_gap
"""
import argparse, json, os, sys
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import voice_common as G  # noqa: E402  (VOICES, held_out, teacher, student_tf, mos, dev)
from voice_common import ROOT, SR, HOP, dev  # noqa: E402
from tiny_model import N_FFT  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="runs/voice_gap")
ap.add_argument("--voices", nargs="+", default=list(G.VOICES))
ap.add_argument("--n", type=int, default=50)
ap.add_argument("--spec", default="{}")
a = ap.parse_args()
G.add_spec(a.spec)
os.makedirs(a.out, exist_ok=True)
WIN = torch.hann_window(N_FFT, device=dev)
fhz = torch.arange(N_FFT // 2 + 1, device=dev) * SR / N_FFT


def stft(y):
    return torch.stft(torch.as_tensor(y, device=dev), N_FFT, HOP, N_FFT, WIN, center=True, return_complex=True)


def istft(S, n):
    return torch.istft(S, N_FFT, HOP, N_FFT, WIN, center=True, length=n).cpu().numpy()


def frames_mask(m, F):
    """80 Hz frame mask (length T) -> STFT frame mask (length F = T + 1)."""
    out = torch.zeros(F, dtype=torch.bool, device=dev)
    out[: len(m)] = torch.as_tensor(m, device=dev)
    if F > len(m):
        out[len(m):] = out[len(m) - 1]
    return out


def periodicity(y, f0, lo, hi):
    """Median over voiced frames (F0 > 60) of the normalised autocorrelation peak within +-10 % of the F0 period,
    on the band-passed signal (bins outside [lo, hi) zeroed)."""
    S = stft(y)
    S[(fhz < lo) | (fhz >= hi)] = 0
    x = torch.istft(S, N_FFT, HOP, N_FFT, WIN, center=True, length=len(y))
    T = min(len(f0), len(x) // HOP)
    xp = torch.nn.functional.pad(x, (N_FFT // 2, N_FFT // 2))
    fr = xp.unfold(0, N_FFT, HOP)[:T] * WIN                                  # [T, 1200]
    ac = torch.fft.irfft(torch.fft.rfft(fr, 2 * N_FFT).abs() ** 2, 2 * N_FFT)[:, :N_FFT]
    ac = ac / ac[:, :1].clamp(min=1e-12)
    f = torch.as_tensor(f0[:T], device=dev)
    ok = f > 60
    if ok.sum() == 0:
        return float("nan")
    P = SR / f[ok]
    lags = torch.arange(N_FFT, device=dev)[None]
    win = (lags >= (0.9 * P)[:, None]) & (lags <= (1.1 * P)[:, None])
    r = torch.where(win, ac[ok], torch.full_like(ac[ok], -1.0)).max(1).values
    return float(r.median())


res_path = os.path.join(a.out, "swap.json")
results = json.load(open(res_path)) if os.path.exists(res_path) else {}
for v in a.voices:
    cfg, m, _ = G.load_voice(v)
    rows = G.held_out(os.path.join(ROOT, cfg["index"]), a.n)
    print(f"[*] {v}: {cfg['run']} {cfg['snap']}, {len(rows)} rows", flush=True)
    scores = {}; per = {"0-1k": [[], []], "1-4k": [[], []], "4-8k": [[], []]}
    for i, r in enumerate(rows):
        y_t, f0, n = G.teacher(r)
        _, _, y_s = G.student_tf(m, r, f0, n, i)
        N = min(len(y_t), len(y_s)); y_t, y_s = y_t[:N], y_s[:N]
        St, Ss = stft(y_t), stft(y_s)
        F = St.shape[1]
        T = min(len(f0), N // HOP)
        db_t = G.frame_db(y_t, T)
        voiced = f0[:T] > 40
        sil = db_t < G.SIL_DB
        sil2 = np.convolve(sil.astype(float), np.ones(5), "same") > 0              # +-2 frames
        bnd = 2 * np.cumsum(r["dur"])[:-1]
        dist = np.abs(np.arange(T)[:, None] - bnd[None, :]).min(1) if len(bnd) else np.full(T, 9)
        vt = np.zeros(T, bool)
        for c_ in np.flatnonzero(np.diff(voiced.astype(int))):
            vt[max(0, c_ - 1): c_ + 3] = True
        bands = {"low": (0, 1000), "mid": (1000, 4000), "high": (4000, 12001), "2-6k": (2000, 6000)}
        var = {"student": Ss, "teacher": St}
        for k, (lo, hi) in bands.items():
            sel = ((fhz >= lo) & (fhz < hi))[:, None]
            var[f"T_{k}"] = torch.where(sel, St, Ss)                         # teacher's band into the student
            var[f"S_{k}"] = torch.where(sel, Ss, St)                         # student's band into the teacher
        for k, msk in (("voiced", voiced), ("unvoiced", ~voiced & ~sil), ("silent", sil2), ("bnd", dist <= 1), ("vtrans", vt)):
            fm = frames_mask(msk, F)[None]
            var[f"T_{k}"] = torch.where(fm, St, Ss)
            var[f"S_{k}"] = torch.where(fm, Ss, St)
        var["T_phase"] = torch.polar(Ss.abs(), St.angle())                   # student magnitude, teacher phase
        var["S_phase"] = torch.polar(St.abs(), Ss.angle())
        for k, S in var.items():
            scores.setdefault(k, []).append(G.mos(istft(S, N)))
        for k, (lo, hi) in (("0-1k", (0, 1000)), ("1-4k", (1000, 4000)), ("4-8k", (4000, 8000))):
            per[k][0].append(periodicity(y_t, f0, lo, hi)); per[k][1].append(periodicity(y_s, f0, lo, hi))
        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{len(rows)}", flush=True)
    u = {k: float(np.mean(x)) for k, x in scores.items()}
    gap = u["teacher"] - u["student"]
    rec = {k: (x - u["student"]) / gap for k, x in u.items() if k.startswith("T_")}
    hurt = {k: (u["teacher"] - x) / gap for k, x in u.items() if k.startswith("S_")}
    results[v] = {"utmos": u, "gap": gap, "recovered": rec, "lost": hurt,
                  "periodicity": {k: {"teacher": float(np.nanmedian(x[0])), "student": float(np.nanmedian(x[1]))} for k, x in per.items()},
                  "snap": cfg["snap"], "run": cfg["run"]}
    print(f"    teacher {u['teacher']:.3f} student {u['student']:.3f} gap {gap:.3f}")
    print("    recovered by teacher's: " + "  ".join(f"{k[2:]} {100 * x:4.0f}%" for k, x in rec.items()))
    print("    lost with student's:    " + "  ".join(f"{k[2:]} {100 * x:4.0f}%" for k, x in hurt.items()))
    print("    periodicity t/s: " + "  ".join(f"{k} {np.nanmedian(x[0]):.3f}/{np.nanmedian(x[1]):.3f}" for k, x in per.items()), flush=True)
    json.dump(results, open(res_path, "w"), indent=1)
    del m; torch.cuda.empty_cache()
