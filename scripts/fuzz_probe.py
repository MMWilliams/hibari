"""Why does a row sound fuzzy/robotic? Per eval row, with teacher durations so frames line up:
  HNR: harmonic peak vs between-harmonic valley energy (dB) in voiced frames, per band; lower = fuzzier
  pitch range: std of voiced log-F0 in semitones, predicted vs teacher; lower = flatter, more robotic
  branches: share of voiced-frame energy from the pulse, noise and free-phase branches

    python scripts/fuzz_probe.py runs/tiny_v3/snap/20k.pt --rows 0 1 2
"""
import argparse, json, os, sys
import numpy as np, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "training"))
from tiny_model import TinyKokoro, to_wav, pulses, N_FFT, HOP  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--rows", type=int, nargs="+", default=[0, 1, 2])
a = ap.parse_args()
c = torch.load(a.ckpt, map_location="cpu", weights_only=False)
m = TinyKokoro(**c["kw"]).eval(); m.load_state_dict(c["model"])
rows = json.load(open("runs/tiny_v3/eval_rows.json"))
win = torch.hann_window(N_FFT)
BANDS = ((0, 1000), (1000, 2500), (2500, 5000))


def hnr(y, f0):
    S = torch.stft(torch.as_tensor(y), N_FFT, HOP, N_FFT, win, center=True, return_complex=True).abs().pow(2).numpy()
    out = []
    for lo, hi in BANDS:
        pk = vl = 0.0
        for t in np.nonzero(f0 > 80)[0]:
            if t >= S.shape[1]:
                break
            bw = f0[t] / 20.0                                        # harmonic spacing in bins (20 Hz bins)
            for h in range(1, int(hi / f0[t]) + 1):
                if h * f0[t] < lo:
                    continue
                b = int(round(h * bw)); v = int(round((h + 0.5) * bw))
                if v + 1 < S.shape[0]:
                    pk += S[b - 1: b + 2, t].max(); vl += S[v, t]
        out.append(10 * np.log10(pk / max(vl, 1e-12)))
    return out


def semis(f):
    f = f[f > 80]
    return 12 * np.log2(f).std()


for i in a.rows:
    r = rows[i]
    yt = np.memmap(r["bin"], dtype=np.int16, mode="r")[r["offset"]: r["offset"] + r["samples"]].astype(np.float32) / 32767
    T = 2 * sum(r["dur"])
    fn = np.memmap(r["f0_bin"], dtype=np.float16, mode="r")[r["f0_offset"]: r["f0_offset"] + 2 * T].astype(np.float32)
    ft, nt = fn[:T], fn[T:]
    with torch.no_grad():
        h, _ = m.encode(torch.tensor([r["ids"]]), torch.zeros(1, len(r["ids"]), dtype=torch.bool))
        hf = m.expand(h, torch.tensor([r["dur"]]))
        fp, npred = m.prosody(hf)
        parts = m.decode(hf, fp, npred)
        ys = to_wav(parts, fp)[0].numpy()
        ytf = to_wav(m.decode(hf, torch.tensor(ft)[None], torch.tensor(nt)[None]), torch.tensor(ft)[None])[0].numpy()
    print(f"row {i}: {len(r['ids'])} tokens, {len(yt)/24000:.1f} s")
    for k, y in (("teacher", yt), ("student", ys), ("student, teacher F0", ytf)):
        print(f"  HNR dB {k:20s} " + "  ".join(f"{lo//1000}-{hi/1000:g}k {v:5.1f}" for (lo, hi), v in zip(BANDS, hnr(y, ft))))
    print(f"  pitch range semitones: teacher {semis(ft):.2f}  predicted {semis(fp[0].numpy()):.2f}"
          f"  |err| median {np.median(np.abs(12*np.log2(np.maximum(fp[0].numpy(),1)/np.maximum(ft,1)))[ft > 80]):.2f}")
    if m.src:
        lh, th, ln, lr, phi = (p[0] for p in parts)
        E = torch.stft(pulses(fp), N_FFT, HOP, N_FFT, win, center=True, return_complex=True)[0, :, : lh.shape[-1]] / (N_FFT / 4)
        v = torch.as_tensor(ft[: lh.shape[-1]] > 80)
        eh = (lh.exp() * E.abs()).pow(2)[:, v].sum(); en = (2 * (2 * ln).exp())[:, v].sum(); er = (2 * lr).exp()[:, v].sum()
        tot = eh + en + er
        print(f"  voiced energy share: pulse {eh/tot:.2f}  noise {en/tot:.2f}  free {er/tot:.2f}")
