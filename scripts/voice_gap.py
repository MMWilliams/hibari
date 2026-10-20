"""What does the decoder get wrong for a voice? Teacher audio vs the student driven with the teacher's own
durations, F0 and energy (so only the decoder differs), on the trainer's held-out rows of that voice's dump,
with heart and kore as the reference. Per voice: mel log-spectral distance and bias by band, mel-cepstral
distortion (MCD), harmonic-to-noise ratio at the teacher's F0 by band, the student's own branch energy split
(pulse / noise / free), F0 statistics and pitch tracking (pyin) of both signals, frame energy envelope
errors, silence and pause floors, errors at phoneme and voicing transitions, teacher clipping, and the
per-sentence UTMOS drop with the sentence features it correlates with. Writes <out>/results.json (merged
per voice), <out>/<voice>/rows.json and the 3 worst sentences as wavs.

    python scripts/voice_gap.py --out runs/voice_gap
    python scripts/voice_gap.py --voices bm_fable@x --spec '{"bm_fable@x": {"run": "runs/voices/bm_fable_x/tiny", "snap": "10k"}}'
"""
import argparse, json, os, sys
import numpy as np, soundfile as sf, torch, librosa
import scipy.fft as sfft
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import voice_common as C  # noqa: E402
from voice_common import ROOT, SR, HOP, dev, SIL_DB, QUIET_DB  # noqa: E402
from tiny_model import to_wav, infer_pnet, pulses, N_FFT  # noqa: E402

BANDS = {"low": (0, 1000), "mid": (1000, 4000), "high": (4000, 12000)}
FINE = {"0-0.5k": (0, 500), "0.5-1k": (500, 1000), "1-2k": (1000, 2000), "2-4k": (2000, 4000), "4-8k": (4000, 8000), "8-12k": (8000, 12000)}
HB = {"0-1k": (0, 1000), "1-4k": (1000, 4000), "4-8k": (4000, 8000), "8-12k": (8000, 12000)}
NFFT = 2400                                        # 50 ms window zero-padded to 10 Hz bins

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="runs/voice_gap")
ap.add_argument("--voices", nargs="+", default=list(C.VOICES))
ap.add_argument("--n", type=int, default=50)
ap.add_argument("--pyin-rows", type=int, default=12)
ap.add_argument("--spec", default="{}", help="JSON overriding VOICES entries (candidate checkpoints); key 'name@tag' inherits name")
a = ap.parse_args()
C.add_spec(a.spec)
os.makedirs(a.out, exist_ok=True)
inv_vocab = {v: k for k, v in json.load(open(os.path.join(ROOT, "export/vocab.json"), encoding="utf-8")).items()}

WIN = torch.hann_window(1200, periodic=True, device=dev)
MEL = torch.tensor(librosa.filters.mel(sr=SR, n_fft=NFFT, n_mels=80, fmin=0, fmax=12000), device=dev)
mel_hz = librosa.mel_frequencies(82, fmin=0, fmax=12000)[1:-1]
fhz = np.arange(N_FFT // 2 + 1) * SR / N_FFT


def power(y, T):
    """|STFT|^2 on the 80 Hz frame grid: [1201, T] power and [80, T] mel power, both numpy."""
    S = torch.stft(torch.as_tensor(y, device=dev), NFFT, HOP, 1200, window=WIN, center=True, return_complex=True)[:, :T]
    P = S.abs().pow(2)
    return P.cpu().numpy(), (MEL @ P).cpu().numpy()


def hnr(P, f0, lo, hi):
    """Per voiced frame (F0 > 60 Hz): 10 log10 of the energy within +-20 Hz of the harmonics h*F0 in [lo, hi) over
    the energy within +-10 Hz of the midpoints (h+1/2)*F0, using the teacher's F0 curve for both signals."""
    out = np.full(len(f0), np.nan, np.float32)
    bw = SR / NFFT
    for t in range(min(len(f0), P.shape[1])):
        f = float(f0[t])
        if f <= 60:
            continue
        h = np.arange(max(1, int(np.ceil(lo / f))), int(hi / f) + 1)
        if len(h) == 0:
            continue
        hb = np.round(h * f / bw).astype(int)[:, None] + np.arange(-2, 3)[None]
        mb = np.round((h + 0.5) * f / bw).astype(int)[:, None] + np.arange(-1, 2)[None]
        hb, mb = hb.clip(0, P.shape[0] - 1), mb.clip(0, P.shape[0] - 1)
        out[t] = 10 * np.log10(P[hb, t].sum() + 1e-12) - 10 * np.log10(P[mb, t].sum() + 1e-12)
    return out


def st(x):
    return 12 * np.log2(np.maximum(x, 1e-3))


def mcep(melP):
    """13 mel-cepstral coefficients from log amplitude: [13, T]."""
    return sfft.dct(0.5 * np.log(melP + 1e-10), type=2, norm="ortho", axis=0)[:13]


def nanmean(x):
    x = np.asarray(x, np.float64)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


def avg(lst):
    if not lst:
        return {}
    if isinstance(lst[0], dict):
        return {k: avg([d[k] for d in lst]) for k in lst[0]}
    if isinstance(lst[0], list):
        return np.nanmean(np.array(lst, np.float64), 0).round(4).tolist()
    return nanmean(lst)


res_path = os.path.join(a.out, "results.json")
results = json.load(open(res_path)) if os.path.exists(res_path) else {}   # one voice at a time merges into the file
for v in a.voices:
    cfg, m, pn = C.load_voice(v)
    rows = C.held_out(os.path.join(ROOT, cfg["index"]), a.n)
    vdir = os.path.join(a.out, v); os.makedirs(vdir, exist_ok=True)
    print(f"[*] {v}: {cfg['run']} {cfg['snap']}, {len(rows)} held-out rows", flush=True)

    acc = {k: [] for k in ["lsd", "bias", "lsd_fine", "bias_fine", "lsd_v", "lsd_u", "mcd", "mcd_v", "hnr_t", "hnr_s", "hnr_d",
                           "share", "share_u", "f0", "env", "sil", "trans", "clip", "pyin"]}
    per_row = []
    for i, r in enumerate(rows):
        y_t, f0, n = C.teacher(r)
        T = len(f0)
        parts, f0g, y_s = C.student_tf(m, r, f0, n, i)
        with torch.no_grad():
            ids = torch.tensor([r["ids"]], device=dev)
            torch.manual_seed(i)
            pp, f0p = infer_pnet(m, pn, ids)
            y_p = to_wav(pp, f0p)[0].cpu().numpy()
            lh, th, ln, lr, phi = (p[0].float() for p in parts)
            E = torch.stft(pulses(f0g), N_FFT, HOP, N_FFT, torch.hann_window(N_FFT, device=dev), center=True,
                           return_complex=True)[0, :, :T] / (N_FFT / 4)
            ph = (torch.exp(lh.clamp(max=7)) * E.abs()).pow(2).cpu().numpy()
            pz = (2 * torch.exp(2 * ln.clamp(max=7))).cpu().numpy()
            pr = torch.exp(2 * lr.clamp(max=7)).cpu().numpy()
        P_t, M_t = power(y_t, T); P_s, M_s = power(y_s, T)
        voiced = f0 > 40
        db_t, db_s = C.frame_db(y_t, T), C.frame_db(y_s, T)
        T = min(len(db_t), len(db_s), T)
        live = db_t[:T] > QUIET_DB
        L_t, L_s = 10 * np.log10(M_t[:, :T] + 1e-10), 10 * np.log10(M_s[:, :T] + 1e-10)
        D = L_s - L_t

        lsd = {k: nanmean(np.sqrt((D[(mel_hz >= lo) & (mel_hz < hi)] ** 2).mean(0))[live]) for k, (lo, hi) in BANDS.items()}
        bias = {k: nanmean(D[(mel_hz >= lo) & (mel_hz < hi)][:, live]) for k, (lo, hi) in BANDS.items()}
        acc["lsd"].append(lsd); acc["bias"].append(bias)
        acc["lsd_fine"].append({k: nanmean(np.sqrt((D[(mel_hz >= lo) & (mel_hz < hi)] ** 2).mean(0))[live]) for k, (lo, hi) in FINE.items()})
        acc["bias_fine"].append({k: nanmean(D[(mel_hz >= lo) & (mel_hz < hi)][:, live]) for k, (lo, hi) in FINE.items()})
        full = np.sqrt((D ** 2).mean(0))
        acc["lsd_v"].append(nanmean(full[live & voiced[:T]])); acc["lsd_u"].append(nanmean(full[live & ~voiced[:T]]))
        dc = mcep(M_t[:, :T]) - mcep(M_s[:, :T])
        mcd_f = (10 / np.log(10)) * np.sqrt(2 * (dc[1:] ** 2).sum(0))
        acc["mcd"].append(nanmean(mcd_f[live])); acc["mcd_v"].append(nanmean(mcd_f[live & voiced[:T]]))

        ht = {k: hnr(P_t, f0, lo, hi) for k, (lo, hi) in HB.items()}
        hs = {k: hnr(P_s, f0, lo, hi) for k, (lo, hi) in HB.items()}
        acc["hnr_t"].append({k: float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan for k, x in ht.items()})
        acc["hnr_s"].append({k: float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan for k, x in hs.items()})
        acc["hnr_d"].append({k: float(np.nanmedian(hs[k] - ht[k])) if np.isfinite(hs[k] - ht[k]).any() else np.nan for k in HB})

        # student branch energy split per band (voiced / unvoiced frames): pulse, noise, free
        for key, msk in (("share", voiced), ("share_u", ~voiced)):
            d = {}
            for k, (lo, hi) in BANDS.items():
                sel = (fhz >= lo) & (fhz < hi)
                tot = np.array([ph[sel][:, msk].sum(), pz[sel][:, msk].sum(), pr[sel][:, msk].sum()])
                d[k] = (tot / max(tot.sum(), 1e-12)).tolist()
            acc[key].append(d)

        vf = f0[voiced]
        acc["f0"].append({"median": float(np.median(vf)) if len(vf) else np.nan, "p5": float(np.percentile(vf, 5)) if len(vf) else np.nan,
                          "p95": float(np.percentile(vf, 95)) if len(vf) else np.nan, "voiced": float(voiced.mean()),
                          "f0_0_40": float(((f0 > 0) & (f0 <= 40)).mean()), "f0_40_60": float(((f0 > 40) & (f0 <= 60)).mean()),
                          "f0_60_80": float(((f0 > 60) & (f0 <= 80)).mean()), "f0_le0": float((f0 <= 0).mean()),
                          "energy_0_40_db": nanmean(db_t[:T][((f0 > 0) & (f0 <= 40))[:T]])})
        e = (db_s - db_t)[:T]
        q = np.percentile(db_t[:T][live], [33, 67]) if live.sum() > 3 else (0, 0)
        acc["env"].append({"signed": nanmean(e[live]), "abs": nanmean(np.abs(e[live])),
                           "quiet": nanmean(e[live & (db_t[:T] < q[0])]), "mid": nanmean(e[live & (db_t[:T] >= q[0]) & (db_t[:T] < q[1])]),
                           "loud": nanmean(e[live & (db_t[:T] >= q[1])]), "voiced": nanmean(e[live & voiced[:T]]), "unvoiced": nanmean(e[live & ~voiced[:T]]),
                           "range_t": float(np.percentile(db_t[:T], 95) - np.percentile(db_t[:T], 10)),
                           "range_s": float(np.percentile(db_s[:T], 95) - np.percentile(db_s[:T], 10)),
                           "corr": float(np.corrcoef(db_t[:T][live], db_s[:T][live])[0, 1]) if live.sum() > 3 else np.nan})
        sil = db_t[:T] < SIL_DB
        inner = np.zeros(T, bool); edge = np.zeros(T, bool); npause = 0
        t = 0
        while t < T:
            if sil[t]:
                u = t
                while u < T and sil[u]:
                    u += 1
                if u - t >= 8 and t > 0 and u < T:
                    npause += 1; inner[t + 2: u - 2] = True; edge[t: t + 2] = True; edge[u - 2: u] = True
                t = u
            else:
                t += 1
        acc["sil"].append({"frac": float(sil.mean()), "student_db": nanmean(db_s[:T][sil]), "teacher_db": nanmean(db_t[:T][sil]),
                           "louder10": float((db_s[:T][sil] > db_t[:T][sil] + 10).mean()) if sil.any() else np.nan,
                           "pauses": npause, "pause_inner_s": nanmean(db_s[:T][inner]), "pause_inner_t": nanmean(db_t[:T][inner]),
                           "pause_edge_s": nanmean(db_s[:T][edge]), "pause_edge_t": nanmean(db_t[:T][edge]),
                           "floor_s": float(np.percentile(db_s[:T], 5)), "floor_t": float(np.percentile(db_t[:T], 5))})
        bnd = 2 * np.cumsum(r["dur"])[:-1]
        dist = np.abs(np.arange(T)[:, None] - bnd[None, :]).min(1) if len(bnd) else np.full(T, 9)
        vt = np.zeros(T, bool)
        for c_ in np.flatnonzero(np.diff(voiced[:T].astype(int))):
            vt[max(0, c_ - 1): c_ + 3] = True
        flux_t = np.abs(np.diff(L_t, axis=1)).mean(0); flux_s = np.abs(np.diff(L_s, axis=1)).mean(0)
        lv = live[1:]
        acc["trans"].append({"lsd_d0": nanmean(full[live & (dist == 0)]), "lsd_d1": nanmean(full[live & (dist == 1)]),
                             "lsd_d2": nanmean(full[live & (dist == 2)]), "lsd_d3": nanmean(full[live & (dist >= 3)]),
                             "lsd_vtrans": nanmean(full[live & vt]), "lsd_steady": nanmean(full[live & ~vt & (dist >= 3)]),
                             "mcd_d0": nanmean(mcd_f[live & (dist <= 1)]), "mcd_d3": nanmean(mcd_f[live & (dist >= 3)]),
                             "flux_t_bnd": nanmean(flux_t[lv & (dist[1:] <= 1)]), "flux_s_bnd": nanmean(flux_s[lv & (dist[1:] <= 1)]),
                             "flux_t_in": nanmean(flux_t[lv & (dist[1:] >= 3)]), "flux_s_in": nanmean(flux_s[lv & (dist[1:] >= 3)])})
        acc["clip"].append({"peak": float(np.abs(y_t).max()), "clipped": float((np.abs(y_t) >= 0.99).mean()),
                            "peak_s": float(np.abs(y_s).max())})
        if i < a.pyin_rows:
            py = {}
            for key, y in (("t", y_t), ("s", y_s)):
                pf, vfl, _ = librosa.pyin(y, fmin=50, fmax=600, sr=SR, frame_length=2400, hop_length=HOP, center=True)
                pf = pf[:T]; ok = np.isfinite(pf) & voiced[:T] & (f0[:T] > 60)
                d = np.abs(st(pf[ok]) - st(f0[:T][ok]))
                py[key] = {"err_st": float(np.median(d)) if ok.any() else np.nan, "gross": float((d > 6).mean()) if ok.any() else np.nan,
                           "voiced_agree": float((np.isfinite(pf) == voiced[:T]).mean())}
            acc["pyin"].append(py)

        u_t, u_s, u_p = C.mos(y_t), C.mos(y_s), C.mos(y_p)
        per_row.append({"i": i, "sid": r["sid"], "bin": r["bin"], "text": C.text_of(r),
                        "phonemes": "".join(inv_vocab.get(x, "?") for x in r["ids"][1:-1]),
                        "utmos_t": u_t, "utmos_s": u_s, "utmos_p": u_p, "drop": u_t - u_s, "drop_p": u_t - u_p,
                        "sec": len(y_t) / SR, "f0_med": acc["f0"][-1]["median"], "f0_std_st": float(np.std(st(vf))) if len(vf) else np.nan,
                        "voiced": float(voiced.mean()), "lowf0": float((vf < 80).mean()) if len(vf) else np.nan,
                        "hnr_mid_t": acc["hnr_t"][-1]["1-4k"], "hnr_high_t": acc["hnr_t"][-1]["4-8k"], "hnr_d_mid": acc["hnr_d"][-1]["1-4k"],
                        "pauses": npause, "mcd": acc["mcd"][-1], "lsd_low": lsd["low"], "lsd_mid": lsd["mid"], "lsd_high": lsd["high"],
                        "bias_high": bias["high"], "env_abs": acc["env"][-1]["abs"], "peak": acc["clip"][-1]["peak"],
                        "lsd_d0": acc["trans"][-1]["lsd_d0"], "sil_student_db": acc["sil"][-1]["student_db"]})
        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{len(rows)}", flush=True)

    res = {k: avg(x) for k, x in acc.items()}
    res["utmos"] = {"teacher": nanmean([p["utmos_t"] for p in per_row]), "t_dur_pro": nanmean([p["utmos_s"] for p in per_row]),
                    "pnet": nanmean([p["utmos_p"] for p in per_row])}
    res["clip"]["rows_clipped"] = float(np.mean([c["clipped"] > 0 for c in acc["clip"]]))
    res["snap"], res["run"] = cfg["snap"], cfg["run"]
    feats = ["sec", "f0_med", "f0_std_st", "voiced", "lowf0", "hnr_mid_t", "hnr_high_t", "hnr_d_mid", "pauses", "mcd", "lsd_low", "lsd_mid",
             "lsd_high", "bias_high", "env_abs", "peak", "lsd_d0", "sil_student_db", "utmos_t"]
    drops = np.array([p["drop"] for p in per_row])
    res["corr"] = {}
    for f in feats:
        x = np.array([p[f] for p in per_row], np.float64)
        ok = np.isfinite(x) & np.isfinite(drops)
        if ok.sum() > 5 and np.std(x[ok]) > 0:
            rho, pv = spearmanr(x[ok], drops[ok])
            res["corr"][f] = [float(rho), float(pv)]
    worst = sorted(per_row, key=lambda p: -p["drop"])[:5]
    res["worst"] = [{k: w[k] for k in ("i", "text", "utmos_t", "utmos_s", "drop", "sec", "f0_med", "hnr_mid_t", "mcd", "lsd_high", "pauses")} for w in worst]
    results[v] = res
    json.dump(per_row, open(os.path.join(vdir, "rows.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for k, w in enumerate(worst[:3]):
        r = rows[w["i"]]
        y_t, f0, n = C.teacher(r)
        _, _, y_s = C.student_tf(m, r, f0, n, w["i"])
        sf.write(os.path.join(vdir, f"worst{k}_teacher.wav"), y_t, SR); sf.write(os.path.join(vdir, f"worst{k}_student.wav"), y_s, SR)
    u = res["utmos"]
    print(f"    UTMOS teacher {u['teacher']:.3f}  t_dur_pro {u['t_dur_pro']:.3f}  pnet {u['pnet']:.3f} | MCD {res['mcd']:.2f} dB | "
          f"LSD low/mid/high {res['lsd']['low']:.2f}/{res['lsd']['mid']:.2f}/{res['lsd']['high']:.2f} dB | "
          f"bias {res['bias']['low']:+.2f}/{res['bias']['mid']:+.2f}/{res['bias']['high']:+.2f} dB", flush=True)
    json.dump(results, open(res_path, "w"), indent=1)
    del m, pn; torch.cuda.empty_cache()

print("\nvoice      UTMOS t/tf/pnet      MCD   LSD low/mid/high   bias low/mid/high   HNR_t 1-4k  HNR_s-t 1-4k/4-8k  F0 med  env|e|  sil_s dB  lsd d0/d3")
for v in a.voices:
    r = results[v]; u = r["utmos"]
    print(f"{v:10s} {u['teacher']:.2f}/{u['t_dur_pro']:.2f}/{u['pnet']:.2f}  {r['mcd']:5.2f}  "
          f"{r['lsd']['low']:.2f}/{r['lsd']['mid']:.2f}/{r['lsd']['high']:.2f}   "
          f"{r['bias']['low']:+.2f}/{r['bias']['mid']:+.2f}/{r['bias']['high']:+.2f}   "
          f"{r['hnr_t']['1-4k']:6.1f}   {r['hnr_d']['1-4k']:+5.1f}/{r['hnr_d']['4-8k']:+5.1f}   {r['f0']['median']:5.0f}  "
          f"{r['env']['abs']:.2f}   {r['sil']['student_db']:6.1f}   {r['trans']['lsd_d0']:.2f}/{r['trans']['lsd_d3']:.2f}")
