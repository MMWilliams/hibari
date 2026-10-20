"""Mel-cepstral distortion (MCD, dB) between the teacher and the student on an eval_voice.py output directory, with
dynamic time warping since the student's durations are its own. Lower is better; it is a different judge from UTMOS.
13 mel-cepstral coefficients (c1..c12 used) from an 80-band mel spectrogram, hop 12.5 ms.

    python scripts/mcd_eval.py runs/voices/bm_fable/eval [more eval dirs]
"""
import glob, json, os, sys
import numpy as np, soundfile as sf, librosa
import scipy.fft as sfft

SR = 24000


def mcep(y):
    M = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=2400, win_length=1200, hop_length=300, n_mels=80, fmax=12000, power=2.0)
    c = sfft.dct(0.5 * np.log(M + 1e-10), type=2, norm="ortho", axis=0)[:13]
    return c, 10 * np.log10(M.sum(0) + 1e-10)


def mcd(a, b, db_a, db_b, quiet=-50.0):
    """DTW on c1..c12, frames where either side is quiet (sum of mel power below `quiet` dB) are dropped."""
    D, wp = librosa.sequence.dtw(X=a[1:], Y=b[1:], metric="euclidean")
    wp = wp[::-1]
    live = (db_a[wp[:, 0]] > db_a.max() + quiet) & (db_b[wp[:, 1]] > db_b.max() + quiet)
    d = np.sqrt(((a[1:, wp[live, 0]] - b[1:, wp[live, 1]]) ** 2).sum(0))
    return float((10 / np.log(10)) * np.sqrt(2) * d.mean())


def score(d, student="tiny_int8"):
    vals = []
    for t in sorted(glob.glob(os.path.join(d, "*_teacher.wav"))):
        s = t.replace("_teacher.wav", f"_{student}.wav")
        if not os.path.exists(s):
            continue
        yt, _ = sf.read(t, dtype="float32"); ys, _ = sf.read(s, dtype="float32")
        ct, dt = mcep(yt); cs, ds = mcep(ys)
        vals.append(mcd(ct, cs, dt, ds))
    return float(np.mean(vals)), len(vals)


if __name__ == "__main__":
    for d in sys.argv[1:]:
        m, n = score(d)
        print(f"{d}: MCD {m:.2f} dB over {n} sentences (int8 student vs teacher, DTW)")
        p = os.path.join(d, "results.json")
        if os.path.exists(p):
            r = json.load(open(p)); r["mcd_int8"] = m; json.dump(r, open(p, "w"), indent=1)
