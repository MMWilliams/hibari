"""TinyKokoro inference with onnxruntime + numpy/scipy.fft only: ids -> ONNX spectrum parts -> iSTFT.
Source models (src=True) also get the F0 pulse train and noise built here, matching tiny_model.to_wav.
scipy.fft keeps float32 (numpy 1.26's pocketfft runs in float64 and is 3-5x slower here).

    python export/infer_numpy.py --model export/tiny.int8.onnx --ids 0,50,83,... --out x.wav
    python export/infer_numpy.py --phonemes "hˈɛlO" --out x.wav      (needs vocab.json next to the model)
    python export/infer_numpy.py --bench runs/tiny_v1/eval_rows.json   (RTF, 1 thread, fp32 vs int8)
"""
import argparse, functools, json, os, time
import numpy as np
import scipy.fft as sfft

N_FFT, HOP, SR = 1200, 300, 24000


WIN = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)   # periodic hann


def ex(x):
    return np.exp(np.minimum(x, 7.0), dtype=np.float32)


def istft(logmag, phase):
    mag = ex(logmag)
    return ola(mag * np.cos(phase) + 1j * (mag * np.sin(phase)))


@functools.lru_cache(maxsize=8)
def _ola_norm(F, n_fft=N_FFT, hop=HOP):
    """1 / summed squared window per output sample: constant inside, tapered at the ends."""
    K = n_fft // hop
    w2 = (WIN * WIN).reshape(K, hop)
    wsum = np.zeros((F + K - 1, hop), np.float32)
    for k in range(K):
        wsum[k: k + F] += w2[k]
    return 1.0 / np.maximum(wsum, 1e-11)


def ola(spec, n_fft=N_FFT, hop=HOP):
    """Matches torch.istft(center=True, hann window, length=F*hop): irfft per frame, overlap-add,
    divide by the summed squared window, trim n_fft/2 on both ends. n_fft = 4*hop, so output block j
    is the sum of quarter k of frame j-k for k in 0..3: a reshape and four shifted adds, no loop."""
    spec = spec.astype(np.complex64, copy=False)                           # [bins, F]
    F, K = spec.shape[1], n_fft // hop
    frames = sfft.irfft(spec.T, n=n_fft, axis=1)                            # float32 [F, n_fft]
    frames *= WIN
    frames = frames.reshape(F, K, hop)
    y = np.zeros((F + K - 1, hop), np.float32)
    for k in range(K):
        y[k: k + F] += frames[:, k]
    y *= _ola_norm(F, n_fft, hop)
    half = n_fft // 2
    return y.reshape(-1)[half: half + F * hop]


def pulses(f0, hop=HOP, sr=SR, vthresh=40.0, fmax=11000.0):
    """tiny_model.pulses: linear-interpolated F0, closed-form band-limited pulse train."""
    F = len(f0)
    x = np.clip((np.arange(F * hop) + 0.5) / hop - 0.5, 0, F - 1)        # interpolate(align_corners=False)
    f = np.interp(x, np.arange(F), f0).astype(np.float32)
    voiced = f > vthresh
    f = np.maximum(f, vthresh)
    t = (2 * np.pi * np.modf(np.cumsum(f / sr, dtype=np.float64))[0]).astype(np.float32)
    H = np.floor(fmax / f)
    s = np.sin(t / 2)
    ok = np.abs(s) > 1e-3
    e = np.where(ok, np.sin((H + 0.5) * t) / (2 * np.where(ok, s, 1)), H + 0.5) - 0.5
    return (e * voiced).astype(np.float32)


def stft(x, F, n_fft=N_FFT, hop=HOP):
    """torch.stft(center=True, reflect pad, hann), first F frames -> [bins, F] complex64."""
    xp = np.pad(x, n_fft // 2, mode="reflect")
    fr = np.lib.stride_tricks.sliding_window_view(xp, n_fft)[::hop][:F] * WIN
    return sfft.rfft(fr, axis=1).T


def synth(parts, f0=None, rng=np.random.default_rng()):
    if len(parts) == 2:
        return istft(*parts)
    lh, th, ln, lr, phi = parts
    E = stft(pulses(f0), lh.shape[1]) / (N_FFT / 4)
    er, ei = np.ascontiguousarray(E.real), np.ascontiguousarray(E.imag)
    z = rng.standard_normal((2, *lh.shape), dtype=np.float32)
    ah, an, ar = ex(lh), ex(ln), ex(lr)
    c = np.tanh(th); c *= np.pi
    cc, sc = np.cos(c), np.sin(c)
    re = cc * er; re -= sc * ei; re *= ah; re += an * z[0]; re += ar * np.cos(phi)   # in place: half the temporaries
    im = sc * er; im += cc * ei; im *= ah; im += an * z[1]; im += ar * np.sin(phi)
    spec = np.empty(lh.shape, np.complex64)
    spec.real, spec.imag = re, im
    return ola(spec)


def wav(out):
    return synth(out[:5], out[5]) if len(out) == 6 else synth(out)


class Tiny:
    def __init__(self, model, threads=1):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = so.inter_op_num_threads = threads
        self.sess = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
        vp = os.path.join(os.path.dirname(model), "vocab.json")
        self.vocab = json.load(open(vp, encoding="utf-8")) if os.path.exists(vp) else None

    def spec(self, ids):                                                  # spectrum parts (+ f0 for src)
        return [o[0] for o in self.sess.run(None, {"ids": np.asarray([ids], np.int64)})]

    def __call__(self, ids):
        return wav(self.spec(ids))

    def phonemes_to_ids(self, ph):
        return [0, *[self.vocab[c] for c in ph if c in self.vocab], 0]


def bench(rows_path, models, threads=1):
    rows = json.load(open(rows_path))
    for mp in models:
        t = Tiny(mp, threads)
        for r in rows[:2]:
            t(r["ids"])                                               # warm-up
        tot_t = tot_net = tot_a = 0.0
        for r in rows:
            t0 = time.perf_counter(); out = t.spec(r["ids"]); t1 = time.perf_counter()
            y = wav(out); t2 = time.perf_counter()
            tot_net += t1 - t0; tot_t += t2 - t0; tot_a += len(y) / SR
        print(f"{mp}: {os.path.getsize(mp)/1e6:.2f} MB  {threads} thread  RTF {tot_t/tot_a:.4f} "
              f"(net {tot_net/tot_a:.4f}, host dsp {(tot_t-tot_net)/tot_a:.4f})  over {tot_a:.1f} s audio")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="export/tiny.onnx")
    ap.add_argument("--ids"); ap.add_argument("--phonemes"); ap.add_argument("--out", default="out.wav")
    ap.add_argument("--bench", help="eval_rows.json: time fp32 + int8 on 1 thread")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    if a.bench:
        bench(a.bench, [a.model, a.model.replace(".onnx", ".int8.onnx")], a.threads)
    else:
        t = Tiny(a.model, a.threads)
        ids = [int(x) for x in a.ids.split(",")] if a.ids else t.phonemes_to_ids(a.phonemes)
        y = t(ids)
        import soundfile as sf
        sf.write(a.out, y, SR)
        print(f"{a.out}: {len(y)/SR:.2f} s")
