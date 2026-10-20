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
    F, K = spec.shape[1], n_fft // hop                                      # spec [bins, F] complex64
    frames = sfft.irfft(spec.T, n=n_fft, axis=1)                            # float32 [F, n_fft]
    frames *= WIN
    frames = frames.reshape(F, K, hop)
    y = np.zeros((F + K - 1, hop), np.float32)
    for k in range(K):
        y[k: k + F] += frames[:, k]
    y *= _ola_norm(F, n_fft, hop)
    half = n_fft // 2
    return y.reshape(-1)[half: half + F * hop]


@functools.lru_cache(maxsize=1)
def _interp_w(hop=HOP):
    """F.interpolate(scale_factor=hop, linear, align_corners=False) weights: sample j of frame k lies at
    k + (j + 0.5) / hop - 0.5, between frames (k-1, k) for j < hop/2 and (k, k+1) after."""
    u = (((np.arange(hop) + 0.5) / hop - 0.5) % 1.0).astype(np.float32)
    return u, (1.0 - u).astype(np.float32)


def pulses(f0, hop=HOP, sr=SR, vthresh=40.0, fmax=11000.0):
    """tiny_model.pulses: linear-interpolated F0, closed-form band-limited pulse train. float32 passes
    with the interpolation as two broadcast products (np.interp's per-sample search cost as much as
    the rest); the phase still accumulates in float64."""
    F, h2 = len(f0), hop // 2
    u, v = _interp_w(hop)
    fp = np.pad(np.asarray(f0, np.float32), 1, mode="edge")               # clamped at both ends
    f = np.empty((F, hop), np.float32)
    np.multiply(fp[:-2, None], v[:h2], out=f[:, :h2]); f[:, :h2] += fp[1:-1, None] * u[:h2]
    np.multiply(fp[1:-1, None], v[h2:], out=f[:, h2:]); f[:, h2:] += fp[2:, None] * u[h2:]
    f = f.reshape(-1)
    voiced = f > vthresh
    np.maximum(f, vthresh, out=f)
    t = np.cumsum(f, dtype=np.float64); t *= 1.0 / sr; np.modf(t, t); t *= 2 * np.pi   # wrap: formula is 2pi-periodic
    t = t.astype(np.float32)
    H = np.floor(fmax / f); H += 0.5
    s = np.sin(t * 0.5); s *= 2
    bad = np.abs(s) <= 2e-3                                               # sin((H+1/2)t) / (2 sin(t/2)) -> H+1/2
    e = np.sin(H * t)
    s[bad] = 1
    e /= s
    e[bad] = H[bad]
    e -= 0.5
    e[~voiced] = 0
    return e


def stft(x, F, win=WIN, n_fft=N_FFT, hop=HOP):
    """torch.stft(center=True, reflect pad, hann), first F frames -> contiguous [bins, F] complex64."""
    xp = np.pad(x, n_fft // 2, mode="reflect")
    fr = np.lib.stride_tricks.sliding_window_view(xp, n_fft)[::hop][:F] * win
    return np.ascontiguousarray(sfft.rfft(fr, axis=1, overwrite_x=True).T)


WIN_SRC = WIN / (N_FFT / 4)       # a unit cosine peaks at sum(win)/2 in the STFT, so A_h is the harmonic's magnitude


def synth(parts, f0=None, rng=np.random.default_rng()):
    """tiny_model.to_wav in complex64: each complex op is one pass where the real/imag form took 3-4."""
    if len(parts) == 2:
        return istft(*parts)
    lh, th, ln, lr, phi = parts
    spec = stft(pulses(f0), lh.shape[1], WIN_SRC)
    rot = np.empty(lh.shape, np.complex64)
    c = np.tanh(th); c *= np.pi
    np.cos(c, out=rot.real); np.sin(c, out=rot.imag); rot *= ex(lh)
    spec *= rot
    z = rng.standard_normal((lh.shape[0], 2 * lh.shape[1]), dtype=np.float32).view(np.complex64)
    z *= ex(ln); spec += z
    np.cos(phi, out=rot.real); np.sin(phi, out=rot.imag); rot *= ex(lr); spec += rot
    return ola(spec)


def wav(out):
    return synth(out[:5], out[5]) if len(out) == 6 else synth(out)


class Tiny:
    def __init__(self, model, threads=1):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = so.inter_op_num_threads = threads
        self.sess = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}            # exports since 2026-10-19 take a speed input
        vp = os.path.join(os.path.dirname(model), "vocab.json")
        self.vocab = json.load(open(vp, encoding="utf-8")) if os.path.exists(vp) else None

    def spec(self, ids, speed=1.0):                                       # spectrum parts (+ f0 for src)
        feed = {"ids": np.asarray([ids], np.int64)}
        if "speed" in self.inputs:
            feed["speed"] = np.asarray([speed], np.float32)
        return [o[0] for o in self.sess.run(None, feed)]

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
