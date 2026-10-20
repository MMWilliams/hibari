"""Host-side signal processing: the network predicts a spectrum per 12.5 ms frame (five parts); this
turns it into a waveform. Pulse train from F0, filtered by the harmonic branch, plus shaped noise,
plus a free-phase branch, then one inverse STFT (n_fft 1200, hop 300, Hann, 24 kHz).

    S = exp(lh) e^{i pi tanh(theta)} STFT(pulses(f0)) + exp(ln) Z + exp(lr) e^{i phi}

scipy.fft keeps float32 and is 3-5x faster than numpy's float64 pocketfft; numpy is the fallback."""
import functools
import numpy as np

try:
    import scipy.fft as _sf

    def rfft(x, axis):
        return _sf.rfft(x, axis=axis, overwrite_x=True)

    def irfft(x, n, axis):
        return _sf.irfft(x, n=n, axis=axis)
    FFT_BACKEND = "scipy"
except ImportError:                                            # pragma: no cover
    def rfft(x, axis):
        return np.fft.rfft(x, axis=axis).astype(np.complex64)

    def irfft(x, n, axis):
        return np.fft.irfft(x, n=n, axis=axis).astype(np.float32)
    FFT_BACKEND = "numpy"

N_FFT, HOP, SR = 1200, 300, 24000
WIN = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)   # periodic Hann
WIN_SRC = WIN / (N_FFT / 4)       # a unit cosine peaks at sum(win)/2 in the STFT, so exp(lh) is the harmonic's magnitude


def ex(x):
    return np.exp(np.minimum(x, 7.0), dtype=np.float32)


@functools.lru_cache(maxsize=8)
def _ola_norm(F):
    """1 / summed squared window per output sample: constant inside, tapered at the ends."""
    K = N_FFT // HOP
    w2 = (WIN * WIN).reshape(K, HOP)
    wsum = np.zeros((F + K - 1, HOP), np.float32)
    for k in range(K):
        wsum[k: k + F] += w2[k]
    return 1.0 / np.maximum(wsum, 1e-11)


def ola(spec):
    """Inverse STFT matching torch.istft(center=True, length=F*hop): irfft per frame, window,
    overlap-add in hop-sized blocks (n_fft = 4 hops: a reshape and four shifted adds), normalize."""
    F, K = spec.shape[1], N_FFT // HOP                                      # spec [bins, F] complex64
    frames = irfft(spec.T, N_FFT, 1)                                        # float32 [F, n_fft]
    frames *= WIN
    frames = frames.reshape(F, K, HOP)
    y = np.zeros((F + K - 1, HOP), np.float32)
    for k in range(K):
        y[k: k + F] += frames[:, k]
    y *= _ola_norm(F)
    half = N_FFT // 2
    return y.reshape(-1)[half: half + F * HOP]


@functools.lru_cache(maxsize=1)
def _interp_w():
    """Linear interpolation weights (align_corners=False): sample j of frame k lies at
    k + (j + 0.5) / hop - 0.5, between frames (k-1, k) for j < hop/2 and (k, k+1) after."""
    u = (((np.arange(HOP) + 0.5) / HOP - 0.5) % 1.0).astype(np.float32)
    return u, (1.0 - u).astype(np.float32)


def pulses(f0, vthresh=40.0, fmax=11000.0):
    """Band-limited pulse train from the 80 Hz F0 curve: every harmonic below fmax at unit amplitude,
    sum_{h<=H} cos(h t) = sin((H+1/2) t) / (2 sin(t/2)) - 1/2. Unvoiced frames (f0 < vthresh) are 0."""
    F, h2 = len(f0), HOP // 2
    u, v = _interp_w()
    fp = np.pad(np.asarray(f0, np.float32), 1, mode="edge")
    f = np.empty((F, HOP), np.float32)
    np.multiply(fp[:-2, None], v[:h2], out=f[:, :h2]); f[:, :h2] += fp[1:-1, None] * u[:h2]
    np.multiply(fp[1:-1, None], v[h2:], out=f[:, h2:]); f[:, h2:] += fp[2:, None] * u[h2:]
    f = f.reshape(-1)
    voiced = f > vthresh
    np.maximum(f, vthresh, out=f)
    t = np.cumsum(f, dtype=np.float64); t *= 1.0 / SR; np.modf(t, t); t *= 2 * np.pi   # phase in [0, 2pi)
    t = t.astype(np.float32)
    H = np.floor(fmax / f); H += 0.5
    s = np.sin(t * 0.5); s *= 2
    bad = np.abs(s) <= 2e-3                                               # limit of sin((H+1/2)t)/(2 sin(t/2))
    e = np.sin(H * t)
    s[bad] = 1
    e /= s
    e[bad] = H[bad]
    e -= 0.5
    e[~voiced] = 0
    return e


def stft(x, F):
    """torch.stft(center=True, reflect pad, Hann) of the pulse train, first F frames -> [bins, F] complex64."""
    xp = np.pad(x, N_FFT // 2, mode="reflect")
    fr = np.lib.stride_tricks.sliding_window_view(xp, N_FFT)[::HOP][:F] * WIN_SRC
    return np.ascontiguousarray(rfft(fr, 1).T)


def synth(lh, th, ln, lr, phi, f0, rng):
    """Network outputs (each [601, F] float32, f0 [F]) + a numpy Generator for the noise -> float32 wav."""
    spec = stft(pulses(f0), lh.shape[1])
    rot = np.empty(lh.shape, np.complex64)
    c = np.tanh(th); c *= np.pi
    np.cos(c, out=rot.real); np.sin(c, out=rot.imag); rot *= ex(lh)
    spec *= rot
    z = rng.standard_normal((lh.shape[0], 2 * lh.shape[1]), dtype=np.float32).view(np.complex64)
    z *= ex(ln); spec += z
    np.cos(phi, out=rot.real); np.sin(phi, out=rot.imag); rot *= ex(lr); spec += rot
    return ola(spec)
