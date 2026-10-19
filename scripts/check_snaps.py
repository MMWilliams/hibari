"""Cheap sanity check of snapshot wavs: RMS, spectral flatness (1 = white noise), silent fraction.
Says whether audio is static or silence, nothing about quality."""
import sys, glob, numpy as np, soundfile as sf
for p in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "runs/tiny_v1/snap/*_eval*.wav")):
    y, sr = sf.read(p)
    rms = np.sqrt((y ** 2).mean())
    n = len(y) // 1024 * 1024
    S = np.abs(np.fft.rfft(y[:n].reshape(-1, 1024) * np.hanning(1024), axis=1)) ** 2 + 1e-12
    flat = np.exp(np.log(S).mean(1)) / S.mean(1)
    fr = np.sqrt((y[:n].reshape(-1, 1024) ** 2).mean(1))
    print(f"{p}: {len(y)/sr:5.2f} s  rms {rms:.4f}  flatness med {np.median(flat):.3f}  "
          f"silent frames {(fr < 1e-3).mean()*100:.0f}%  peak {np.abs(y).max():.3f}")
