"""Time each host-DSP stage of infer_numpy.synth per second of audio (1 thread, eval rows).

    python scripts/profile_dsp.py export/v3_final/tiny.int8.onnx
"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "export"))
import infer_numpy as inf

outs = [inf.Tiny(sys.argv[1]).spec(r["ids"]) for r in json.load(open("runs/tiny_v1/eval_rows.json"))]
secs = sum(o[0].shape[1] for o in outs) * inf.HOP / inf.SR
T = {}
def tick(k, t0):
    T[k] = T.get(k, 0.0) + time.perf_counter() - t0
for rep in range(3):
    for lh, th, ln, lr, phi, f0 in outs:
        t = time.perf_counter(); p = inf.pulses(f0); tick("pulses", t)
        t = time.perf_counter(); E = inf.stft(p, lh.shape[1]) / (inf.N_FFT / 4); tick("stft", t)
        t = time.perf_counter(); z = np.random.default_rng().standard_normal((2, *lh.shape), dtype=np.float32); tick("noise", t)
        t = time.perf_counter()
        er, ei = E.real.astype(np.float32), E.imag.astype(np.float32)
        ah, an, ar, c = inf.ex(lh), inf.ex(ln), inf.ex(lr), np.pi * np.tanh(th)
        cc, sc = np.cos(c), np.sin(c)
        spec = np.empty(lh.shape, np.complex64)
        spec.real = ah * (cc * er - sc * ei) + an * z[0] + ar * np.cos(phi)
        spec.imag = ah * (sc * er + cc * ei) + an * z[1] + ar * np.sin(phi)
        tick("combine", t)
        t = time.perf_counter(); y = inf.ola(spec); tick("ola", t)
        t = time.perf_counter(); y2 = inf.synth((lh, th, ln, lr, phi), f0); tick("synth total", t)
print(f"{secs:.1f} s audio, ms per s of audio (mean of 3):")
for k, v in T.items():
    print(f"  {k:12s} {1000 * v / 3 / secs:.2f}")
