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
rng = np.random.default_rng()
def tick(k, t0):
    T[k] = T.get(k, 0.0) + time.perf_counter() - t0
for rep in range(3):
    for lh, th, ln, lr, phi, f0 in outs:
        t = time.perf_counter(); p = inf.pulses(f0); tick("pulses", t)
        t = time.perf_counter(); spec = inf.stft(p, lh.shape[1], inf.WIN_SRC); tick("stft", t)
        t = time.perf_counter(); z = rng.standard_normal((lh.shape[0], 2 * lh.shape[1]), dtype=np.float32).view(np.complex64); tick("noise", t)
        t = time.perf_counter()
        rot = np.empty(lh.shape, np.complex64)
        c = np.tanh(th); c *= np.pi
        np.cos(c, out=rot.real); np.sin(c, out=rot.imag); rot *= inf.ex(lh)
        spec *= rot
        z *= inf.ex(ln); spec += z
        np.cos(phi, out=rot.real); np.sin(phi, out=rot.imag); rot *= inf.ex(lr); spec += rot
        tick("combine", t)
        t = time.perf_counter(); y = inf.ola(spec); tick("ola", t)
        t = time.perf_counter(); y2 = inf.synth((lh, th, ln, lr, phi), f0); tick("synth total", t)
print(f"{secs:.1f} s audio, ms per s of audio (mean of 3):")
for k, v in T.items():
    print(f"  {k:12s} {1000 * v / 3 / secs:.2f}")
