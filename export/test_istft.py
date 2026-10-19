"""numpy waveform synthesis vs torch (to_wav) on real model output, and ONNX vs torch spectra.
For source models the noise branch is switched off on both sides, since its noise is random.

    python export/test_istft.py runs/tiny_v3/snap/10k.pt export/v3_10k
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa
import numpy as np, torch
from tiny_model import TinyKokoro, to_wav
from infer_numpy import synth, Tiny

ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/tiny_v1/last.pt"
exp = sys.argv[2] if len(sys.argv) > 2 else "export"
c = torch.load(ckpt, map_location="cpu", weights_only=False)
m = TinyKokoro(**c.get("kw", {})).eval(); m.load_state_dict(c["model"])
rows = json.load(open("runs/tiny_v1/eval_rows.json"))
for r in rows[:3]:
    ids = torch.tensor([r["ids"]])
    with torch.no_grad():
        parts, f0 = m.infer(ids)
        if m.src:
            parts = (*parts[:2], torch.full_like(parts[2], -30.0), *parts[3:])
        ref = to_wav(parts, f0)[0].numpy()
    y = synth([p[0].numpy() for p in parts], f0[0].numpy())
    print(f"L={ids.shape[1]:3d} frames={parts[0].shape[-1]:4d}  numpy-vs-torch wav max|err| {np.abs(y-ref).max():.2e} "
          f"(rms {np.sqrt((ref**2).mean()):.3f})", end="  ")
    for mp in (f"{exp}/tiny.onnx", f"{exp}/tiny.int8.onnx"):
        lm2 = Tiny(mp).spec(r["ids"])[0]
        same = lm2.shape == parts[0][0].shape
        d = np.abs(lm2 - parts[0][0].numpy()).mean() if same else float("nan")
        print(f"{os.path.basename(mp)}: frames {lm2.shape[-1]} mean|dlogmag| {d:.3f}", end="  ")
    print()
