"""numpy waveform synthesis vs torch (to_wav) on real model output, and ONNX vs torch spectra.
For source models the noise branch is switched off on both sides, since its noise is random.

    python export/test_istft.py runs/tiny_v3/snap/10k.pt export/v3_10k
    python export/test_istft.py runs/tiny_v3/snap/30k.pt export/v3_final --prosody runs/prosody_v1/best.pt
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa
import numpy as np, torch
from tiny_model import TinyKokoro, ProsodyNet, to_wav, infer_pnet
from infer_numpy import synth, Tiny

ap = argparse.ArgumentParser()
ap.add_argument("ckpt", nargs="?", default="runs/tiny_v1/last.pt")
ap.add_argument("exp", nargs="?", default="export")
ap.add_argument("--prosody")
a = ap.parse_args()
c = torch.load(a.ckpt, map_location="cpu", weights_only=False)
m = TinyKokoro(**c.get("kw", {})).eval(); m.load_state_dict(c["model"])
pn = None
if a.prosody:
    cp = torch.load(a.prosody, map_location="cpu", weights_only=False)
    pn = ProsodyNet(**cp["kw"]).eval(); pn.load_state_dict(cp["model"])
rows = json.load(open("runs/tiny_v1/eval_rows.json"))
for r in rows[:3]:
    ids = torch.tensor([r["ids"]])
    with torch.no_grad():
        parts, f0 = infer_pnet(m, pn, ids) if pn else m.infer(ids)
        if m.src:
            parts = (*parts[:2], torch.full_like(parts[2], -30.0), *parts[3:])
        ref = to_wav(parts, f0)[0].numpy()
    y = synth([p[0].numpy() for p in parts], f0[0].numpy())
    print(f"L={ids.shape[1]:3d} frames={parts[0].shape[-1]:4d}  numpy-vs-torch wav max|err| {np.abs(y-ref).max():.2e} "
          f"(rms {np.sqrt((ref**2).mean()):.3f})", end="  ")
    for mp in (f"{a.exp}/tiny.onnx", f"{a.exp}/tiny.int8.onnx"):
        lm2 = Tiny(mp).spec(r["ids"])[0]
        same = lm2.shape == parts[0][0].shape
        d = np.abs(lm2 - parts[0][0].numpy()).mean() if same else float("nan")
        print(f"{os.path.basename(mp)}: frames {lm2.shape[-1]} mean|dlogmag| {d:.3f}", end="  ")
    print()
