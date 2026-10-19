"""numpy iSTFT vs torch.istft on real model output, and ONNX vs torch spectra."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa
import numpy as np, torch
from tiny_model import TinyKokoro, spec_to_wav
from infer_numpy import istft, Tiny

ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/tiny_v1/last.pt"
c = torch.load(ckpt, map_location="cpu", weights_only=False)
m = TinyKokoro(**c.get("kw", {})).eval(); m.load_state_dict(c["model"])
rows = json.load(open("runs/tiny_v1/eval_rows.json"))
for r in rows[:3]:
    ids = torch.tensor([r["ids"]])
    with torch.no_grad():
        lm, ph = m.infer(ids)
        ref = spec_to_wav(lm, ph)[0].numpy()
    y = istft(lm[0].numpy(), ph[0].numpy())
    print(f"L={ids.shape[1]:3d} frames={lm.shape[-1]:4d}  numpy-vs-torch istft max|err| {np.abs(y-ref).max():.2e} "
          f"(rms {np.sqrt((ref**2).mean()):.3f})", end="  ")
    for mp in ("export/tiny.onnx", "export/tiny.int8.onnx"):
        lm2, ph2 = Tiny(mp).spec(r["ids"])
        same = lm2.shape == lm[0].shape
        d = np.abs(lm2 - lm[0].numpy()).mean() if same else float("nan")
        print(f"{os.path.basename(mp)}: frames {lm2.shape[-1]} mean|dlogmag| {d:.3f}", end="  ")
    print()
