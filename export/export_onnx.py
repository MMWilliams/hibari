"""Export TinyKokoro to ONNX: ids [1,L], speed [1] -> logmag, phase [1,601,F] (src models: lh, theta, ln, lr,
phi [1,601,F] and f0 [1,F]). speed divides the predicted durations (1.0 = as trained). Durations and the token->frame expansion are inside the graph; the pulse
source, noise and iSTFT are left to the runtime (numpy in infer_numpy.py).
--prosody <ckpt> puts a ProsodyNet in the graph: it gives the durations and F0/energy, TinyKokoro
gives the text features and the decoder (its own duration/prosody heads are left out).
Writes tiny.onnx, tiny.int8.onnx and vocab.json next to --out."""
import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))
import _env  # noqa

import numpy as np, torch, torch.nn as nn, onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnx import numpy_helper
from tiny_model import TinyKokoro, ProsodyNet, infer_pnet


def attn_layer(layer, x, heads):
    """nn.TransformerEncoderLayer (norm_first, gelu) written out: nn.MultiheadAttention traces
    its reshapes with the sample's sequence length baked in."""
    sa = layer.self_attn
    y = layer.norm1(x)
    q, k, v = nn.functional.linear(y, sa.in_proj_weight, sa.in_proj_bias).chunk(3, -1)
    L, D = q.shape[1], q.shape[-1] // heads
    q, k, v = (t.view(1, L, heads, D).transpose(1, 2) for t in (q, k, v))      # [1,H,L,D]
    w = torch.softmax(q @ k.transpose(-1, -2) / (D ** 0.5), -1)
    y = (w @ v).transpose(1, 2).reshape(1, L, heads * D)
    x = x + sa.out_proj(y)
    y = layer.norm2(x)
    return x + layer.linear2(nn.functional.gelu(layer.linear1(y)))


class Export(nn.Module):
    def __init__(self, m, pn=None):
        super().__init__(); self.m, self.pn = m, pn

    @staticmethod
    def encode(te, ids):                                          # TextEncoder, no padding
        x = te.emb(ids).transpose(1, 2)
        for c in te.convs:
            x = c(x)
        x = x.transpose(1, 2)
        for _ in range(te.layers):
            x = attn_layer(te.attn, x, te.attn.self_attn.num_heads)
        return te.norm(x).transpose(1, 2)                         # [1,C,L]

    def forward(self, ids, speed):                                # ids [1,L] int64, speed [1] float32
        m, pn = self.m, self.pn
        h = self.encode(m.text, ids)                              # decoder's text features
        side = pn if pn is not None else m                        # durations and F0/energy come from here
        hp = self.encode(pn.text, ids) if pn is not None else h
        logd = side.dur(hp)[0, 0]                                 # [L]
        dur = torch.round((torch.exp(logd) - 1.0) / speed).clamp(min=1).long()   # speed 1 = teacher pace
        cum = torch.cumsum(dur, 0)                                # frame f belongs to the first token with cum > f
        frames = torch.arange(dur.sum(), device=ids.device)         # cum[-1] mis-exports
        tok = (frames[:, None] >= cum[None, :]).sum(1)            # [F]
        hf = torch.index_select(h, 2, tok)                        # [1,C,F]
        f0, n = side.prosody(hf if pn is None else torch.index_select(hp, 2, tok))
        parts = m.decode(hf, f0, n)
        return (*parts, f0) if m.src else parts


def fold_weight_transposes(path):
    """torch's constant folding mis-folds the dynamic duration path (wrong frame count), so export runs
    without it and Linear weights come out as Transpose(W), one per use (the shared attention layer
    transposes its weights every repeat). Fold each (W, perm) once and point all consumers at it, so
    the quantizer sees plain MatMul weights."""
    m = onnx.load(path)
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    folded, rename, drop = {}, {}, []
    for n in g.node:
        if n.op_type == "Transpose" and n.input[0] in inits:
            perm = tuple(next(a.ints for a in n.attribute if a.name == "perm"))
            key = (n.input[0], perm)
            if key not in folded:
                w = numpy_helper.to_array(inits[n.input[0]])
                g.initializer.append(numpy_helper.from_array(np.ascontiguousarray(w.transpose(perm)), n.output[0]))
                folded[key] = n.output[0]
            rename[n.output[0]] = folded[key]
            drop.append(n)
    for n in drop:
        g.node.remove(n)
    for n in g.node:
        for i, x in enumerate(n.input):
            if x in rename:
                n.input[i] = rename[x]
    used = {i for n in g.node for i in n.input}
    for t in [t for t in g.initializer if t.name not in used]:
        g.initializer.remove(t)
    onnx.save(m, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/tiny_v1/last.pt")
    ap.add_argument("--out", default="export/tiny.onnx")
    ap.add_argument("--prosody", help="ProsodyNet ckpt: its durations and F0/energy replace TinyKokoro's heads")
    ap.add_argument("--per-channel", action="store_true", help="per-channel int8 weight scales (+0.09 MB; closer to fp32: "
                    "fenrir UTMOS 4.385 vs 4.349 per-tensor, fp32 4.380)")
    a = ap.parse_args()
    c = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = TinyKokoro(**c.get("kw", {})).eval()
    model.load_state_dict(c["model"])
    pn = None
    if a.prosody:
        cp = torch.load(a.prosody, map_location="cpu", weights_only=False)
        pn = ProsodyNet(**cp["kw"]).eval()
        pn.load_state_dict(cp["model"])
    ex = Export(model, pn).eval()
    ids = torch.tensor([[0, 50, 83, 54, 156, 57, 135, 0]])
    speed = torch.ones(1)
    with torch.no_grad():
        ref = ex(ids, speed)
        parts, f0 = infer_pnet(model, pn, ids) if pn else model.infer(ids)
        for r, t in zip(ref, (*parts, f0)):
            assert torch.allclose(t, r, atol=1e-4), (t - r).abs().max()
    names = ["lh", "theta", "ln", "lr", "phi"] if model.src else ["logmag", "phase"]
    axes = {"ids": {1: "L"}, **{k: {2: "F"} for k in names}, **({"f0": {1: "F"}} if model.src else {})}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.onnx.export(ex, (ids, speed), a.out, dynamo=False, opset_version=17, do_constant_folding=False, input_names=["ids", "speed"],
                      output_names=names + (["f0"] if model.src else []), dynamic_axes=axes)
    onnx.checker.check_model(a.out)
    fold_weight_transposes(a.out)
    q = a.out.replace(".onnx", ".int8.onnx")
    # MatMul only: ConvInteger ran slower than fp32 Conv on x86, and convs are 15% of the weights
    quantize_dynamic(a.out, q, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"], per_channel=a.per_channel)
    try:
        from huggingface_hub import hf_hub_download
        cfg = json.load(open(hf_hub_download("hexgrad/Kokoro-82M", "config.json"), encoding="utf-8"))
        json.dump(cfg["vocab"], open(os.path.join(os.path.dirname(a.out), "vocab.json"), "w",
                                      encoding="utf-8"), ensure_ascii=False)
    except Exception as e:                                        # offline without cache: ids still work
        print(f"[!] vocab not written: {e}")
    import onnxruntime as ort
    for p in (a.out, q):
        s = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
        outs = s.run(None, {"ids": ids.numpy(), "speed": speed.numpy()})
        mb = os.path.getsize(p) / 1e6
        if outs[0].shape == ref[0].shape:
            err = max(np.abs(o - r.numpy()).max() for o, r in zip(outs, ref))
            print(f"{p}: {mb:.2f} MB  frames {outs[0].shape[-1]}  max|err| over all outputs {err:.4f}")
        else:
            print(f"{p}: {mb:.2f} MB  frames {outs[0].shape[-1]} (torch {ref[0].shape[-1]}): durations differ")


if __name__ == "__main__":
    main()
