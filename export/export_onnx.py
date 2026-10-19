"""Export TinyKokoro to ONNX: ids [1,L] -> logmag, phase [1,601,F]. Durations and the token->frame
expansion are inside the graph; the iSTFT is left to the runtime (numpy irfft in infer_numpy.py).
Writes tiny.onnx, tiny.int8.onnx and vocab.json next to --out."""
import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))
import _env  # noqa
import collections
import numpy as np, torch, torch.nn as nn, onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnx import numpy_helper
from tiny_model import TinyKokoro


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
    def __init__(self, m):
        super().__init__(); self.m = m

    def encode(self, ids):
        te = self.m.text
        x = te.emb(ids).transpose(1, 2)
        for c in te.convs:
            x = c(x)
        x = x.transpose(1, 2)
        for _ in range(te.layers):
            x = attn_layer(te.attn, x, te.attn.self_attn.num_heads)
        h = te.norm(x).transpose(1, 2)
        return h, self.m.dur(h).squeeze(1)

    def forward(self, ids):                                       # [1,L] int64
        m = self.m
        h, logd = self.encode(ids)                                # [1,C,L], [1,L]
        dur = torch.round(torch.exp(logd[0]) - 1.0).clamp(min=1).long()
        cum = torch.cumsum(dur, 0)                                # frame f belongs to the first token with cum > f
        frames = torch.arange(dur.sum(), device=ids.device)         # cum[-1] mis-exports
        tok = (frames[:, None] >= cum[None, :]).sum(1)            # [F]
        hf = torch.index_select(h, 2, tok)                        # [1,C,F]
        f0, n = m.prosody(hf)
        return m.decode(hf, f0, n)


def fold_weight_transposes(path):
    """torch's constant folding mis-folds the dynamic duration path (wrong frame count), so export runs
    without it and Linear weights come out as Transpose(W). Fold those so the quantizer sees them."""
    m = onnx.load(path)
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    used = collections.Counter(i for n in g.node for i in n.input)
    rename, drop = {}, []
    for n in g.node:
        if n.op_type == "Transpose" and n.input[0] in inits and used[n.input[0]] == 1:
            w = numpy_helper.to_array(inits[n.input[0]])
            perm = next(a.ints for a in n.attribute if a.name == "perm")
            t = numpy_helper.from_array(np.ascontiguousarray(w.transpose(perm)), n.output[0])
            g.initializer.remove(inits[n.input[0]]); g.initializer.append(t)
            drop.append(n)
    for n in drop:
        g.node.remove(n)
    onnx.save(m, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/tiny_v1/last.pt")
    ap.add_argument("--out", default="export/tiny.onnx")
    a = ap.parse_args()
    c = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = TinyKokoro(**c.get("kw", {})).eval()
    model.load_state_dict(c["model"])
    ex = Export(model).eval()
    ids = torch.tensor([[0, 50, 83, 54, 156, 57, 135, 0]])
    with torch.no_grad():
        ref = ex(ids)
        lm0, _ = model.infer(ids)
        assert torch.allclose(lm0, ref[0], atol=1e-4), (lm0 - ref[0]).abs().max()
    torch.onnx.export(ex, (ids,), a.out, dynamo=False, opset_version=17, do_constant_folding=False, input_names=["ids"],
                      output_names=["logmag", "phase"],
                      dynamic_axes={"ids": {1: "L"}, "logmag": {2: "F"}, "phase": {2: "F"}})
    onnx.checker.check_model(a.out)
    fold_weight_transposes(a.out)
    q = a.out.replace(".onnx", ".int8.onnx")
    # MatMul only: ConvInteger ran slower than fp32 Conv on x86, and convs are 15% of the weights
    quantize_dynamic(a.out, q, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])
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
        lm, ph = s.run(None, {"ids": ids.numpy()})
        err = np.abs(lm - ref[0].numpy()).max()
        print(f"{p}: {os.path.getsize(p)/1e6:.2f} MB  frames {lm.shape[-1]}  max|logmag err| {err:.4f}")


if __name__ == "__main__":
    main()
