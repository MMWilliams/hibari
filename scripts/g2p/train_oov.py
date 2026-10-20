"""Train the OOV grapheme-to-phoneme net: a small transformer encoder over characters with 4 output
slots per character, CTC loss, so one forward pass (no decoder loop) reads a word. Targets are
misaki's espeak-ng fallback outputs (make_oov_data.py). Exports ONNX (argmax ids out) + vocab json.

    python scripts/g2p/train_oov.py --gpu 1 --out src/hibari/assets/oov
"""
import argparse, json, math, os, random, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--train", default="g2p_work/oov_train.tsv")
ap.add_argument("--heldout", default="g2p_work/oov_heldout.tsv")
ap.add_argument("--out", default="src/hibari/assets/oov")
ap.add_argument("--gpu", type=int, default=1)
ap.add_argument("--d", type=int, default=128); ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--heads", type=int, default=4); ap.add_argument("--ff", type=int, default=384)
ap.add_argument("--slots", type=int, default=4); ap.add_argument("--maxlen", type=int, default=24)
ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--bs", type=int, default=512)
ap.add_argument("--lr", type=float, default=2e-3)
a = ap.parse_args()
dev = torch.device(f"cuda:{a.gpu}")


def read(p):
    return [l.rstrip("\n").split("\t") for l in open(p, encoding="utf-8") if "\t" in l]


train, heldout = read(a.train), read(a.heldout)
val, train = train[:5000], train[5000:]
in_chars = sorted({c for w, _ in train for c in w})
out_chars = sorted({c for _, ps in train for c in ps})
cin = {c: i + 1 for i, c in enumerate(in_chars)}                    # 0 = pad
cout = {c: i + 1 for i, c in enumerate(out_chars)}                  # 0 = CTC blank
print(f"train {len(train)} val {len(val)} heldout {len(heldout)}; {len(in_chars)} input chars, {len(out_chars)} output chars")


def encode(pairs):
    L = a.maxlen
    x = np.zeros((len(pairs), L), np.int64)
    y, ylen = [], []
    for i, (w, ps) in enumerate(pairs):
        ids = [cin.get(c, 0) for c in w[:L]]
        x[i, :len(ids)] = ids
        t = [cout[c] for c in ps if c in cout]
        y.append(t); ylen.append(len(t))
    Y = np.zeros((len(pairs), max(ylen)), np.int64)
    for i, t in enumerate(y):
        Y[i, :len(t)] = t
    return torch.from_numpy(x), torch.from_numpy(Y), torch.tensor(ylen)


class Net(nn.Module):
    def __init__(self, n_in, n_out, d, layers, heads, ff, slots, maxlen):
        super().__init__()
        self.emb = nn.Embedding(n_in, d, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(maxlen, d) * 0.02)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d, heads, ff, dropout=0.1, activation="gelu",
                                                                batch_first=True, norm_first=True) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, slots * n_out)
        self.slots, self.n_out = slots, n_out

    def forward(self, x):                                  # [B,L] -> logits [B, L*slots, n_out]
        pad = x == 0
        h = self.emb(x) + self.pos[: x.shape[1]]
        for layer in self.layers:
            h = layer(h, src_key_padding_mask=pad)
        h = self.head(self.norm(h))
        B, L, _ = h.shape
        return h.view(B, L * self.slots, self.n_out)


class Export(nn.Module):
    """Single word, no padding mask, argmax ids out: the runtime collapses CTC repeats/blanks."""
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, x):
        m = self.m
        h = m.emb(x) + m.pos[: x.shape[1]]
        for layer in m.layers:
            sa = layer.self_attn
            y = layer.norm1(h)
            q, k, v = F.linear(y, sa.in_proj_weight, sa.in_proj_bias).chunk(3, -1)
            L, D = q.shape[1], q.shape[-1] // sa.num_heads
            q, k, v = (t.view(1, L, sa.num_heads, D).transpose(1, 2) for t in (q, k, v))
            w = torch.softmax(q @ k.transpose(-1, -2) / (D ** 0.5), -1)
            h = h + sa.out_proj((w @ v).transpose(1, 2).reshape(1, L, sa.num_heads * D))
            h = h + layer.linear2(F.gelu(layer.linear1(layer.norm2(h))))
        h = m.head(m.norm(h))
        return torch.argmax(h.view(1, -1, m.n_out), -1)


def collapse(ids):
    out, prev = [], 0
    for i in ids:
        if i != prev and i != 0:
            out.append(i)
        prev = i
    return out


def decode(ids):
    inv = {i: c for c, i in cout.items()}
    return "".join(inv[i] for i in collapse(ids))


def edit(a_, b_):
    d = list(range(len(b_) + 1))
    for i, x in enumerate(a_, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b_, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[-1]


@torch.no_grad()
def evaluate(m, pairs):
    m.eval()
    x, _, _ = encode(pairs)
    ok = per_n = per_d = 0
    for i in range(0, len(pairs), 1024):
        xb = x[i:i + 1024]
        pred = m(xb.to(dev)).argmax(-1).cpu().numpy()
        for j, (w, ps) in enumerate(pairs[i:i + 1024]):
            hyp = decode(pred[j][: int((xb[j] > 0).sum()) * a.slots])     # only the word's own slots
            ok += hyp == ps
            per_d += edit(hyp, ps); per_n += len(ps)
    m.train()
    return ok / len(pairs), per_d / per_n


torch.manual_seed(0)
model = Net(len(in_chars) + 1, len(out_chars) + 1, a.d, a.layers, a.heads, a.ff, a.slots, a.maxlen).to(dev)
print("params", sum(p.numel() for p in model.parameters()))
X, Y, YL = encode(train)
XL = (X > 0).sum(1)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
steps = a.epochs * math.ceil(len(train) / a.bs)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
ctc = nn.CTCLoss(blank=0, zero_infinity=True)
t0, step = time.time(), 0
for epoch in range(a.epochs):
    perm = torch.randperm(len(train))
    tot = 0.0
    for i in range(0, len(train), a.bs):
        idx = perm[i:i + a.bs]
        x, y, yl, xl = X[idx].to(dev), Y[idx].to(dev), YL[idx], XL[idx]
        logits = model(x)
        lp = logits.log_softmax(-1).transpose(0, 1)        # [T,B,V]
        loss = ctc(lp, y, (xl * a.slots), yl)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); step += 1
        tot += loss.item() * len(idx)
    if epoch % 5 == 4 or epoch == a.epochs - 1:
        acc, per = evaluate(model, val)
        print(f"epoch {epoch}: loss {tot / len(train):.3f}  val word acc {acc:.4f} PER {per:.4f}  {time.time() - t0:.0f}s", flush=True)
    else:
        print(f"epoch {epoch}: loss {tot / len(train):.3f}  {time.time() - t0:.0f}s", flush=True)

acc, per = evaluate(model, heldout)
print(f"held-out corpus OOV words (unseen): word acc {acc:.4f} PER {per:.4f}")
os.makedirs(os.path.dirname(a.out), exist_ok=True)
model.eval().cpu()
torch.save({"model": model.state_dict(), "kw": dict(d=a.d, layers=a.layers, heads=a.heads, ff=a.ff, slots=a.slots,
            maxlen=a.maxlen), "in": in_chars, "out": out_chars}, a.out + ".pt")
ex = Export(model).eval()
x = torch.tensor([[cin[c] for c in "hibari"]])
torch.onnx.export(ex, (x,), a.out + ".onnx", dynamo=False, opset_version=17, input_names=["chars"],
                  output_names=["ids"], dynamic_axes={"chars": {1: "L"}, "ids": {1: "T"}})
json.dump({"in": in_chars, "out": out_chars, "slots": a.slots, "maxlen": a.maxlen},
          open(a.out + ".json", "w", encoding="utf-8"), ensure_ascii=False)
import onnxruntime as ort
s = ort.InferenceSession(a.out + ".onnx", providers=["CPUExecutionProvider"])
bad = 0
for w, ps in heldout[:300]:
    ids = np.array([[cin.get(c, 0) for c in w[:a.maxlen]]], np.int64)
    o = s.run(None, {"chars": ids})[0][0]
    with torch.no_grad():
        ref = model(torch.from_numpy(ids))[0].argmax(-1).numpy()
    bad += decode(o) != decode(ref)
print(f"onnx vs torch on 300 words: {bad} differ; {os.path.getsize(a.out + '.onnx') / 1e6:.2f} MB")
t = time.perf_counter()
for _ in range(200):
    s.run(None, {"chars": ids})
print(f"onnx latency {(time.perf_counter() - t) / 200 * 1000:.3f} ms per word")
