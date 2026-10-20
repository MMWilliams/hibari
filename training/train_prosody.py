"""Train ProsodyNet alone on the teacher's durations and F0/energy curves: no audio, whole sentences,
so it takes minutes per thousand steps instead of the decoder's hours.

    python training/train_prosody.py --out runs/prosody_v1 --baseline runs/tiny_v3/snap/20k.pt
"""
import argparse, os, glob, json, random, time
from pathlib import Path
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader

import _env  # noqa: F401
from tiny_model import ProsodyNet, TinyKokoro


class Rows(Dataset):
    def __init__(self, rows):
        self.rows, self._mm = rows, {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        mm = self._mm.setdefault(r["f0_bin"], np.memmap(r["f0_bin"], dtype=np.float16, mode="r"))
        T = 2 * sum(r["dur"])
        fn = np.asarray(mm[r["f0_offset"]: r["f0_offset"] + 2 * T], dtype=np.float32)
        return r["ids"], r["dur"], fn[:T], fn[T:]


def collate(b):
    B, L, T = len(b), max(len(x[0]) for x in b), max(len(x[2]) for x in b)
    ids = torch.zeros(B, L, dtype=torch.long); dur = torch.zeros(B, L, dtype=torch.long)
    f0, n, fm = torch.zeros(B, T), torch.zeros(B, T), torch.zeros(B, T)
    lens = torch.tensor([len(x[0]) for x in b])
    for i, (a, d, f, e) in enumerate(b):
        ids[i, : len(a)] = torch.tensor(a); dur[i, : len(d)] = torch.tensor(d)
        f0[i, : len(f)] = torch.from_numpy(f); n[i, : len(e)] = torch.from_numpy(e); fm[i, : len(f)] = 1
    return ids, dur, lens, f0, n, fm


def expand(h, dur, F40):
    """Batched token -> 40 Hz frame expansion; frames past a sentence's end read its last token and are masked."""
    cum = dur.cumsum(1)                                                  # [B,L]
    tok = (torch.arange(F40, device=h.device)[None, :, None] >= cum[:, None, :]).sum(-1).clamp(max=h.shape[2] - 1)
    return h.gather(2, tok[:, None].expand(-1, h.shape[1], -1))


def losses(model, batch, dev):
    ids, dur, lens, f0, n, fm = (x.to(dev) for x in batch)
    pad = torch.arange(ids.shape[1], device=dev)[None] >= lens[:, None]
    h, logd = model.encode(ids, pad)
    valid = (~pad).float()
    l_dur = ((logd - torch.log1p(dur.float())).abs() * valid).sum() / valid.sum()
    hf = expand(h, dur, f0.shape[1] // 2)
    fp, npr = model.prosody(hf, fm[:, ::2, None].transpose(1, 2))
    l_pro = (((fp - f0).abs() / 100.0 + (npr - n).abs() / 5.0) * fm).sum() / fm.sum()
    return l_dur, l_pro


@torch.no_grad()
def evaluate(model, rows, dev):
    """Teacher durations so frames align: median voiced pitch error and pitch-range ratio in semitones,
    duration error in frames per token."""
    err, rng, derr = [], [], []
    for r in rows:
        ids = torch.tensor([r["ids"]], device=dev)
        h, logd = model.encode(ids, torch.zeros_like(ids, dtype=torch.bool))
        dur = torch.tensor([r["dur"]], device=dev)
        derr.append((torch.expm1(logd).round().clamp(min=1) - dur).abs().float().mean().item())
        fp = model.prosody(expand(h, dur, int(dur.sum())))[0][0].cpu().numpy()
        mm = np.memmap(r["f0_bin"], dtype=np.float16, mode="r")
        ft = np.asarray(mm[r["f0_offset"]: r["f0_offset"] + len(fp)], dtype=np.float32)
        v = ft > 80
        st = lambda f: 12 * np.log2(np.maximum(f, 1.0))
        err.append(np.abs(st(fp) - st(ft))[v]); rng.append(st(fp)[v & (fp > 80)].std() / st(ft)[v].std())
    return float(np.median(np.concatenate(err))), float(np.mean(rng)), float(np.mean(derr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/dist/shard*.jsonl")
    ap.add_argument("--out", default="runs/prosody_v1")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--model", default="{}", help="ProsodyNet kwargs as JSON")
    ap.add_argument("--baseline", help="TinyKokoro checkpoint to score with the same metrics first")
    ap.add_argument("--init", help="start from this ProsodyNet checkpoint (fine-tuning to another voice)")
    a = ap.parse_args()
    dev = "cuda"
    if os.environ.get("HIBARI_GPU_FRAC"):  # share of GPU memory this process may use (user limit)
        torch.cuda.set_per_process_memory_fraction(float(os.environ["HIBARI_GPU_FRAC"]))
    torch.manual_seed(0)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    # same split as train_tiny.py
    rows = [json.loads(l) for f in sorted(glob.glob(a.index)) for l in open(f, encoding="utf-8")]
    rows = [r for r in rows if sum(r["dur"]) >= 66 and len(r["ids"]) <= 510]
    random.Random(1234).shuffle(rows)
    train_rows, eval_rows = rows[:-200], rows[-200:]

    if a.baseline:
        c = torch.load(a.baseline, map_location="cpu", weights_only=False)
        b = TinyKokoro(**c["kw"]).to(dev).eval(); b.load_state_dict(c["model"])
        e, r, d = evaluate(b, eval_rows, dev)
        print(f"[*] baseline {a.baseline}: pitch err {e:.2f} st  range {r:.2f}  dur err {d:.2f} frames", flush=True)
        del b

    kw = json.loads(a.model)
    model = ProsodyNet(**kw).to(dev)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location="cpu", weights_only=False)["model"])
        print(f"[*] init from {a.init}", flush=True)
    print(f"[*] ProsodyNet {sum(p.numel() for p in model.parameters())/1e6:.3f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.98), weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    dl = DataLoader(Rows(train_rows), batch_size=a.batch, shuffle=True, collate_fn=collate, num_workers=2,
                    drop_last=True, persistent_workers=True)
    step, best, t0 = 0, 1e9, time.time()
    while step < a.steps:
        for batch in dl:
            l_dur, l_pro = losses(model, batch, dev)
            opt.zero_grad(set_to_none=True)
            (l_dur + l_pro).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); step += 1
            if step % 2000 == 0 or step == a.steps:
                model.eval()
                e, r, d = evaluate(model, eval_rows, dev)
                model.train()
                tag = ""
                if e < best:
                    best = e; tag = " *"
                    torch.save({"model": model.state_dict(), "kw": kw, "step": step}, out / "best.pt")
                print(f"  step {step}  dur {l_dur.item():.4f}  pro {l_pro.item():.4f} | eval pitch err {e:.2f} st  "
                      f"range {r:.2f}  dur err {d:.2f} frames  {time.time()-t0:.0f}s{tag}", flush=True)
            if step >= a.steps:
                break


if __name__ == "__main__":
    main()
