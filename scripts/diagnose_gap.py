"""Where does the student lose to the teacher? UTMOS on held-out rows for each snapshot with
progressively more teacher help: free inference, teacher durations, teacher durations + F0/energy.
If teacher-forced audio is still far below the teacher, the decoder (vocoder) is the bottleneck.

    python scripts/diagnose_gap.py 60k 150k --n 50
"""
import argparse, glob, json, os, random, sys
import numpy as np, soundfile as sf, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "training"))
from tiny_model import TinyKokoro, spec_to_wav, SR  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("snaps", nargs="+")
ap.add_argument("--run", default="runs/tiny_v1")
ap.add_argument("--n", type=int, default=50)
a = ap.parse_args()
out = f"{a.run}/diag"; os.makedirs(out, exist_ok=True)

# the trainer's held-out split
rows = [json.loads(l) for f in sorted(glob.glob("data/dist/shard*.jsonl")) for l in open(f, encoding="utf-8")]
rows = [r for r in rows if sum(r["dur"]) >= 66 and len(r["ids"]) <= 510]
random.Random(1234).shuffle(rows)
rows = rows[-200:][: a.n]

def teacher(r):
    w = np.memmap(r["bin"], dtype=np.int16, mode="r")[r["offset"]: r["offset"] + r["samples"]]
    T = 2 * sum(r["dur"])
    fn = np.memmap(r["f0_bin"], dtype=np.float16, mode="r")[r["f0_offset"]: r["f0_offset"] + 2 * T]
    return w.astype(np.float32) / 32767, torch.tensor(fn[:T], dtype=torch.float32)[None].cuda(), \
        torch.tensor(fn[T:], dtype=torch.float32)[None].cuda()

utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).cuda().eval()
def mos(y):
    with torch.no_grad():
        return float(utmos(torch.as_tensor(y, dtype=torch.float32)[None].cuda(), SR))

res = {"teacher": np.mean([mos(teacher(r)[0]) for r in rows])}
for snap in a.snaps:
    c = torch.load(f"{a.run}/snap/{snap}.pt", map_location="cpu", weights_only=False)
    m = TinyKokoro(**c["kw"]).cuda().eval(); m.load_state_dict(c["model"])
    s = {"free": [], "t_dur": [], "t_dur_pro": []}
    with torch.no_grad():
        for i, r in enumerate(rows):
            _, f0, n = teacher(r)
            ids = torch.tensor([r["ids"]]).cuda()
            h, logd = m.encode(ids, torch.zeros_like(ids, dtype=torch.bool))
            free = torch.round(torch.expm1(logd)).clamp(min=1).long()
            hf_free = m.expand(h, free)
            hf = m.expand(h, torch.tensor([r["dur"]]).cuda())
            for k, (x, p) in {"free": (hf_free, m.prosody(hf_free)), "t_dur": (hf, m.prosody(hf)),
                              "t_dur_pro": (hf, (f0, n))}.items():
                y = spec_to_wav(*m.decode(x, *p))[0].cpu().numpy()
                s[k].append(mos(y))
                if i < 3:
                    sf.write(f"{out}/{snap}_{i}_{k}.wav", y, SR)
    res.update({f"{snap}_{k}": np.mean(v) for k, v in s.items()})

for k, v in res.items():
    print(f"{k:22s} UTMOS {v:.3f}")
json.dump(res, open(f"{out}/results.json", "w"), indent=1)
