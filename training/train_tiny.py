"""Distill Kokoro-82M (af_heart) into TinyKokoro.

Everything the student needs is taken from the teacher's own forward pass: durations (so the
alignment is exact), F0 and energy curves (direct prosody targets, teacher-forced into the
decoder), and audio. Adversarial training (MPD + multi-resolution spectrogram discriminators)
is required: magnitude losses alone leave phase unconstrained and the audio goes to static.
"""
import _env  # noqa: F401
import argparse, glob, json, math, os, random, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, torchaudio, soundfile as sf
from torch.utils.data import Dataset, DataLoader

from tiny_model import TinyKokoro, spec_to_wav, HOP, SR
from styletts2_gan.discriminators import MultiPeriodDiscriminator, MultiResSpecDiscriminator
from styletts2_gan import losses_styletts2 as L
from styletts2_gan.losses_styletts2 import GeneratorLoss, DiscriminatorLoss, MultiResolutionSTFTLoss

SPF = 600   # samples per 40 Hz duration frame


class Dist(Dataset):
    def __init__(self, rows):
        self.rows, self._mm = rows, {}

    def _m(self, p, dt):
        if p not in self._mm:
            self._mm[p] = np.memmap(p, dtype=dt, mode="r")
        return self._mm[p]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        a = np.asarray(self._m(r["bin"], np.int16)[r["offset"]: r["offset"] + r["samples"]],
                       dtype=np.float32) / 32767.0
        T = 2 * sum(r["dur"])
        fn = np.asarray(self._m(r["f0_bin"], np.float16)[r["f0_offset"]: r["f0_offset"] + 2 * T],
                        dtype=np.float32)
        return r["ids"], r["dur"], a, fn[:T], fn[T:]


def collate(batch):
    L = max(len(b[0]) for b in batch)
    ids = torch.zeros(len(batch), L, dtype=torch.long)
    dur = torch.zeros(len(batch), L, dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, : len(b[0])] = torch.tensor(b[0])
        dur[i, : len(b[1])] = torch.tensor(b[1])
    return ids, dur, [torch.from_numpy(b[2]) for b in batch], \
        [torch.from_numpy(b[3]) for b in batch], [torch.from_numpy(b[4]) for b in batch]


class LogMel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(SR, n_fft=1024, hop_length=256, n_mels=100,
                                                        power=1.0)

    def forward(self, x):
        return torch.log(self.mel(x).clamp(min=1e-5))


def silence_loss(y, y_hat, thresh=1e-3):
    """Hinge on output RMS where the teacher is silent: a small frame-rate vocoder floors at an
    audible hiss that log-magnitude losses barely penalise (Kokoro-7M's measured failure)."""
    H = SR // 100
    n = y.shape[-1] // H
    yr = y[..., : n * H].reshape(y.shape[0], n, H).pow(2).mean(-1).add(1e-12).sqrt()
    hr = y_hat[..., : n * H].reshape(y.shape[0], n, H).pow(2).mean(-1).add(1e-12).sqrt()
    sil = (yr < thresh).float()
    return (F.relu(hr - yr) * sil).sum() / sil.sum().clamp(min=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/dist/shard*.jsonl")
    ap.add_argument("--out", default="runs/tiny_v1")
    ap.add_argument("--steps", type=int, default=150000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=64, help="40 Hz frames per training crop (1.6 s)")
    ap.add_argument("--gan-crop", type=int, default=32,
                    help="40 Hz frames the discriminators see (random sub-window of --crop)")
    ap.add_argument("--tprls", action="store_true", help="StyleTTS2 relativistic GAN term (slower)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--disc-warmup", type=int, default=2000)
    ap.add_argument("--mel-w", type=float, default=45.0)
    ap.add_argument("--stft-w", type=float, default=2.0)
    ap.add_argument("--sil-w", type=float, default=100.0)
    ap.add_argument("--dur-w", type=float, default=2.0)
    ap.add_argument("--pro-w", type=float, default=2.0)
    ap.add_argument("--model", default="{}", help="TinyKokoro kwargs as JSON")
    ap.add_argument("--snap-every", type=int, default=10000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--init", help="copy same-shape weights from this checkpoint (e.g. a narrower run's text side)")
    a = ap.parse_args()

    dev = "cuda"
    torch.manual_seed(0); random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    out = Path(a.out); (out / "snap").mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for f in sorted(glob.glob(a.index)) for l in open(f, encoding="utf-8")]
    rows = [r for r in rows if sum(r["dur"]) >= a.crop + 2 and len(r["ids"]) <= 510]
    random.Random(1234).shuffle(rows)
    train_rows, eval_rows = rows[:-200], rows[-200:]
    json.dump(eval_rows[:8], open(out / "eval_rows.json", "w"))
    hours = sum(r["samples"] for r in train_rows) / SR / 3600
    print(f"[*] train {len(train_rows):,} rows ({hours:.1f} h) | held out {len(eval_rows)}", flush=True)

    kw = json.loads(a.model)
    model = TinyKokoro(**kw).to(dev)
    print(f"[*] student {sum(p.numel() for p in model.parameters())/1e6:.3f}M", flush=True)
    # Vocos hops (fft/4) instead of UnivNet's 120/240/50: 2-2.5x fewer MRD frames
    mpd = MultiPeriodDiscriminator().to(dev)
    msd = MultiResSpecDiscriminator(fft_sizes=[1024, 2048, 512], hop_sizes=[256, 512, 128],
                                    win_lengths=[1024, 2048, 512]).to(dev)
    L.USE_TPRLS = a.tprls
    gen_loss, disc_loss = GeneratorLoss(mpd, msd).to(dev), DiscriminatorLoss(mpd, msd).to(dev)
    stft_loss, logmel = MultiResolutionSTFTLoss().to(dev), LogMel().to(dev)

    opt_g = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.8, 0.99), weight_decay=1e-2)
    opt_d = torch.optim.AdamW(list(mpd.parameters()) + list(msd.parameters()), lr=a.lr,
                              betas=(0.8, 0.99), weight_decay=1e-2)
    sch_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, a.steps, eta_min=a.lr * 0.1)
    sch_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, a.steps, eta_min=a.lr * 0.1)

    step = 0
    ck = out / "last.pt"
    if a.resume and ck.exists():
        c = torch.load(ck, map_location="cpu", weights_only=False)
        for k, o in (("model", model), ("mpd", mpd), ("msd", msd), ("opt_g", opt_g),
                     ("opt_d", opt_d), ("sch_g", sch_g), ("sch_d", sch_d)):
            o.load_state_dict(c[k])
        step = c["step"]
        print(f"[*] resumed at {step}", flush=True)
    elif a.init:
        src = torch.load(a.init, map_location="cpu", weights_only=False)["model"]
        own = model.state_dict()
        keep = {k: v for k, v in src.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(keep, strict=False)
        print(f"[*] init {len(keep)}/{len(own)} tensors from {a.init}", flush=True)

    dl = DataLoader(Dist(train_rows), batch_size=a.batch, shuffle=True, collate_fn=collate,
                    num_workers=4, drop_last=True, pin_memory=True, persistent_workers=True)
    C = a.crop

    def run(batch):
        ids, dur, auds, f0s, ns = batch
        ids, dur = ids.to(dev), dur.to(dev)
        pad = ids.new_zeros(ids.shape, dtype=torch.bool)
        for i, d in enumerate(dur):
            pad[i, int((d > 0).sum()):] = True
        h, logd = model.encode(ids, pad)
        valid = (~pad).float()
        l_dur = ((logd - torch.log1p(dur.float())).abs() * valid).sum() / valid.sum()

        hf_c, f0_c, n_c, y = [], [], [], []
        for i in range(ids.shape[0]):
            frames = int(dur[i].sum())
            s = random.randint(0, frames - C)
            hf = torch.repeat_interleave(h[i], dur[i], dim=-1)           # [C,frames]
            hf_c.append(hf[:, s: s + C])
            f0_c.append(f0s[i][2 * s: 2 * (s + C)]); n_c.append(ns[i][2 * s: 2 * (s + C)])
            y.append(auds[i][s * SPF: (s + C) * SPF])
        hf_c = torch.stack(hf_c)
        f0_t, n_t = torch.stack(f0_c).to(dev), torch.stack(n_c).to(dev)
        y = torch.stack(y).to(dev)
        # prosody sees the crop with 3x k7 receptive field; edge frames are a small minority
        f0_p, n_p = model.prosody(hf_c)
        l_pro = (f0_p - f0_t).abs().mean() / 100.0 + (n_p - n_t).abs().mean() / 5.0
        logmag, phase = model.decode(hf_c, f0_t, n_t)                    # teacher-forced prosody
        y_hat = spec_to_wav(logmag, phase)
        return y, y_hat, l_dur, l_pro

    if a.probe:
        y, y_hat, l_dur, l_pro = run(next(iter(dl)))
        print(f"[+] probe y {tuple(y.shape)} y_hat {tuple(y_hat.shape)} dur {l_dur.item():.3f} "
              f"pro {l_pro.item():.3f}", flush=True)
        return

    G = a.gan_crop * SPF
    idx = torch.arange(G, device=dev)

    def gan_window(y, y_hat):
        """Discriminators get a random 0.8 s sub-window per sample: the GAN step was 85% of the
        step time at 1.6 s, and HiFi-GAN/Vocos train their discriminators on <1 s segments."""
        off = torch.randint(0, y.shape[-1] - G + 1, (y.shape[0], 1), device=dev)
        ix = (off + idx)[:, None]                                        # [B,1,G]
        return y[:, None].gather(2, ix), y_hat[:, None].gather(2, ix)

    model.train(); t0 = time.time(); acc = {}; t_log = time.time(); s_log = step
    while step < a.steps:
        for batch in dl:
            if step >= a.steps:
                break
            y, y_hat, l_dur, l_pro = run(batch)
            gan = step >= a.disc_warmup

            if gan:
                y1, yh1 = gan_window(y, y_hat)
                opt_d.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    d_l = disc_loss(y1, yh1.detach())
                d_l.backward()
                torch.nn.utils.clip_grad_norm_(list(mpd.parameters()) + list(msd.parameters()), 10.0)
                opt_d.step()
            else:
                d_l = torch.zeros((), device=dev)

            opt_g.zero_grad(set_to_none=True)
            l_mel = F.l1_loss(logmel(y_hat), logmel(y))
            l_stft = stft_loss(y_hat, y)
            l_sil = silence_loss(y, y_hat)
            if gan:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    l_g = gen_loss(y1, yh1)
            else:
                l_g = torch.zeros((), device=dev)
            loss = (a.mel_w * l_mel + a.stft_w * l_stft + a.sil_w * l_sil + a.dur_w * l_dur
                    + a.pro_w * l_pro + l_g)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt_g.step(); sch_g.step(); sch_d.step()
            step += 1

            for k, v in (("mel", l_mel), ("stft", l_stft), ("sil", l_sil), ("dur", l_dur),
                         ("pro", l_pro), ("gen", l_g), ("disc", d_l)):
                acc[k] = acc.get(k, 0.0) + v.detach()                   # no per-step GPU sync
            if step % 100 == 0:
                now = time.time()
                print(f"  step {step}/{a.steps}  " + "  ".join(f"{k} {float(v)/100:.4f}" for k, v in acc.items())
                      + f"  {(step-s_log)/(now-t_log):.1f} it/s", flush=True)
                acc = {}; t_log, s_log = now, step
            if step % 2000 == 0:
                tmp = out / "last.pt.tmp"
                torch.save({"model": model.state_dict(), "mpd": mpd.state_dict(), "msd": msd.state_dict(),
                            "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                            "sch_g": sch_g.state_dict(), "sch_d": sch_d.state_dict(),
                            "step": step, "kw": kw}, tmp)
                os.replace(tmp, ck)
            if step % a.snap_every == 0:
                # the best checkpoint is usually mid-run while loss keeps falling: keep them all
                torch.save({"model": model.state_dict(), "kw": kw, "step": step},
                           out / "snap" / f"{step//1000}k.pt")
                model.eval()
                with torch.no_grad():
                    for j, r in enumerate(eval_rows[:3]):
                        lm, ph = model.infer(torch.tensor([r["ids"]], device=dev))
                        sf.write(out / "snap" / f"{step//1000}k_eval{j}.wav",
                                 spec_to_wav(lm, ph)[0].cpu().numpy(), SR)
                model.train()
    print(f"[+] done at {step}", flush=True)


if __name__ == "__main__":
    main()
