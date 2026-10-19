"""Time one GAN-phase training step (D step + G step) under several configurations on GPU 0."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))
import _env  # noqa
import torch, torch.nn.functional as F
from tiny_model import TinyKokoro, spec_to_wav
from styletts2_gan.discriminators import MultiPeriodDiscriminator, MultiResSpecDiscriminator
from styletts2_gan import losses_styletts2 as L

dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
B, C = 16, 64
model = TinyKokoro().to(dev)
VOCOS = dict(fft_sizes=[1024, 2048, 512], hop_sizes=[256, 512, 128], win_lengths=[1024, 2048, 512])
mpd = MultiPeriodDiscriminator().to(dev)
msd = MultiResSpecDiscriminator(**(VOCOS if "--vocos" in sys.argv else {})).to(dev)
if "--cl" in sys.argv:
    import styletts2_gan.discriminators as D; D.CHANNELS_LAST = True
    mpd.to(memory_format=torch.channels_last); msd.to(memory_format=torch.channels_last)
gen_loss, disc_loss = L.GeneratorLoss(mpd, msd).to(dev), L.DiscriminatorLoss(mpd, msd).to(dev)
opt_g = torch.optim.AdamW(model.parameters(), 2e-4)
opt_d = torch.optim.AdamW(list(mpd.parameters()) + list(msd.parameters()), 2e-4)
hf = torch.randn(B, 128, C, device=dev); f0 = torch.rand(B, 2 * C, device=dev) * 200
n = torch.randn(B, 2 * C, device=dev); y = torch.randn(B, 1, C * 600, device=dev) * 0.1

def step(gan_samples, amp, tprls, n_mrd):
    y1 = y[..., :gan_samples]
    opt_d.zero_grad(set_to_none=True)
    lm, ph = model.decode(hf, f0, n)
    yh = spec_to_wav(lm, ph)[:, None]
    yh1 = yh[..., :gan_samples]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        d = disc_loss(y1, yh1.detach())
    d.backward(); opt_d.step()
    opt_g.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        g = gen_loss(y1, yh1)
    (g + yh.abs().mean()).backward(); opt_g.step()

def bench(name, **kw):
    L.USE_TPRLS = kw.pop("tprls", True)
    nm = kw.pop("n_mrd", 3)
    msd.discriminators = msd.discriminators[:nm] if nm < 3 else msd.discriminators
    for _ in range(3): step(tprls=L.USE_TPRLS, n_mrd=nm, **kw)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(10): step(tprls=L.USE_TPRLS, n_mrd=nm, **kw)
    torch.cuda.synchronize(); dt = (time.perf_counter() - t) / 10
    print(f"{name:40s} {dt*1000:7.1f} ms/step  {1/dt:5.1f} it/s", flush=True)

full = C * 600
torch.backends.cudnn.benchmark = True
bench("bf16 -tprls 1.6s", gan_samples=full, amp=True, tprls=False)
bench("bf16 tprls 0.8s", gan_samples=full // 2, amp=True, tprls=True)
bench("bf16 -tprls 0.8s", gan_samples=full // 2, amp=True, tprls=False)
