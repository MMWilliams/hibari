"""Per-discriminator fwd+bwd cost at batch 16 x 1.6 s."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))
import _env  # noqa
import torch
from styletts2_gan.discriminators import MultiPeriodDiscriminator, MultiResSpecDiscriminator
dev = "cuda"; torch.backends.cudnn.benchmark = True
mpd, msd = MultiPeriodDiscriminator().to(dev), MultiResSpecDiscriminator().to(dev)
y = torch.randn(16, 1, 38400, device=dev); yh = torch.randn(16, 1, 38400, device=dev, requires_grad=True)
def t(name, f, amp):
    for _ in range(3):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp): o = f()
        o.backward()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(10):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp): o = f()
        o.backward()
    torch.cuda.synchronize(); print(f"{name:28s} amp={amp}  {(time.perf_counter()-t0)*100:6.1f} ms", flush=True)
for amp in (False, True):
    for i, d in enumerate(mpd.discriminators):
        t(f"mpd p={d.period}", lambda: sum(x.float().mean() for x in d(y)[1]) + sum(x.float().mean() for x in d(yh)[1]), amp)
    for i, d in enumerate(msd.discriminators):
        t(f"mrd fft={d.fft_size} hop={d.shift_size}", lambda: sum(x.float().mean() for x in d(y)[1]) + sum(x.float().mean() for x in d(yh)[1]), amp)
