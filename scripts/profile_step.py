import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))
import _env  # noqa
import torch
from torch.profiler import profile, ProfilerActivity
exec(open("scripts/profile_gan.py").read().split("full = C * 600")[0])
L.USE_TPRLS = False
full = C * 600
torch.backends.cudnn.benchmark = True
for _ in range(3): step(full, True, False, 3)
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
    for _ in range(3): step(full, True, False, 3)
    torch.cuda.synchronize()
print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=25))
