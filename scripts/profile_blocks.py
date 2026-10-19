"""Per-block CPU time for a Kokoro-architecture model, to see where synthesis time goes."""
import sys, time, os, torch
UP = os.path.join(os.path.dirname(__file__), "..", "upstream", "kokoro-7m")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kokoro_patched"))
from kokoro import KModel, KPipeline

torch.set_num_threads(4)
cfg, wts = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else (f"{UP}/config.json", f"{UP}/kokoro_en_7m.pth")
m = KModel(repo_id="hexgrad/Kokoro-82M", config=cfg, model=(wts if wts != "random" else os.path.join(os.path.dirname(cfg), "empty.pth")), disable_complex=True).eval()
print(f"params {sum(p.numel() for p in m.parameters())/1e6:.2f}M")
p = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=False)
text = "The quick brown fox jumps over the lazy dog, and then it runs far away into the quiet green forest."
ps = next(p(text, voice="af_heart")).phonemes
ids = torch.LongTensor([[0, *[m.vocab[c] for c in ps if c in m.vocab], 0]])
ref = torch.load(f"{UP}/af_msa.pt", weights_only=True)[ids.shape[1] - 3]

T = {}
def timed(name, mod):
    f = mod.forward
    def w(*a, **k):
        t = time.perf_counter(); r = f(*a, **k); T[name] = T.get(name, 0) + time.perf_counter() - t; return r
    mod.forward = w
for n in ("bert", "bert_encoder", "text_encoder"):
    timed(n, getattr(m, n))
timed("predictor.text_encoder", m.predictor.text_encoder)
dec = m.decoder
timed("decoder.generator", dec.generator)
timed("decoder.total", dec)

with torch.no_grad():
    for _ in range(3):
        a, d = m.forward_with_tokens(ids, ref, 1.0)
    T.clear()
    N = 10
    t0 = time.perf_counter()
    for _ in range(N):
        a, d = m.forward_with_tokens(ids, ref, 1.0)
    tot = (time.perf_counter() - t0) / N
dur = a.shape[-1] / 24000
print(f"audio {dur:.2f}s  synth {tot*1000:.1f} ms  RTF {tot/dur:.4f}")
for k, v in T.items():
    print(f"  {k:24s} {v/N*1000:7.1f} ms  {100*v/N/tot:5.1f}%")

if os.environ.get("DEEP"):
    T.clear()
    g = m.decoder.generator
    for n, c in list(g.named_children()):
        timed("gen." + n, c) if not isinstance(c, torch.nn.ModuleList) else [timed(f"gen.{n}.{i}", cc) for i, cc in enumerate(c)]
    for n, c in m.decoder.named_children():
        if n != "generator":
            timed("dec." + n, c) if not isinstance(c, torch.nn.ModuleList) else [timed(f"dec.{n}.{i}", cc) for i, cc in enumerate(c)]
    with torch.no_grad():
        for _ in range(N):
            m.forward_with_tokens(ids, ref, 1.0)
    for k, v in sorted(T.items(), key=lambda kv: -kv[1]):
        print(f"  {k:24s} {v/N*1000:7.1f} ms  {100*v/N/tot:5.1f}%")
