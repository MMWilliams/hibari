"""Size and speed sheet: ours (export/v3_final int8 / fp32 + numpy host DSP) vs Paradee-8M (ONNX int8 / fp32),
Kokoro-7M-Distill and the Kokoro-82M teacher (torch, CPU). Writes docs/METRICS.md and runs/bench_all/results.json.

    python scripts/bench_all.py                 # ~10 min; waits for the CPU to go quiet before each timing run

Every measurement runs in a fresh child process: --child timing --threads N (all models, 20 sentences,
round-robin, best of 3) and --child mem --model NAME (import + load time, peak working set).
"""
import time; T0 = time.perf_counter()  # noqa: E702  (load time counts the imports)
import argparse, ctypes, glob, json, os, platform, random, subprocess, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, d) for d in ("kokoro_patched", "export", "training")]
OUT = os.path.join(ROOT, "runs", "bench_all")
SR = 24000
K7M = os.path.join(ROOT, "upstream", "kokoro-7m")
MODELS = {                                  # name: (label, kind, weights file, fp32 file for the parameter count)
    "tiny_int8": ("ours v3 40k + ProsodyNet int8", "tiny", "export/v3_final/tiny.int8.onnx", "export/v3_final/tiny.onnx"),
    "tiny_fp32": ("ours v3 40k + ProsodyNet fp32", "tiny", "export/v3_final/tiny.onnx", "export/v3_final/tiny.onnx"),
    "paradee_int8": ("Paradee-8M int8", "paradee", "upstream/paradee/onnx/paradee_int8.onnx", "upstream/paradee/onnx/paradee.onnx"),
    "paradee_fp32": ("Paradee-8M fp32", "paradee", "upstream/paradee/onnx/paradee.onnx", "upstream/paradee/onnx/paradee.onnx"),
    "kokoro7m": ("Kokoro-7M-Distill fp32", "kokoro", "upstream/kokoro-7m/kokoro_en_7m.pth", None),
    "kokoro82m": ("Kokoro-82M teacher fp32", "kokoro", None, None),
}
BASES = {"base_ort": "onnxruntime + numpy + scipy.fft imported, no model",
         "base_torch": "torch + kokoro imported, no model"}
SHORT = ["Hello, how are you today?", "The train leaves at noon.", "Thanks, see you tomorrow.",
         "It's cold outside this morning."]


# ---------------------------------------------------------------- measurement helpers (no psutil)
def peak_rss_mb():
    if sys.platform == "win32":
        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32)] + [
                (f, ctypes.c_size_t) for f in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                               "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                                               "PagefileUsage", "PeakPagefileUsage")]
        k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        k32.GetCurrentProcess.restype = ctypes.c_void_p                     # default int restype truncates the handle
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_uint32]
        pmc = PMC(cb=ctypes.sizeof(PMC))
        psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return pmc.PeakWorkingSetSize / 2 ** 20
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2 ** 20 if sys.platform == "darwin" else r / 1024


def cpu_load(secs=2.0):
    """System-wide busy fraction over `secs` (Windows GetSystemTimes; nan elsewhere)."""
    if sys.platform != "win32":
        return float("nan")
    def times():
        t = [ctypes.c_uint64() for _ in range(3)]                           # idle, kernel (incl. idle), user
        ctypes.windll.kernel32.GetSystemTimes(*(ctypes.byref(x) for x in t))
        return [x.value for x in t]
    a = times(); time.sleep(secs); b = times()
    idle, kern, user = (y - x for x, y in zip(a, b))
    return 100.0 * (1.0 - idle / (kern + user))


def quiet_cpu(limit=20.0, tries=6, wait=10.0):
    for _ in range(tries):
        load = cpu_load()
        if load <= limit:
            break
        print(f"  cpu load {load:.0f}% > {limit:.0f}%, waiting {wait:.0f} s"); time.sleep(wait)
    print(f"  cpu load {load:.0f}%")
    return load


def cpu_name():
    if sys.platform == "win32":
        return subprocess.check_output(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                                       text=True).strip()
    return platform.processor()


def onnx_params(path):
    import onnx
    m = onnx.load(path, load_external_data=False)
    used = {i for n in m.graph.node for i in n.input}
    return int(sum(np.prod(t.dims) for t in m.graph.initializer if t.name in used))


# ---------------------------------------------------------------- models
def load(name, threads):
    """-> synth(ids) -> (wav float32, network seconds), parameter count, weights file size in MB."""
    label, kind, path, fp32 = MODELS[name]
    if kind == "tiny":
        from infer_numpy import Tiny, wav
        t = Tiny(os.path.join(ROOT, path), threads)
        def synth(ids):
            t0 = time.perf_counter(); out = t.spec(ids); tn = time.perf_counter() - t0
            return wav(out), tn
    elif kind == "paradee":
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = so.inter_op_num_threads = threads
        s = ort.InferenceSession(os.path.join(ROOT, path), so, providers=["CPUExecutionProvider"])
        def synth(ids):
            t0 = time.perf_counter()
            y = s.run(None, {"input_ids": np.asarray([ids], np.int64), "speed": np.ones(1, np.float32)})[0][0]
            return y, time.perf_counter() - t0
    else:
        import torch
        torch.set_num_threads(threads)
        from kokoro import KModel
        if name == "kokoro7m":                        # as upstream/kokoro-7m/load_model.py builds it (voice af_heart)
            path = os.path.join(ROOT, path)
            m = KModel(repo_id="oddadmix/Kokoro-7M-Distill", config=os.path.join(K7M, "config.json"), model=path,
                       disable_complex=True)
            pack = torch.load(os.path.join(K7M, "af_heart.pt"), map_location="cpu", weights_only=True)
        else:
            from huggingface_hub import hf_hub_download
            m = KModel(repo_id="hexgrad/Kokoro-82M")
            path = hf_hub_download("hexgrad/Kokoro-82M", "kokoro-v1_0.pth")
            pack = torch.load(hf_hub_download("hexgrad/Kokoro-82M", "voices/af_heart.pt"), map_location="cpu",
                              weights_only=True)
        m.eval()
        def synth(ids):                               # KPipeline.infer: ref_s = pack[len(phonemes) - 1]; ids carry the two 0 pads
            t0 = time.perf_counter()
            y, _ = m.forward_with_tokens(torch.LongTensor([ids]), pack[len(ids) - 3])
            return y.numpy(), time.perf_counter() - t0
    params = onnx_params(os.path.join(ROOT, fp32)) if fp32 else sum(p.numel() for p in m.parameters())
    return synth, params, os.path.getsize(path) / 1e6


def sentences():
    """20 rows: 4 short phrases (misaki G2P), the 8 eval rows, 8 held-out rows spread over 2..10 s. Cached."""
    p = os.path.join(OUT, "sentences.json")
    if os.path.exists(p):
        return json.load(open(p, encoding="utf-8"))
    ev = json.load(open(os.path.join(ROOT, "runs", "tiny_v1", "eval_rows.json")))
    rows = [json.loads(l) for f in sorted(glob.glob(os.path.join(ROOT, "data", "dist", "shard*.jsonl")))
            for l in open(f, encoding="utf-8")]
    rows = [r for r in rows if sum(r["dur"]) >= 66 and len(r["ids"]) <= 510]
    random.Random(1234).shuffle(rows)
    held = rows[-200:]                                                      # the trainer's held-out split
    sel = []
    for tgt in (2, 2.5, 3, 3.5, 4, 7, 8, 10):
        r = min(held, key=lambda r: abs(r["samples"] / SR - tgt)); held.remove(r); sel.append(r)
    from kokoro import KPipeline
    vocab = json.load(open(os.path.join(ROOT, "export", "v3_final", "vocab.json"), encoding="utf-8"))
    pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=False)
    out = [{"text": t, "ids": [0, *[vocab[c] for c in next(pipe(t)).phonemes if c in vocab], 0]} for t in SHORT]
    out += [{"ids": r["ids"], "teacher_s": r["samples"] / SR} for r in (*ev, *sel)]
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(p, "w", encoding="utf-8"))
    return out


def tiny_split():
    """Parameter split of what is in the shipping graph (its dur/prosody heads are not exported)."""
    import torch
    from tiny_model import TinyKokoro, ProsodyNet
    c = torch.load(os.path.join(ROOT, "runs", "tiny_v3", "snap", "40k.pt"), map_location="cpu", weights_only=False)
    m = TinyKokoro(**c["kw"]); m.load_state_dict(c["model"])
    c = torch.load(os.path.join(ROOT, "runs", "prosody_v1", "best.pt"), map_location="cpu", weights_only=False)
    pn = ProsodyNet(**c["kw"]); pn.load_state_dict(c["model"])
    n = lambda *ms: sum(p.numel() for mm in ms for p in mm.parameters())
    return {"ProsodyNet": n(pn), "text encoder": n(m.text), "decoder (dec_in + 8 ConvNeXt + norm)": n(m.dec_in, m.dec, m.dec_norm),
            "output head": n(m.head)}


# ---------------------------------------------------------------- children
def child_timing(threads, sents):
    synth = {n: load(n, threads)[0] for n in MODELS}
    for f in synth.values():
        for s in sents[:2]:
            f(s["ids"])
    best = {n: [[1e9, 1e9, 0.0] for _ in sents] for n in MODELS}            # [wall s, network s, audio s]
    for _ in range(3):
        for i, s in enumerate(sents):
            for n, f in synth.items():
                t0 = time.perf_counter(); y, tn = f(s["ids"]); dt = time.perf_counter() - t0
                if dt < best[n][i][0]:
                    best[n][i] = [dt, tn, len(y) / SR]
    return best


def child_mem(name, sents):
    if name == "base_ort":
        import onnxruntime, scipy.fft  # noqa: F401
        return {"load_s": time.perf_counter() - T0, "peak_mb": peak_rss_mb()}
    if name == "base_torch":
        import torch; from kokoro import KModel  # noqa: F401,E702
        return {"load_s": time.perf_counter() - T0, "peak_mb": peak_rss_mb()}
    synth, params, mb = load(name, 1)
    r = {"load_s": time.perf_counter() - T0, "params": params, "file_mb": mb, "rss_after_load_mb": peak_rss_mb()}
    s = max(sents, key=lambda s: s.get("teacher_s", 0))                     # longest row -> peak activations
    t0 = time.perf_counter(); synth(s["ids"]); r["first_call_s"] = time.perf_counter() - t0
    r["peak_mb"] = peak_rss_mb()
    return r


# ---------------------------------------------------------------- parent
def run_child(args, threads=1):
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": str(threads),
           "MKL_NUM_THREADS": str(threads)}
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", *args], env=env, cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    for line in p.stdout.splitlines()[::-1]:
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    raise RuntimeError(f"child {args} failed:\n{p.stderr[-3000:]}")


def profile_dsp():
    p = subprocess.run([sys.executable, "scripts/profile_dsp.py", "export/v3_final/tiny.int8.onnx"], cwd=ROOT,
                       capture_output=True, text=True, env={**os.environ, "OMP_NUM_THREADS": "1"})
    return {k.strip(): float(v) for k, v in (l.rsplit(None, 1) for l in p.stdout.splitlines()[1:])}


def fmt_x(x):
    return f"{x:.0f}" if x >= 20 else f"{x:.1f}"


def write_metrics(R, sents):
    t1, t4, M = R["timing1"], R["timing4"], R["mem"]
    def agg(T, n):
        return [sum(b[k] for b in T[n]) for k in range(3)]                   # wall, net, audio
    L = [np.mean([t1[n][i][2] for n in MODELS]) for i in range(len(sents))]
    si = min(range(len(L)), key=lambda i: abs(L[i] - 1.5)); li = min(range(len(L)), key=lambda i: abs(L[i] - 8))
    lines = ["# Size and speed sheet", "",
             "| model | params | file MB | x realtime 1 thread (RTF) | x realtime 4 threads (RTF) | "
             "latency short / long ms (1 thread) | peak RSS MB | load s |", "|---|---|---|---|---|---|---|---|"]
    for n, (label, *_) in MODELS.items():
        w1, _, a1 = agg(t1, n); w4, _, a4 = agg(t4, n); m = M[n]
        lines.append(f"| {label} | {m['params'] / 1e6:.2f} M | {m['file_mb']:.2f} | {fmt_x(a1 / w1)} ({w1 / a1:.4f}) | "
                     f"{fmt_x(a4 / w4)} ({w4 / a4:.4f}) | {1000 * t1[n][si][0]:.0f} / {1000 * t1[n][li][0]:.0f} | "
                     f"{m['peak_mb']:.0f} | {m['load_s']:.2f} |")
    sp = R["split"]; tot = sum(sp.values())
    split = ", ".join(f"{k} {v / 1e6:.2f} M" for k, v in sp.items())
    ours = []
    for n in ("tiny_int8", "tiny_fp32"):
        for thr, T in (("1", t1), ("4", t4)):
            w, net, a = agg(T, n)
            ours.append(f"{n.split('_')[1]} {thr} thr: net {1000 * net / a:.2f} ms, host DSP {1000 * (w - net) / a:.2f} ms")
    dsp = ", ".join(f"{k} {v:.2f}" for k, v in R["dsp"].items() if k != "synth total")
    au = agg(t1, "tiny_int8")[2]
    notes = [
        f"Machine: {R['cpu']} (12 cores / 24 threads), {platform.system()} build {platform.version()}, "
        f"Python {platform.python_version()}, onnxruntime {R['versions']['onnxruntime']}, torch {R['versions']['torch']}, "
        f"numpy {R['versions']['numpy']}, scipy {R['versions']['scipy']}.",
        "System CPU load (all 24 threads) measured over 2 s before each run (threshold 20%, up to 6 retries of 10 s) and "
        "again after it: " + ", ".join(f"{k} {'/'.join(f'{x:.0f}' for x in v)}%" for k, v in R["load"].items()) + ". "
        "Other applications were running; nothing was stopped.",
        f"Speed: {len(sents)} phoneme-id sentences ({au:.1f} s of audio from ours int8): 4 short phrases (misaki G2P), "
        f"the 8 runs/tiny_v1/eval_rows.json rows, 8 held-out corpus rows spread over 2-10 s. All models in one process, "
        "warm (2 sentences), rounds x sentences x models round-robin, best of 3 rounds per sentence; x realtime = audio s / "
        "summed best wall s, RTF = its inverse. ONNX: intra-op threads = N; torch: torch.set_num_threads(N); ours includes "
        "the numpy host DSP (single-threaded at both thread counts).",
        f"Latency: best-of-3 wall time for sentence {si} ({L[si]:.2f} s of audio, {len(sents[si]['ids'])} ids) and sentence "
        f"{li} ({L[li]:.2f} s, {len(sents[li]['ids'])} ids), 1 thread. The {len(SHORT)} short phrases came out at "
        f"{min(L[:len(SHORT)]):.1f}-{max(L[:len(SHORT)]):.1f} s.",
        f"Ours, where the time goes per second of audio: {'; '.join(ours)}. Host DSP stages (scripts/profile_dsp.py, int8, "
        f"ms per s of audio): {dsp}.",
        f"Ours parameters ({tot / 1e6:.2f} M in the graph): {split}. TinyKokoro's own duration/prosody heads are not exported.",
        "Parameters: ONNX models counted from the fp32 graph initializers (int8 holds the same weights); torch models from "
        "model.parameters(). File MB = the weights file only; the Kokoro voice pack af_heart.pt adds 0.52 MB.",
        "Memory and load: each model in a fresh python process, 1 thread. Load = imports + weights (interpreter start "
        "excluded), peak RSS = Windows peak working set after loading and synthesizing the longest sentence (14.3 s). "
        + "; ".join(f"{v}: {M[k]['peak_mb']:.0f} MB, {M[k]['load_s']:.2f} s" for k, v in BASES.items()) + ". "
        + "RSS right after load / first-call time: "
        + ", ".join(f"{n} {M[n]['rss_after_load_mb']:.0f} MB / {M[n]['first_call_s']:.2f} s" for n in MODELS) + ".",
        "Kokoro-7M-Distill is built as its load_model.py does (config.json, kokoro_en_7m.pth, disable_complex=True) through "
        "the repo's kokoro_patched (a superset of the one shipped with it), voice af_heart.pt; the teacher is the stock "
        "KModel (torch.stft path), voice af_heart. Both are driven with forward_with_tokens on the same phoneme ids. "
        "Paradee gets input_ids + speed 1.0.",
    ]
    md = "\n".join(lines) + "\n\nNotes\n" + "\n".join(f"- {n}" for n in notes) + "\n"
    with open(os.path.join(ROOT, "docs", "METRICS.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", choices=["timing", "mem"])
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--model")
    a = ap.parse_args()
    if a.child:
        sents = json.load(open(os.path.join(OUT, "sentences.json"), encoding="utf-8"))
        r = child_timing(a.threads, sents) if a.child == "timing" else child_mem(a.model, sents)
        print("RESULT " + json.dumps(r))
        return
    sents = sentences()
    import onnxruntime, scipy, torch
    R = {"cpu": cpu_name(), "load": {}, "mem": {},
         "versions": {"onnxruntime": onnxruntime.__version__, "torch": torch.__version__, "numpy": np.__version__,
                      "scipy": scipy.__version__}}
    for thr in (1, 4):
        print(f"timing, {thr} thread(s)")
        before = quiet_cpu()
        R[f"timing{thr}"] = run_child(["timing", "--threads", str(thr)], thr)
        R["load"][f"timing {thr} thread"] = [before, cpu_load()]
    print("memory / load time")
    R["load"]["memory"] = [quiet_cpu()]
    for n in (*MODELS, *BASES):
        R["mem"][n] = run_child(["mem", "--model", n])
    R["split"] = tiny_split()
    R["dsp"] = profile_dsp()
    os.makedirs(OUT, exist_ok=True)
    json.dump(R, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    write_metrics(R, sents)


if __name__ == "__main__":
    main()
