"""TinyKokoro inference with onnxruntime + numpy only: ids -> ONNX (logmag, phase) -> numpy iSTFT.

    python export/infer_numpy.py --model export/tiny.int8.onnx --ids 0,50,83,... --out x.wav
    python export/infer_numpy.py --phonemes "hˈɛlO" --out x.wav      (needs vocab.json next to the model)
    python export/infer_numpy.py --bench runs/tiny_v1/eval_rows.json   (RTF, 1 thread, fp32 vs int8)
"""
import argparse, json, os, time
import numpy as np

N_FFT, HOP, SR = 1200, 300, 24000


def istft(logmag, phase, n_fft=N_FFT, hop=HOP):
    """Matches torch.istft(center=True, hann window, length=F*hop): irfft per frame, overlap-add,
    divide by the summed squared window, trim n_fft/2 on both ends. n_fft = 4*hop, so output block j
    is the sum of quarter k of frame j-k for k in 0..3: a reshape and four shifted adds, no loop."""
    mag = np.exp(np.minimum(logmag, 7.0), dtype=np.float32)
    spec = (mag * np.cos(phase) + 1j * (mag * np.sin(phase))).astype(np.complex64)   # [bins, F]
    F, K = spec.shape[1], n_fft // hop
    win = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)).astype(np.float32)   # periodic hann
    frames = (np.fft.irfft(spec.T, n=n_fft, axis=1).astype(np.float32) * win).reshape(F, K, hop)
    nb = F + K - 1
    y = np.zeros((nb, hop), np.float32); wsum = np.zeros((nb, hop), np.float32)
    w2 = (win * win).reshape(K, hop)
    for k in range(K):
        y[k: k + F] += frames[:, k]
        wsum[k: k + F] += w2[k]
    y = (y / np.maximum(wsum, 1e-11)).reshape(-1)
    half = n_fft // 2
    return y[half: half + F * hop]


class Tiny:
    def __init__(self, model, threads=1):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = so.inter_op_num_threads = threads
        self.sess = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
        vp = os.path.join(os.path.dirname(model), "vocab.json")
        self.vocab = json.load(open(vp, encoding="utf-8")) if os.path.exists(vp) else None

    def spec(self, ids):
        lm, ph = self.sess.run(None, {"ids": np.asarray([ids], np.int64)})
        return lm[0], ph[0]

    def __call__(self, ids):
        return istft(*self.spec(ids))

    def phonemes_to_ids(self, ph):
        return [0, *[self.vocab[c] for c in ph if c in self.vocab], 0]


def bench(rows_path, models, threads=1):
    rows = json.load(open(rows_path))
    for mp in models:
        t = Tiny(mp, threads)
        for r in rows[:2]:
            t(r["ids"])                                               # warm-up
        tot_t = tot_net = tot_a = 0.0
        for r in rows:
            t0 = time.perf_counter(); lm, ph = t.spec(r["ids"]); t1 = time.perf_counter()
            y = istft(lm, ph); t2 = time.perf_counter()
            tot_net += t1 - t0; tot_t += t2 - t0; tot_a += len(y) / SR
        print(f"{mp}: {os.path.getsize(mp)/1e6:.2f} MB  {threads} thread  RTF {tot_t/tot_a:.4f} "
              f"(net {tot_net/tot_a:.4f}, istft {(tot_t-tot_net)/tot_a:.4f})  over {tot_a:.1f} s audio")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="export/tiny.onnx")
    ap.add_argument("--ids"); ap.add_argument("--phonemes"); ap.add_argument("--out", default="out.wav")
    ap.add_argument("--bench", help="eval_rows.json: time fp32 + int8 on 1 thread")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    if a.bench:
        bench(a.bench, [a.model, a.model.replace(".onnx", ".int8.onnx")], a.threads)
    else:
        t = Tiny(a.model, a.threads)
        ids = [int(x) for x in a.ids.split(",")] if a.ids else t.phonemes_to_ids(a.phonemes)
        y = t(ids)
        import soundfile as sf
        sf.write(a.out, y, SR)
        print(f"{a.out}: {len(y)/SR:.2f} s")
