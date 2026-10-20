"""Out-of-vocabulary words: a 0.65M-parameter character transformer (assets/oov.onnx) trained on
espeak-ng's readings of 225k words, so unknown words sound the way the reference pipeline read them.
One ONNX call per word; the output is CTC-style (collapse repeats, drop blanks)."""
import json, os
import numpy as np

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")


class OOV:
    def __init__(self, path=None, session=None):
        path = path or os.path.join(ASSETS, "oov.onnx")
        with open(path[:-5] + ".json", encoding="utf-8") as f:
            meta = json.load(f)
        self.cin = {c: i + 1 for i, c in enumerate(meta["in"])}
        self.cout = {i + 1: c for i, c in enumerate(meta["out"])}
        self.maxlen = meta["maxlen"]
        if session is None:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = so.inter_op_num_threads = 1
            so.log_severity_level = 3
            session = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.sess = session

    def __call__(self, word):
        ids = [self.cin[c] for c in word if c in self.cin][: self.maxlen]
        if not ids:
            return None
        out = self.sess.run(None, {"chars": np.asarray([ids], np.int64)})[0][0]
        ps, prev = [], 0
        for i in out.tolist():
            if i != prev and i != 0:
                ps.append(self.cout[i])
            prev = i
        return "".join(ps)
