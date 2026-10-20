"""hibari: text to speech.

    from hibari import Hibari
    tts = Hibari()
    audio = tts("Hello there.")          # float32 numpy array, 24 kHz
    tts.save("Hello there.", "hello.wav")
    for chunk in tts.stream(long_text): ...   # one array per sentence, as soon as each is ready
"""
import json, os, re, time, wave
import numpy as np

from . import dsp

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
SAMPLE_RATE = dsp.SR
MAX_PHONEMES = 400                     # sentences longer than this are split at clause punctuation
_SENT_END = re.compile(r'(?<=[.!?…])["”’)\]]*\s+(?=\S)')
_CLAUSE = re.compile(r"(?<=[,;:—])\s+")
with open(os.path.join(ASSETS, "normalize.json"), encoding="utf-8") as _f:
    _TITLES = set(json.load(_f)["titles"])


def _ends_sentence(para, end, nxt):
    """False for a period after a title or an initial ("Dr. Smith", "J. K. Rowling") or before a lowercase word."""
    if nxt.islower():
        return False
    if para[end - 1] != ".":
        return True
    word = para[para.rfind(" ", 0, end) + 1: end - 1].lstrip("(\"“'‘")
    return not (word in _TITLES or (len(word) == 1 and word.isupper()))


class Hibari:
    """Loads the ONNX model and the G2P once; call the instance to synthesize.

    threads: onnxruntime intra-op threads (1 is enough; the model runs ~130x realtime on one core).
    model: path to another ONNX file with the same inputs/outputs (e.g. the fp32 export).
    """
    sample_rate = SAMPLE_RATE

    def __init__(self, model=None, threads=1, g2p=None):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = so.inter_op_num_threads = threads
        so.log_severity_level = 3
        self.model_path = model or os.path.join(ASSETS, "hibari.onnx")
        self.sess = ort.InferenceSession(self.model_path, so, providers=["CPUExecutionProvider"])
        with open(os.path.join(ASSETS, "vocab.json"), encoding="utf-8") as f:
            self.vocab = json.load(f)
        self._g2p = g2p

    @property
    def g2p(self):
        if self._g2p is None:
            from .g2p import G2P
            self._g2p = G2P()
        return self._g2p

    # -- text -------------------------------------------------------------------------------
    @staticmethod
    def split_sentences(text):
        """Sentences (paragraph breaks and end punctuation), each a string the model sees at once."""
        out = []
        for para in re.split(r"\n\s*\n|\n", text.strip()):
            para, cut = para.strip(), 0
            for m in _SENT_END.finditer(para):
                if _ends_sentence(para, m.start(), para[m.end()]):
                    out.append(para[cut:m.start()])
                    cut = m.end()
            out.append(para[cut:])
        return [s for s in out if s.strip()]

    def phonemize(self, text):
        """Phoneme strings, one per sentence, in the model's alphabet (misaki conventions)."""
        chunks = []
        for sent in self.split_sentences(text):
            ps = self.g2p(sent)
            if not ps:
                continue
            if len(ps) <= MAX_PHONEMES:
                chunks.append(ps)
                continue
            cur = ""
            for part in _CLAUSE.split(ps):
                if cur and len(cur) + len(part) + 1 > MAX_PHONEMES:
                    chunks.append(cur)
                    cur = part
                else:
                    cur = f"{cur} {part}" if cur else part
            if cur:
                chunks.append(cur)
        return chunks

    def ids(self, phonemes):
        v = self.vocab
        return [0, *[v[c] for c in phonemes if c in v], 0]

    # -- audio ------------------------------------------------------------------------------
    def from_ids(self, ids, speed=1.0, rng=None):
        """Phoneme ids (with the 0 pads) -> float32 wav."""
        if len(ids) < 3:
            return np.zeros(0, np.float32)
        rng = rng or np.random.default_rng()
        out = self.sess.run(None, {"ids": np.asarray([ids], np.int64), "speed": np.asarray([speed], np.float32)})
        lh, th, ln, lr, phi, f0 = (o[0] for o in out)
        return dsp.synth(lh, th, ln, lr, phi, f0, rng)

    def from_phonemes(self, phonemes, speed=1.0, seed=None):
        """Synthesize a phoneme string directly (misaki/Kokoro alphabet, e.g. 'həlˈO ðˈɛɹ.')."""
        return self.from_ids(self.ids(phonemes), speed, np.random.default_rng(seed))

    def stream(self, text, speed=1.0, seed=None):
        """Yield one float32 wav array per sentence, each as soon as it is synthesized."""
        rng = np.random.default_rng(seed)
        for ps in self.phonemize(text):
            yield self.from_ids(self.ids(ps), speed, rng)

    def __call__(self, text, speed=1.0, seed=None):
        """Text -> float32 wav at 24 kHz. speed 1.0 is the teacher's pace (0.5 slow .. 2.0 fast);
        seed fixes the noise component so repeated calls are bit-identical."""
        parts = list(self.stream(text, speed, seed))
        return np.concatenate(parts) if parts else np.zeros(0, np.float32)

    def save(self, text, path, speed=1.0, seed=None):
        """Synthesize and write a 16-bit mono WAV; returns the path."""
        write_wav(path, self(text, speed, seed))
        return path

    def bench(self, text=None, speed=1.0):
        """Time one synthesis: dict with seconds of audio, wall time, x realtime, G2P time."""
        text = text or ("The quick brown fox jumps over the lazy dog. She sells sea shells by the sea shore. "
                        "On the 3rd of March 2024, 1,250 people paid $9.99 each to hear it read at 7:30 pm.")
        self("Warm up.")
        t0 = time.perf_counter()
        chunks = self.phonemize(text)
        t1 = time.perf_counter()
        parts = [self.from_ids(self.ids(ps), speed) for ps in chunks]
        t2 = time.perf_counter()
        secs = sum(len(p) for p in parts) / SAMPLE_RATE
        return {"audio_s": secs, "g2p_ms": (t1 - t0) * 1e3, "synth_ms": (t2 - t1) * 1e3,
                "x_realtime": secs / (t2 - t0), "sentences": len(chunks)}


def write_wav(path, audio, sample_rate=SAMPLE_RATE):
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
