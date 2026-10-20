"""Listening material for an improved voice: the same 3 sentences as the shipped A/B, each as teacher, shipped hibari
and improved hibari in one wav (runs/voices/<run>/abN_teacher_shipped_improved.wav, 0.5 s gaps), and demo1-3 through
the package itself from the staged file (HIBARI_HOME = the staging directory, so the registry must already point at
the new file: run stage_voices.py first).

    python scripts/ab_v2.py --name emma --kokoro bf_emma --run bf_emma_v2 --stage hf_stage/voices_v2
"""
import argparse, json, os, sys
import numpy as np, soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 24000
ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--kokoro", required=True)
ap.add_argument("--run", required=True, help="runs/voices/<run>: the improved voice (eval/ from eval_voice.py)")
ap.add_argument("--shipped", help="runs/voices/<dir> of the shipped voice (default <kokoro>)")
ap.add_argument("--stage", default=os.path.join(ROOT, "hf_stage", "voices_v2"))
ap.add_argument("--n", type=int, default=3)
ap.add_argument("--no-demos", action="store_true", help="A/B wavs only (candidate not staged)")
a = ap.parse_args()
new, old = os.path.join(ROOT, "runs", "voices", a.run), os.path.join(ROOT, "runs", "voices", a.shipped or a.kokoro)
gap = np.zeros(SR // 2, np.float32)
for i in range(a.n):
    t = sf.read(os.path.join(new, "eval", f"{i:02d}_teacher.wav"), dtype="float32")[0]
    s = sf.read(os.path.join(old, "eval", f"{i:02d}_tiny_int8.wav"), dtype="float32")[0]
    n = sf.read(os.path.join(new, "eval", f"{i:02d}_tiny_int8.wav"), dtype="float32")[0]
    t_old = sf.read(os.path.join(old, "eval", f"{i:02d}_teacher.wav"), dtype="float32")[0]
    assert len(t) == len(t_old), "the two eval runs used different sentences"
    sf.write(os.path.join(new, f"ab{i}_teacher_shipped_improved.wav"), np.concatenate([t, gap, s, gap, n]), SR)
    print(os.path.join(new, f"ab{i}_teacher_shipped_improved.wav"))
if a.no_demos:
    sys.exit()

sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ["HIBARI_HOME"] = a.stage
from hibari import Hibari  # noqa: E402
demos = json.load(open(os.path.join(old, "demos.json"), encoding="utf-8"))
tts = Hibari(voice=a.name)
for i, d in enumerate(demos, 1):
    tts.save(d, os.path.join(new, f"demo{i}.wav"), seed=0)
    print(os.path.join(new, f"demo{i}.wav"))
json.dump(demos, open(os.path.join(new, "demos.json"), "w", encoding="utf-8"), indent=1)
