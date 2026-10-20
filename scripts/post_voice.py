"""After a voice's decoder and ProsodyNet are trained: pick the decoder snapshot by UTMOS with the voice's
ProsodyNet on its held-out rows (diagnose_gap.py), export (fp32 + int8, speed input), numpy/ONNX parity
check, teacher-vs-student evaluation on 50 unseen sentences, stage the files for Hugging Face and register
the voice, and write three hibari-only demo sentences. Light GPU use only (UTMOS, Whisper base).

    python scripts/post_voice.py --kokoro bf_emma --name emma --language en-GB --accent British --gender female
"""
import argparse, json, os, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
ap = argparse.ArgumentParser()
ap.add_argument("--kokoro", required=True); ap.add_argument("--name", required=True)
ap.add_argument("--language", required=True); ap.add_argument("--accent", required=True); ap.add_argument("--gender", required=True)
ap.add_argument("--snaps", default="10k 15k 20k 25k 30k")
ap.add_argument("--best", help="skip the snapshot selection and use this one (e.g. 15k)")
ap.add_argument("--run", help="run dir (default runs/voices/<kokoro>)"); ap.add_argument("--export", help="export dir (default export/<kokoro>)")
ap.add_argument("--prosody", help="ProsodyNet checkpoint (default <run>/prosody/best.pt)")
ap.add_argument("--no-stage", action="store_true", help="evaluate only: no staging, registry entry or demos (experiments)")
a = ap.parse_args()
v, run = a.kokoro, a.run or f"runs/voices/{a.kokoro}"
exp = a.export or f"export/{v}"
pn = a.prosody or f"{run}/prosody/best.pt"
env = {**os.environ, "PYTHONIOENCODING": "utf-8", "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")}


def sh(*cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)


if not a.best:
    sh(PY, "scripts/diagnose_gap.py", *a.snaps.split(), "--run", f"{run}/tiny", "--prosody", pn, "--index", f"data/dist_{v}/shard*.jsonl", "--n", "50")
res = json.load(open(f"{run}/tiny/diag/results.json"))
best = a.best or max(a.snaps.split(), key=lambda s: res[f"{s}_pnet"])
print(f"[*] {v}: teacher UTMOS {res['teacher']:.3f}; " + ", ".join(f"{s} {res[f'{s}_pnet']:.3f}" for s in a.snaps.split()) + f" -> {best}", flush=True)
ck = f"{run}/tiny/snap/{best}.pt"
sh(PY, "export/export_onnx.py", "--ckpt", ck, "--prosody", pn, "--out", f"{exp}/tiny.onnx")
sh(PY, "export/test_istft.py", ck, exp, "--prosody", pn)
sh(PY, "scripts/eval_voice.py", "--voice", v, "--tiny", exp, "--n", "50", "--out", f"{run}/eval")
json.dump({"best": best, "diag": res, "eval": json.load(open(f"{run}/eval/results.json"))}, open(f"{run}/summary.json", "w"), indent=1)
if a.no_stage:
    sys.exit(0)
sh(PY, "scripts/stage_voices.py", "--name", a.name, "--kokoro", v, "--export", exp, "--language", a.language,
   "--accent", a.accent, "--gender", a.gender)
# demos through the package itself (its own G2P, the staged int8 file found through HIBARI_HOME)
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ["HIBARI_HOME"] = os.path.join(ROOT, "hf_stage", "voices")
from hibari import Hibari  # noqa: E402
tts = Hibari(voice=a.name)
who = {"emma": "Emma, hibari's British voice", "fenrir": "Fenrir, hibari's American male voice",
       "george": "George, hibari's British male voice", "fable": "Fable, hibari's British male voice",
       "lewis": "Lewis, hibari's British male voice", "daniel": "Daniel, hibari's British male voice",
       "bella": "Bella, hibari's second American voice", "nicole": "Nicole, hibari's second American voice",
       "sarah": "Sarah, hibari's second American voice", "kore": "Kore, hibari's second American voice",
       "aoede": "Aoede, hibari's second American voice"}.get(a.name, a.name)
demos = [f"Hello! I'm {who}. I run at over a hundred times realtime on a single CPU core, from a model under eight megabytes.",
         "On the 3rd of March 2024, 1,250 people paid £9.99 each to hear it read at 7:30 pm; the queue stretched round the block.",
         "Could you pass me the water, please? Thank you very much. Now, where were we? Ah yes: the skylark sings as it climbs."]
for i, d in enumerate(demos, 1):
    tts.save(d, f"{run}/demo{i}.wav", seed=0)
json.dump(demos, open(f"{run}/demos.json", "w", encoding="utf-8"), indent=1)
print(f"[*] {v}: demos in {run}/demo1-3.wav; A/B in {run}/ab0-2_teacher_student.wav; best snapshot {best}", flush=True)
