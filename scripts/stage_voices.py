"""Stage an exported voice for the Hugging Face repo and register it in the package: copies the int8 and
fp32 ONNX files to <stage>/<kokoro><suffix>.{int8,fp32}.onnx (gitignored; uploaded by hand after
listening) and writes the int8 file's size and sha256 into src/hibari/assets/voices.json.

    python scripts/stage_voices.py --name emma --kokoro bf_emma --export export/bf_emma --language en-GB --accent British --gender female
    python scripts/stage_voices.py --name emma --kokoro bf_emma --export export/bf_emma_v2 --language en-GB --accent British \
        --gender female --stage hf_stage/voices_v2 --suffix _v2 --run bf_emma_v2      # 0.2.1: new file name, old file untouched
"""
import argparse, json, os, shutil, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hibari.voices import sha256_of  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--kokoro", required=True); ap.add_argument("--export", required=True)
ap.add_argument("--language", required=True, help="en-US or en-GB"); ap.add_argument("--accent", required=True)
ap.add_argument("--gender", required=True)
ap.add_argument("--stage", default=os.path.join(ROOT, "hf_stage", "voices"), help="staging directory")
ap.add_argument("--suffix", default="", help="appended to the file name (a new release keeps the old file for old registries)")
ap.add_argument("--run", help="runs/voices/<run> holding this voice's summary.json (for voice_table.py) if not <kokoro>")
a = ap.parse_args()
os.makedirs(a.stage, exist_ok=True)
int8 = os.path.join(a.stage, f"{a.kokoro}{a.suffix}.int8.onnx")
shutil.copyfile(os.path.join(a.export, "tiny.int8.onnx"), int8)
shutil.copyfile(os.path.join(a.export, "tiny.onnx"), os.path.join(a.stage, f"{a.kokoro}{a.suffix}.fp32.onnx"))
reg_path = os.path.join(ROOT, "src", "hibari", "assets", "voices.json")
reg = json.load(open(reg_path, encoding="utf-8"))
reg[a.name] = {"kokoro": a.kokoro, "language": a.language, "accent": a.accent, "gender": a.gender,
               "file": os.path.basename(int8), "bundled": False, "bytes": os.path.getsize(int8), "sha256": sha256_of(int8),
               "british": a.language.endswith("GB"), **({"run": a.run} if a.run else {})}
heart = os.path.join(ROOT, "src", "hibari", "assets", "hibari.onnx")
reg["heart"]["bytes"], reg["heart"]["sha256"] = os.path.getsize(heart), sha256_of(heart)
json.dump(reg, open(reg_path, "w", encoding="utf-8"), indent=1)
print(f"{a.name}: {int8} {os.path.getsize(int8) / 1e6:.2f} MB sha256 {reg[a.name]['sha256'][:16]}...; registry {reg_path}")
