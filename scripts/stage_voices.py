"""Stage an exported voice for the Hugging Face repo and register it in the package: copies the int8 and
fp32 ONNX files to hf_stage/voices/<kokoro>.{int8,fp32}.onnx (gitignored; uploaded by hand after
listening) and writes the int8 file's size and sha256 into src/hibari/assets/voices.json.

    python scripts/stage_voices.py --name emma --kokoro bf_emma --export export/bf_emma --language en-GB --accent British --gender female
"""
import argparse, json, os, shutil, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hibari.voices import sha256_of  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--kokoro", required=True); ap.add_argument("--export", required=True)
ap.add_argument("--language", required=True, help="en-US or en-GB"); ap.add_argument("--accent", required=True)
ap.add_argument("--gender", required=True)
a = ap.parse_args()
stage = os.path.join(ROOT, "hf_stage", "voices"); os.makedirs(stage, exist_ok=True)
int8 = os.path.join(stage, f"{a.kokoro}.int8.onnx")
shutil.copyfile(os.path.join(a.export, "tiny.int8.onnx"), int8)
shutil.copyfile(os.path.join(a.export, "tiny.onnx"), os.path.join(stage, f"{a.kokoro}.fp32.onnx"))
reg_path = os.path.join(ROOT, "src", "hibari", "assets", "voices.json")
reg = json.load(open(reg_path, encoding="utf-8"))
reg[a.name] = {"kokoro": a.kokoro, "language": a.language, "accent": a.accent, "gender": a.gender,
               "file": os.path.basename(int8), "bundled": False, "bytes": os.path.getsize(int8), "sha256": sha256_of(int8),
               "british": a.language.endswith("GB")}
heart = os.path.join(ROOT, "src", "hibari", "assets", "hibari.onnx")
reg["heart"]["bytes"], reg["heart"]["sha256"] = os.path.getsize(heart), sha256_of(heart)
json.dump(reg, open(reg_path, "w", encoding="utf-8"), indent=1)
print(f"{a.name}: {int8} {os.path.getsize(int8) / 1e6:.2f} MB sha256 {reg[a.name]['sha256'][:16]}...; registry {reg_path}")
