"""Upload the staged voice files and the model card to the Hugging Face repo, then download every voice file
back anonymously and check its sha256 against the package registry (every byte must match).

    python scripts/publish_hf.py [--dry-run]
"""
import argparse, hashlib, json, os, subprocess, sys, urllib.request
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hibari.voices import BASE_URL, VOICES, sha256_of  # noqa: E402

REPO = "maureesewilliams/hibari"
ap = argparse.ArgumentParser()
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
stage = os.path.join(ROOT, "hf_stage", "voices")
files = sorted(os.listdir(stage))
print("staged:", files)
registered = {v["file"] for v in VOICES.values() if not v.get("bundled")}
missing = registered - set(files)
assert not missing, f"registry names files that are not staged: {missing}"
cmds = [["hf", "upload", REPO, stage, "voices", "--repo-type", "model", "--commit-message", "voices: " + ", ".join(sorted(registered))],
        ["hf", "upload", REPO, os.path.join(ROOT, "hf", "README.md"), "README.md", "--repo-type", "model", "--commit-message", "model card: voices"]]
for c in cmds:
    print("$", " ".join(c), flush=True)
    if not a.dry_run:
        subprocess.run(c, check=True, shell=(os.name == "nt"))
if a.dry_run:
    sys.exit(0)
bad = 0
for name, v in VOICES.items():
    if v.get("bundled"):
        continue
    url = BASE_URL + v["file"]
    req = urllib.request.Request(url, headers={"User-Agent": "hibari-verify"})      # no token: anonymous, like users
    data = urllib.request.urlopen(req, timeout=120).read()
    h = hashlib.sha256(data).hexdigest()
    local = sha256_of(os.path.join(stage, v["file"]))
    ok = h == v["sha256"] == local and len(data) == v["bytes"]
    bad += not ok
    print(f"{name:8s} {v['file']:24s} {len(data):9d} bytes  sha256 {h[:16]}...  {'OK' if ok else 'MISMATCH'}", flush=True)
card = urllib.request.urlopen(f"https://huggingface.co/{REPO}/raw/main/README.md", timeout=60).read()
same = card == open(os.path.join(ROOT, "hf", "README.md"), "rb").read()
print("README.md", "identical" if same else "DIFFERS", flush=True)
sys.exit(0 if (bad == 0 and same) else 1)
