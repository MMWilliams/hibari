"""Teacher-then-student A/B wav per eval row for a snapshot: python scripts/ab_teacher.py 60k"""
import json, sys
import numpy as np, soundfile as sf

snap = sys.argv[1]
run = sys.argv[2] if len(sys.argv) > 2 else "runs/tiny_v1"
rows = json.load(open(f"{run}/eval_rows.json"))
gap = np.zeros(12000, np.float32)
for j, r in enumerate(rows[:3]):
    teacher = np.memmap(r["bin"], dtype=np.int16, mode="r")[r["offset"]: r["offset"] + r["samples"]]
    student, sr = sf.read(f"{run}/snap/{snap}_eval{j}.wav", dtype="float32")
    sf.write(f"{run}/snap/{snap}_ab{j}.wav",
             np.concatenate([teacher.astype(np.float32) / 32767, gap, student]), sr)
    print(f"{run}/snap/{snap}_ab{j}.wav")
