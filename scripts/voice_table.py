"""Markdown rows for the per-voice evaluation tables (README, MODEL_CARD, hf/README.md, docs/METRICS.md) from
runs/voices/<kokoro>/summary.json (written by post_voice.py) and the registry.

    python scripts/voice_table.py
"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hibari.voices import VOICES  # noqa: E402

HEART = {"teacher": (4.52, 0.028), "int8": (4.43, 0.027), "xrt": 128, "snap": "40k (60k run from scratch)"}
rows = []
for name, v in VOICES.items():
    if v["bundled"]:
        rows.append((name, v, HEART["teacher"], HEART["int8"], HEART["xrt"], HEART["snap"], None))
        continue
    p = os.path.join(ROOT, "runs", "voices", v["kokoro"], "summary.json")
    if not os.path.exists(p):
        continue
    s = json.load(open(p))
    e = s["eval"]["rows"]
    rows.append((name, v, (e["teacher"]["utmos"], e["teacher"]["wer"]), (e["tiny_int8"]["utmos"], e["tiny_int8"]["wer"]),
                 e["tiny_int8"]["x_realtime"], s["best"], s["diag"]))
print("| voice | Kokoro voice | teacher UTMOS / WER | hibari int8 UTMOS / WER | UTMOS gap | x realtime (1 thread) | file MB |")
print("|---|---|---|---|---|---|---|")
for name, v, t, st, xrt, snap, diag in rows:
    print(f"| {name} | {v['kokoro']} ({v['accent']} {v['gender']}) | {t[0]:.2f} / {100 * t[1]:.1f}% | {st[0]:.2f} / {100 * st[1]:.1f}% | "
          f"{t[0] - st[0]:.2f} | {xrt:.0f} | {v['bytes'] / 1e6:.2f} |")
print()
for name, v, t, st, xrt, snap, diag in rows:
    if diag:
        snaps = ", ".join(f"{k[:-5]} {x:.3f}" for k, x in diag.items() if k.endswith("_pnet"))
        print(f"- {name}: snapshot {snap} picked by UTMOS with the voice's ProsodyNet on 50 held-out training rows "
              f"(teacher {diag['teacher']:.3f}; {snaps}).")
