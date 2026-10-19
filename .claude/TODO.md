# kokoro-tiny TODO

Goal: Kokoro-82M distilled into a single-voice student for very low CPU (target: Pi Zero 2 / low-end ARM single core).
Student = training/tiny_model.py (1.34M params, frame-rate Vocos-style iSTFT head, no style, no sine source).
Baseline on this desktop, 1 thread: tiny RTF 0.0038 vs Kokoro-7M-Distill 0.0496.

## Open
- [ ] Teacher dump: 30k sentences, 3 shards on GPU 1 (logs/dump*.log), expected done ~22:00 EDT 2026-10-18. When done: restart training with --resume so it reads the full index (current run started on 17,197 rows).
- [ ] Training run runs/tiny_v1 (GPU 0, logs/train_v1.log, ~7.2 it/s in GAN phase after the fix). Listen to snap/*k_eval*.wav at each 10k; pick by ear, not by loss. `python scripts/check_snaps.py` = static/silence check only.
- [ ] On-device G2P: misaki+spacy is far heavier than the model. Options: (a) misaki's dictionary (us_gold/us_silver json) lookup + espeak-ng for OOV, no spacy; (b) tiny G2P net (LSTM/transformer <0.5M params, trained on misaki's dictionary + teacher phonemes in data/dist); (c) precomputed phoneme cache for a fixed phrase set. The export takes phoneme ids (export/vocab.json), so G2P is a separate module.
- [ ] Measure on a real low-end device (or emulate: 1 thread, low clock). numpy 1.26 irfft runs in float64; scipy.fft / pocketfft float32 may be faster on ARM.
- [ ] ConvInteger (int8 convs) is 2x slower than fp32 Conv in onnxruntime 1.29 on x86: export quantizes MatMul only (2.53 MB). Re-test on ARM, or QLinearConv static quantization.
- [ ] If audio sounds robotic: add WavLM feature-matching loss (Kokoro-7M notes say it fixes phase).
- [ ] Corpus is mostly declarative prose: add questions/exclamations/dialogue for prosody range.

## Done
- [x] GAN phase 1.8 -> ~7.2 it/s (batch 16): bf16 autocast for D/G losses, MRD hops fft/4 (Vocos), real+fake batched per discriminator, 0.8 s random GAN sub-window, TPRLS off (--tprls restores). Step was WDDM launch-bound (~3300 launches/step).
- [x] ONNX export (export/export_onnx.py, durations in-graph, dynamic L) + numpy/onnxruntime inference (export/infer_numpy.py) + int8 dynamic quant. numpy iSTFT matches torch.istft to 1e-7.
