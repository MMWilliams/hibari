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
- [ ] reese on 10k samples: "eval0 is the best but still needs to improve". Sent 60k teacher/student A/B (scripts/ab_teacher.py <snap>). Eval rows changed at the full-index restart, so 10k and later snaps are different sentences. Next lever if the 150k endpoint still falls short by ear: WavLM feature matching (below), then a wider decoder (c_dec 160, still frame-rate).
- [ ] Paradee-8M head-to-head (scripts/compare_paradee.py, 50 unseen sentences, 1 ORT thread, results in runs/compare/results.json). 60k int8: 2.53 MB, 326x realtime, UTMOS 3.63, WER 4.9%. Paradee int8: 9.04 MB, 24x, UTMOS 4.40, WER 2.8%. Teacher: UTMOS 4.52, WER 3.0% (matches Paradee's paper, so the harness agrees with theirs). We're 13x faster and 3.6x smaller; the quality gap is real by metric, reese liked 60k by ear. Rerun on later snapshots. Paradee tricks worth borrowing: train text side and decoder separately on teacher intermediates; parameter-free phase-locking filter for voiced 2-8 kHz buzz.
- [ ] If audio sounds robotic: add WavLM feature-matching loss (Kokoro-7M notes say it fixes phase).
- [ ] Corpus is mostly declarative prose: add questions/exclamations/dialogue for prosody range.

## Done
- [x] GAN phase 1.8 -> ~7.2 it/s (batch 16): bf16 autocast for D/G losses, MRD hops fft/4 (Vocos), real+fake batched per discriminator, 0.8 s random GAN sub-window, TPRLS off (--tprls restores). Step was WDDM launch-bound (~3300 launches/step).
- [x] ONNX export (export/export_onnx.py, durations in-graph, dynamic L) + numpy/onnxruntime inference (export/infer_numpy.py) + int8 dynamic quant. numpy iSTFT matches torch.istft to 1e-7.

## Portability (reese 2026-10-18: "needs to run on many devices")
- [ ] Keep the graph to portable ops only: conv1d (incl. depthwise), linear, layernorm, gelu, softmax attention. Current export keeps duration expansion in-graph with CumSum/Range/Less/ReduceSum/Gather (all opset-11 ops, fine for ORT-web); iSTFT lives in host code.
- [ ] Targets (reese: "any device, Mac or Windows"): Windows x64 + ARM64, macOS Apple Silicon + Intel, including old/weak laptops. No Python on the user's machine: ship a self-contained binary (dependency-free C runtime, int8 weights, real FFT; or ONNX Runtime static build) per platform, macOS as a universal binary. Also ORT Android/iOS, onnxruntime-web WASM/WebGPU.
- [ ] Portable G2P: no Python/spaCy on device. Lexicon lookup (misaki gold+silver dicts, compressed) + small fallback (espeak-ng or a tiny G2P net).
