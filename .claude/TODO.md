# kokoro-tiny TODO

Goal: Kokoro-82M distilled into a single-voice student for very low CPU (target: Pi Zero 2 / low-end ARM single core).
Student = training/tiny_model.py (1.34M params, frame-rate Vocos-style iSTFT head, no style, no sine source).
Baseline on this desktop, 1 thread: tiny RTF 0.0038 vs Kokoro-7M-Distill 0.0496.

## Open
- [ ] GAN phase is slow: 15 it/s before step 2000, ~1.8 it/s after (MPD + MRD + TPRLS on 16 x 1.6 s crops). 150k steps = ~23 h. Profile and fix (bf16 autocast for discriminators, smaller GAN crop, fewer MRD resolutions, drop TPRLS) before the long run.
- [ ] Teacher dump: 30k sentences, 3 shards on GPU 1 (logs/dump*.log). When done: restart training with --resume so it reads the full index.
- [ ] Training run runs/tiny_v1 (GPU 0, logs/train_v1.log). Listen to snap/*k_eval*.wav at each 10k; pick by ear, not by loss.
- [ ] ONNX export of TinyKokoro up to (logmag, phase); iSTFT via numpy irfft outside the graph (a DFT matrix in-graph costs more than the net).
- [ ] int8 dynamic quantization of the ONNX; measure size + RTF on 1 thread.
- [ ] On-device G2P: misaki+spacy is far heavier than the model. Options: misaki dictionary lookup + espeak-ng fallback, or a tiny G2P net.
- [ ] Measure on a real low-end device (or emulate: 1 thread, low clock).
- [ ] If audio sounds robotic: add WavLM feature-matching loss (Kokoro-7M notes say it fixes phase).
- [ ] Corpus is mostly declarative prose: add questions/exclamations/dialogue for prosody range.

## Portability (reese 2026-10-18: "needs to run on many devices")
- [ ] Keep the graph to portable ops only: conv1d (incl. depthwise), linear, layernorm, gelu, softmax attention. Duration expansion (repeat_interleave) and iSTFT live in host code, not the graph.
- [ ] Targets: ONNX Runtime (Windows/Linux/macOS, Android, iOS), onnxruntime-web WASM/WebGPU (browser), and a dependency-free C runtime (int8 weights, real FFT) for embedded Linux / very small CPUs.
- [ ] Portable G2P: no Python/spaCy on device. Lexicon lookup (misaki gold+silver dicts, compressed) + small fallback (espeak-ng or a tiny G2P net).
