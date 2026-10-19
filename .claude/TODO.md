# kokoro-tiny TODO

Goal: Kokoro-82M (af_heart) distilled into a single-voice student that is smaller and faster than Paradee-8M and as close to the teacher as possible, for weak Mac/Windows CPUs.
Student = training/tiny_model.py: TinyKokoro v3 (c_dec 256 x 8 blocks, ff 768, src=True pulse source + noise + free-phase branch, 4.19M) + ProsodyNet (durations, 80 Hz F0/energy, 1.68M, runs/prosody_v1/best.pt).
Shipping export: export/v3_final (one ONNX graph: ProsodyNet durations/F0/energy, TinyKokoro 40k text features + decoder, its own dur/prosody heads dropped; host DSP = pulses + noise + iSTFT in export/infer_numpy.py).

## Result 2026-10-19 (runs/compare_v3, 50 unseen sentences, 1 ORT thread; machine ~70% busy with other sessions, so xRT is low for everyone)
| model | MB | xRT | UTMOS | WER |
| teacher Kokoro-82M | | | 4.52 | 2.8% |
| Paradee int8 | 9.04 | 7.4 (14 best-of-4 re-run; 22-24 on a quiet machine) | 4.41 | 2.8% |
| v3 40k + ProsodyNet int8 | 7.94 | 62 (134 best-of-4) | 4.43 | 2.7% |
| v3 40k + ProsodyNet fp32 | 24.75 | 46 (99 best-of-4) | 4.43 | 2.7% |
A/B wavs: runs/compare_v3/ab{0,1,2}_teacher_paradee_tiny.wav. Snapshot pick by pnet UTMOS (runs/tiny_v3/diag/results.json): 30k 4.404, 40k 4.437, 50k 4.431, 60k 4.430; teacher 4.529.

## Open
- [ ] Listen to runs/compare_v3/ab*.wav (before ProsodyNet reese found v3 10k ab0 "fuzzy and robotic", ab2 "nearly perfect").
- [ ] Size: int8 7.94 MB vs Paradee 9.04. Convs stay fp32 (ConvInteger 2x slower than fp32 Conv in ORT 1.29 x86; re-test on ARM or QLinearConv static quant). ProsodyNet (c 192, 6 shared attention layers, 6 dilated ConvNeXt) is the next thing to shrink.
- [ ] Speed: int8 net ~4-5 ms per s of audio, host DSP ~3.3 ms (pulses 0.6, stft 0.6, noise 0.6, combine 1.2, ola 0.4; scripts/profile_dsp.py export/v3_final/tiny.int8.onnx). C runtime idea: analytic pulse spectrum per frame (Hann kernel at h*f0, phase h*theta) skips the audio-rate pulses and forward FFT; approximation, needs a short fine-tune.
- [ ] int8 F0 differs from fp32 by 0.6 Hz mean / 5.4 Hz max (ProsodyNet MatMuls quantized), durations within +-2 frames; UTMOS equal. If pitch is off by ear, leave ProsodyNet's dur/pro heads fp32.
- [ ] Training schedule: v1 and v2 peaked at 60k, v3 peaks at 40k with ProsodyNet (free prosody keeps improving to 50k). Shorter cosine or lower GAN weight late if retraining.
- [ ] v3 noise branch carries ~0.35% of spectral energy (per-element up to 23x the other branches, so not droppable); the free-phase branch carries most above ~2.5 kHz.
- [ ] On-device G2P: misaki+spacy is far heavier than the model. (a) misaki dictionary (us_gold/us_silver json) + espeak-ng for OOV, no spacy; (b) tiny G2P net (<0.5M, trained on misaki's dictionary + teacher phonemes in data/dist); (c) precomputed phoneme cache. The export takes phoneme ids (export/vocab.json).
- [ ] Measure on a real low-end device (or emulate: 1 thread, low clock). scipy.fft keeps float32; check pocketfft on ARM.
- [ ] If audio sounds robotic: WavLM feature-matching loss (Kokoro-7M notes say it fixes phase).
- [ ] Corpus is mostly declarative prose: add questions/exclamations/dialogue for prosody range.

## History
- Teacher dump: 30k sentences (29,999 rows, 66 h) in data/dist. Held-out split = last 200 of the shuffled rows (scripts/diagnose_gap.py). compare_paradee.py uses sentences [30000:] that no model trained on; its teacher UTMOS/WER match Paradee's paper.
- v1 (1.34M, free-phase Vocos-style head, runs/tiny_v1): 60k int8 2.53 MB, 326x, UTMOS 3.63, WER 4.9%; 150k worse. Teacher-forced prosody didn't close the gap, so the decoder was the bottleneck.
- v2 (decoder c_dec 256 x 8, runs/tiny_v2, init text side from v1 60k): 60k UTMOS 3.74; harmonics above ~500 Hz broke up (frame-wise absolute phase incoherent; teacher and Paradee keep a sine source).
- v3 (pulse source, runs/tiny_v3, init from v2 60k, 60k steps): 10k UTMOS 4.32 clean harmonics; with ProsodyNet 30k 4.40, 40k 4.44. scripts/fuzz_probe.py showed the remaining per-row loss came from predicted pitch (~1 st off, range 16% flat): ProsodyNet (training/train_prosody.py, whole sentences, 18 min on GPU 1) pitch err 1.06 -> 0.47 st, range 0.84 -> 0.95, dur err 0.32 -> 0.25 frames.
- Export: export/export_onnx.py (--prosody <ckpt>; durations and token->frame expansion in-graph with opset-11 ops; shared attention weights folded once; int8 dynamic MatMul only). infer_numpy.py: numpy/scipy float32 host DSP, matches torch to_wav to 3e-5..8e-4 wav max err (export/test_istft.py <ckpt> <dir> --prosody <pnet>); pulses via broadcast interpolation, branches combined in complex64.
- GAN phase 1.8 -> ~7.2 it/s (batch 16): bf16 autocast for D/G losses, MRD hops fft/4, real+fake batched, 0.8 s GAN sub-window, TPRLS off (--tprls). Step was WDDM launch-bound.

## Portability (reese 2026-10-18: "needs to run on many devices")
- [ ] Keep the graph to portable ops only: conv1d (incl. depthwise), linear, layernorm, gelu, softmax attention, CumSum/Range/Less/ReduceSum/Gather for the duration expansion (all opset-11, fine for ORT-web); pulses/noise/iSTFT live in host code.
- [ ] Targets: Windows x64 + ARM64, macOS Apple Silicon + Intel, old/weak laptops. No Python on the user's machine: self-contained binary per platform (dependency-free C runtime with int8 weights and a real FFT, or ONNX Runtime static build), macOS universal. Also ORT Android/iOS, onnxruntime-web WASM/WebGPU.
- [ ] Portable G2P: lexicon lookup (misaki gold+silver dicts, compressed) + small fallback (espeak-ng or a tiny G2P net).
