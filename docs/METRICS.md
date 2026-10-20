# Size and speed sheet

| model | params | file MB | x realtime 1 thread (RTF) | x realtime 4 threads (RTF) | latency short / long ms (1 thread) | peak RSS MB | load s |
|---|---|---|---|---|---|---|---|
| ours v3 40k + ProsodyNet int8 | 6.15 M | 7.94 | 128 (0.0078) | 175 (0.0057) | 12 / 71 | 158 | 0.81 |
| ours v3 40k + ProsodyNet fp32 | 6.15 M | 24.75 | 104 (0.0096) | 167 (0.0060) | 15 / 76 | 171 | 0.86 |
| Paradee-8M int8 | 9.06 M | 9.04 | 18.3 (0.0547) | 27 (0.0365) | 87 / 602 | 176 | 2.05 |
| Paradee-8M fp32 | 9.06 M | 36.99 | 18.9 (0.0529) | 28 (0.0361) | 80 / 540 | 191 | 1.26 |
| Kokoro-7M-Distill fp32 | 7.48 M | 30.10 | 17.4 (0.0573) | 27 (0.0371) | 108 / 357 | 889 | 10.91 |
| Kokoro-82M teacher fp32 | 81.81 M | 327.21 | 1.9 (0.5277) | 3.9 (0.2552) | 739 / 3880 | 1527 | 10.63 |

Notes
- Machine: AMD Ryzen 9 9900X 12-Core Processor (12 cores / 24 threads), Windows build 10.0.26200, Python 3.11.15, onnxruntime 1.29.0, torch 2.11.0+cu128, numpy 1.26.4, scipy 1.17.1.
- System CPU load (all 24 threads) measured over 2 s before each run (threshold 20%, up to 6 retries of 10 s) and again after it: timing 1 thread 13/35%, timing 4 thread 9/21%, memory 20%. Other applications were running; nothing was stopped.
- Speed: 20 phoneme-id sentences (109.4 s of audio from ours int8): 4 short phrases (misaki G2P), the 8 runs/tiny_v1/eval_rows.json rows, 8 held-out corpus rows spread over 2-10 s. All models in one process, warm (2 sentences), rounds x sentences x models round-robin, best of 3 rounds per sentence; x realtime = audio s / summed best wall s, RTF = its inverse. ONNX: intra-op threads = N; torch: torch.set_num_threads(N); ours includes the numpy host DSP (single-threaded at both thread counts).
- Latency: best-of-3 wall time for sentence 12 (1.80 s of audio, 27 ids) and sentence 18 (7.86 s, 133 ids), 1 thread. The 4 short phrases came out at 1.9-2.2 s.
- Ours, where the time goes per second of audio: int8 1 thr: net 4.10 ms, host DSP 3.72 ms; int8 4 thr: net 2.17 ms, host DSP 3.55 ms; fp32 1 thr: net 6.48 ms, host DSP 3.10 ms; fp32 4 thr: net 3.10 ms, host DSP 2.89 ms. Host DSP stages (scripts/profile_dsp.py, int8, ms per s of audio): pulses 0.61, stft 0.65, noise 0.69, combine 1.56, ola 0.50.
- Ours parameters (6.15 M in the graph): ProsodyNet 1.68 M, text encoder 0.29 M, decoder (dec_in + 8 ConvNeXt + norm) 3.41 M, output head 0.77 M. TinyKokoro's own duration/prosody heads are not exported.
- Parameters: ONNX models counted from the fp32 graph initializers (int8 holds the same weights); torch models from model.parameters(). File MB = the weights file only; the Kokoro voice pack af_heart.pt adds 0.52 MB.
- Memory and load: each model in a fresh python process, 1 thread. Load = imports + weights (interpreter start excluded), peak RSS = Windows peak working set after loading and synthesizing the longest sentence (14.3 s). onnxruntime + numpy + scipy.fft imported, no model: 59 MB, 0.44 s; torch + kokoro imported, no model: 744 MB, 9.45 s. RSS right after load / first-call time: tiny_int8 136 MB / 0.12 s, tiny_fp32 150 MB / 0.15 s, paradee_int8 171 MB / 0.76 s, paradee_fp32 191 MB / 0.64 s, kokoro7m 812 MB / 0.62 s, kokoro82m 1382 MB / 8.42 s.
- Kokoro-7M-Distill is built as its load_model.py does (config.json, kokoro_en_7m.pth, disable_complex=True) through the repo's kokoro_patched (a superset of the one shipped with it), voice af_heart.pt; the teacher is the stock KModel (torch.stft path), voice af_heart. Both are driven with forward_with_tokens on the same phoneme ids. Paradee gets input_ids + speed 1.0.

## Through the hibari package (2026-10-19, scripts/bench_package.py)
Text in, audio out via `Hibari()` (int8 model, hibari G2P, numpy/scipy DSP), 20 held-out corpus sentences from
g2p_work/eval.jsonl (160.5 s of audio, 8 s average), best of 3 rounds per sentence, system load 12-18%:
175x realtime on 1 thread (RTF 0.0057), 225x on 4 threads (RTF 0.0044); G2P median 0.9 ms per sentence.
Same wheel in WSL Ubuntu-22.04 (python 3.10, onnxruntime 1.23.2, `hibari --bench`, 3 sentences / 15.6 s): 210x / 298x.
Load: imports (numpy, onnxruntime) 0.21 s, model session 0.16 s, G2P 0.31 s (lexicon 73 ms, tagger 95 ms, OOV net 25 ms).
The longer sentences here amortize the fixed per-call cost better than the 20-sentence mix above (128x), and the machine was quieter.

## Voices (2026-10-20, scripts/eval_voice.py)
Each voice is its own decoder + ProsodyNet with heart's architecture; fenrir, emma, fable and kore were fine-tuned from
heart's weights (training/train_tiny.py --init runs/tiny_v3/snap/40k.pt --init-disc runs/tiny_v3/last.pt, 30k steps at
lr 1e-4, and train_prosody.py --init runs/prosody_v1/best.pt, 20k steps) on a 15k-sentence teacher dump of that voice
(emma's second run: 30k sentences, 60k steps). 50 unseen sentences (corpus lines[30000:], never synthesized for any
voice's training), the student fed the teacher's phoneme ids; UTMOS utmos22_strong, WER Whisper base; speed = audio
seconds / wall seconds on one onnxruntime thread with the numpy DSP, measured while training jobs shared the machine
(so the heart row, measured earlier on a quieter mix of short and long sentences, is not directly comparable).

<!-- voices-table -->
| voice | Kokoro voice | teacher UTMOS / WER | hibari int8 UTMOS / WER | UTMOS gap | x realtime (1 thread) | file MB |
|---|---|---|---|---|---|---|
| heart | af_heart (American female) | 4.52 / 2.8% | 4.43 / 2.7% | 0.09 | 128 | 7.94 |
| emma | bf_emma (British female) | 4.40 / 2.9% | 4.16 / 2.6% | 0.24 | 139 | 7.94 |
| fenrir | am_fenrir (American male) | 4.48 / 2.6% | 4.35 / 2.9% | 0.13 | 166 | 7.94 |
| fable | bm_fable (British male) | 4.33 / 2.6% | 4.17 / 2.6% | 0.16 | 167 | 7.94 |
| kore | af_kore (American female) | 4.52 / 2.4% | 4.45 / 2.5% | 0.06 | 170 | 7.94 |

- emma: snapshot 40k picked by UTMOS with the voice's ProsodyNet on 50 held-out training rows (teacher 4.394; 20k 4.107, 30k 4.125, 40k 4.174, 50k 4.174, 60k 4.173).
- fenrir: snapshot 30k picked by UTMOS with the voice's ProsodyNet on 50 held-out training rows (teacher 4.496; 10k 4.382, 15k 4.400, 20k 4.397, 25k 4.403, 30k 4.414).
- fable: snapshot 25k picked by UTMOS with the voice's ProsodyNet on 50 held-out training rows (teacher 4.333; 10k 4.143, 15k 4.167, 20k 4.151, 25k 4.176, 30k 4.161).
- kore: snapshot 25k picked by UTMOS with the voice's ProsodyNet on 50 held-out training rows (teacher 4.521; 10k 4.454, 15k 4.465, 20k 4.466, 25k 4.471, 30k 4.469).
<!-- /voices-table -->
