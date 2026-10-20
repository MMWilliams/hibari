# kokoro-tiny / hibari TODO

Goal: Kokoro-82M (af_heart) distilled into a single-voice student that is smaller and faster than Paradee-8M and as
close to the teacher as possible, for weak Mac/Windows CPUs. Product name: **hibari** (skylark).
Student = training/tiny_model.py: TinyKokoro v3 decoder (4.19M) + ProsodyNet (1.68M, runs/prosody_v1/best.pt).
Shipping graph: export/v3_speed (= v3_final + a `speed` input; identical outputs at speed 1.0), copied to
src/hibari/assets/hibari.onnx (int8, 7.94 MB).

## Phase 1 done 2026-10-19 (package + G2P + docs)
- `hibari` package (pyproject.toml, src/hibari): numpy + onnxruntime only (scipy optional), py3-none-any wheel
  11.8 MB (dist/, rebuild with `python -m build --wheel`), API Hibari()/save/stream/phonemize/from_phonemes/speed/seed, CLI (`hibari`, --play/--phonemes/--bench),
  tests/ (39 pass). Verified: fresh venv on Windows (py 3.11) and WSL Ubuntu-22.04 (py 3.10) via scripts/test_wheel.sh.
- G2P without spaCy/espeak (src/hibari/g2p): normalizer (assets/normalize.json), spaCy-rule tokenizer + special cases,
  averaged-perceptron tagger (assets/tagger.json.gz 1.24 MB, 97.5% vs spaCy), misaki lexicon/rules port (100% exact on
  3000 dev sentences given spaCy's tokens+tags), OOV net (assets/oov.onnx 0.73M params, 3 MB; 72% word acc / 9% PER vs
  espeak on unseen words). vs misaki+spaCy+espeak on 2500 held-out sentences: PER 0.50% / exact 88.0% (normalization on),
  0.27% / 90.9% (off); sentences without OOV words 95.4% / 97.5%. End to end 50 unseen sentences through the same model:
  47/50 identical, UTMOS 4.431 vs 4.431, WER 2.6% vs 2.7%. G2P 0.5-1 ms per sentence, G2P() load ~0.3 s (lexicon 73 ms,
  tagger 95 ms, OOV 25 ms); Hibari() ready in ~0.7 s incl. imports. scripts/bench_package.py: 175x realtime 1 thread / 225x 4 threads
  (20 held-out sentences, 160 s audio, load 16%); WSL 210x / 300x.
- Docs: README.md (users/developers/researchers, ONNX I/O spec, reproduce commands), MODEL_CARD.md, LICENSE (Apache-2.0),
  NOTICE (Kokoro-82M, misaki, spaCy, espeak-ng data provenance, StyleTTS2 MIT, num2words).
- Research data for the G2P lives in g2p_work/ (gitignored, ~1 GB): dump_misaki.py reference dumps, espeak labels, trainings.
- Published 2026-10-19 (reese's go): GitHub MMWilliams/hibari (public, master) + release v0.1.0 with the wheel;
  Hugging Face maureesewilliams/hibari (ONNX int8/fp32, g2p/, safetensors checkpoints, samples/; card source in
  hf/README.md). Sentence splitter no longer breaks after titles/initials ("Dr. Smith").

## Open (G2P)
- [ ] Tagger: "that" DT vs IN (12/2500), "in" RP vs IN (8), "I object" (VBP vs NN); 2.2 MB unpruned -> 1.24 MB pruned at |w|>=1.5
      (97.46%); prune 2.5 gives 0.69 MB / 38 ms load at 97.1% if size matters more.
- [ ] OOV net: 72% word accuracy vs espeak on unseen names; a decoder or more data (espeak on more names) would help; words
      with letters+digits are split (letters -> net, digits -> number reader); "H2O" reads "H to O" like misaki.
- [ ] Normalizer: "9/11" -> "September eleventh" (date) vs misaki "nine eleven"; dates assume US month/day; "St." -> Street
      except before a capitalized word (Saint); add more units/abbreviations as users hit them. Rules are regexes in
      normalize.py, tables in assets/normalize.json (native port needs to re-implement the regexes).
- [ ] Load time: lexicon JSON gz 73 ms (pickle would be 2x faster but not language-neutral); tagger 95 ms; whole Hibari()
      incl. onnxruntime import ~0.5 s.
- [ ] Phase 2: native no-Python CLI/library (C/Rust) for Mac/Windows/Linux: ONNX Runtime static or own int8 kernels, real FFT;
      port tokenizer regexes, perceptron, lexicon lookup, number words, OOV net (plain transformer encoder + CTC collapse).
      Mac build via GitHub Actions now possible (repo is up). PyPI: not published; ask first.
- [ ] OOV net doubles letters sometimes ("hibari" -> hɪbˈɑɹɹi); samples use inline [hibari](/hibˈɑɹi/).

## Open (model)
- [ ] Re-run `python scripts/bench_all.py` on a quiet machine (Paradee did 22-24x when quiet).
- [ ] Listen to runs/compare_v3/ab*.wav and runs/g2p_e2e/*.wav (hibari G2P vs misaki G2P).
- [ ] Size: convs stay fp32 (ConvInteger 2x slower than fp32 Conv in ORT 1.29 x86; re-test on ARM / QLinearConv).
      ProsodyNet (c 192, 6 shared attention layers, 6 dilated ConvNeXt) is the next thing to shrink.
- [ ] Speed: int8 net 4.1 ms per s of audio, host DSP 3.7 ms (single-threaded numpy). C runtime idea: analytic pulse spectrum.
- [ ] int8 F0 differs from fp32 by 0.6 Hz mean / 5.4 Hz max; UTMOS equal. If pitch is off by ear, leave ProsodyNet heads fp32.
- [ ] Training schedule: v3 peaks at 40k with ProsodyNet. Shorter cosine or lower GAN weight late if retraining.
- [ ] Corpus is mostly declarative prose: add questions/exclamations/dialogue for prosody range.
- [ ] Measure on a real low-end device (1 thread, low clock); scipy.fft keeps float32; check pocketfft on ARM.
- [ ] Cross-sentence prosody: sentences are synthesized independently and concatenated (no extra pause inserted).

## Results (see docs/METRICS.md for conditions)
| model | MB | xRT 1 thr | UTMOS | WER |
| teacher Kokoro-82M | 327 | 1.9 | 4.52 | 2.8% |
| Paradee int8 | 9.04 | 18 | 4.41 | 2.8% |
| hibari int8 (v3 40k + ProsodyNet) | 7.94 | 128 | 4.43 | 2.7% |

## History
- Teacher dump: 30k sentences (66 h) in data/dist. compare_paradee.py uses sentences [30000:] that no model trained on.
- v1 (1.34M free-phase): UTMOS 3.63. v2 (c_dec 256 x 8): 3.74, harmonics broke up. v3 (pulse source): 10k 4.32;
  + ProsodyNet 40k 4.44 (pitch err 1.06 -> 0.47 st).
- Export: export/export_onnx.py (--prosody <ckpt>; durations and token->frame expansion in-graph, opset-11 ops; speed input
  added 2026-10-19). infer_numpy.py / src/hibari/dsp.py: numpy/scipy float32 host DSP, matches torch to_wav (tests/test_parity.py).
- G2P eval sets: g2p_work/eval.jsonl = shuffled corpus lines[31000:33500], dev = [33500:36500], tagger/OOV training from
  [40000:340000] + espeak labels of lexicon words and 192k corpus OOV words; e2e sentences = lines[30000:] first 50.
