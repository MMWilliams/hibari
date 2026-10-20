# kokoro-tiny / hibari TODO

Goal: Kokoro-82M distilled into single-voice students that are smaller and faster than Paradee-8M and as close to
the teacher as possible, for weak Mac/Windows CPUs. Product name: **hibari** (skylark). 0.1.0 is public (GitHub
MMWilliams/hibari, PyPI hibari-tts, HF maureesewilliams/hibari). 0.2.0 (voices) is built but NOT published.
Student = training/tiny_model.py: TinyKokoro v3 decoder (4.19M) + ProsodyNet (1.68M); 7.94 MB int8 ONNX per voice.

## 0.2.0 in progress (2026-10-20): more voices + British G2P + language study
- Voices: heart (af_heart, in the wheel), fenrir (am_fenrir, best US male by teacher UTMOS/WER), emma (bf_emma).
  Each voice = own decoder + ProsodyNet fine-tuned from heart's weights on a 15k-sentence teacher dump
  (data/dist_<voice>, 4.7 GB each): training/train_tiny.py --init 40k.pt --init-disc last.pt, train_prosody.py --init.
  Pipeline: logs/voices/chain.sh (one GPU-0 job at a time, RAM gate) then scripts/post_voice.py (snapshot by UTMOS,
  export, parity, eval_voice.py on 50 unseen sentences, stage_voices.py -> hf_stage/voices + assets/voices.json, demos).
  Results: runs/voices/<voice>/summary.json, ab*_teacher_student.wav, demo1-3.wav.
  emma: decoder 30k steps done 2026-10-20 00:00 (8.6 it/s), snapshot 15k by UTMOS (pnet 4.185 vs teacher 4.394 on
  held-out rows; teacher-forced 4.18-4.21, so the decoder, not prosody, is the gap); ProsodyNet pitch err 0.22 st.
  fenrir: ProsodyNet training (pitch err 0.71 st at 6k), decoder next. export/infer_numpy.py now feeds the speed input.
- [ ] Listen to runs/voices/*/ab*.wav and demo*.wav; then upload hf_stage/voices/ to HF (voices/ folder), re-upload
      hf/README.md, bump + publish 0.2.0 (gh workflow run publish.yml -f publish=true), GitHub release.
- [ ] Constraints added mid-run by the user: GPU 0 only (GPU 1 free), >= 31 GB RAM free, <= 2-3 heavy python processes,
      DataLoader workers <= 2 (train_tiny.py now uses 2).
- British G2P: src/hibari/g2p british=True (gb lexicon 1.5 MB + 3 suffix rules + oov_gb.onnx int8 0.94 MB):
  PER 0.57% / exact 85.9% on 1500 held-out sentences (US: 0.50% / 88.0%). US OOV net on British words: 15% word acc.
- Package: hibari.voices (registry assets/voices.json, HIBARI_HOME / OS cache dir, urllib download, sha256, atomic
  rename), Hibari(voice=...), CLI --voice/--list-voices, tests/test_voices.py (file:// downloads), British test cases.
- Languages: docs/LANGUAGES.md. Teacher (Whisper large-v3, 30 sentences): en 0.6% WER, es 3.4%, pt 2.3%, it 6.1%,
  fr 5.8%, hi 12.4%; ja 5.5% CER on readings, zh 4.4% CER (simplified only). Spanish rule G2P scripts/g2p_lang/es.py:
  PER 1.1% vs espeak on 1000 sentences, 12.5 KB, no lexicon; Italian it.py 7.8% (needs a lexicon). Verdict: Spanish go,
  Portuguese/Italian/French not yet, Hindi/Japanese/Mandarin no. Recommended next: Spanish (plan in the doc, ~4 days).
- [ ] Spanish: finish es.py (acronym spelling, exceptions, number separators), port to src/hibari/g2p/es, dump ef_dora
      (dump_teacher.py --voice ef_dora --lang e), fine-tune from heart, eval with Whisper large-v3 Spanish.

## Open (G2P)
- [ ] Tagger: "that" DT vs IN (12/2500), "in" RP vs IN (8); 1.24 MB pruned at |w|>=1.5 (97.46%).
- [ ] OOV nets: 72% (US) / 74% (GB) word accuracy vs espeak on unseen names; doubles letters sometimes ("hibari").
- [ ] Normalizer: "9/11" -> September eleventh; dates US style (also for the British voice); "St." Street/Saint.
- [ ] Load time: lexicon JSON gz 73 ms; tagger 95 ms; Hibari() incl. onnxruntime import ~0.5 s.
- [ ] Phase 2: native no-Python CLI/library (C/Rust). Release checklist: bump version in pyproject + __init__, push,
      `gh workflow run publish.yml -f publish=true`, attach the wheel to a GitHub release, re-upload hf/README.md.

## Open (model)
- [ ] Size: convs stay fp32 (ConvInteger slower in ORT 1.29 x86); ProsodyNet is the next thing to shrink.
- [ ] Speed: int8 net 4.1 ms per s of audio, host DSP 3.7 ms (single-threaded numpy).
- [ ] Corpus is mostly declarative prose: add questions/exclamations/dialogue for prosody range.
- [ ] Measure on a real low-end device (1 thread, low clock); cross-sentence prosody.

## Results (see docs/METRICS.md for conditions)
| model | MB | xRT 1 thr | UTMOS | WER |
| teacher Kokoro-82M | 327 | 1.9 | 4.52 | 2.8% |
| Paradee int8 | 9.04 | 18 | 4.41 | 2.8% |
| hibari heart int8 (v3 40k + ProsodyNet) | 7.94 | 128 | 4.43 | 2.7% |

## History
- Teacher dump: 30k sentences (66 h) in data/dist; compare_paradee.py uses sentences [30000:] that no model trained on.
- v1 (1.34M free-phase): UTMOS 3.63. v2 (c_dec 256 x 8): 3.74. v3 (pulse source): 10k 4.32; + ProsodyNet 40k 4.44.
- Export: export/export_onnx.py (--prosody <ckpt>; durations and expansion in-graph, speed input). infer_numpy.py /
  src/hibari/dsp.py: numpy/scipy float32 host DSP, matches torch to_wav (tests/test_parity.py).
- G2P eval sets: g2p_work/eval.jsonl = corpus lines[31000:33500] (US), eval_gb.jsonl [31000:32500] British, dev [33500:36500].
