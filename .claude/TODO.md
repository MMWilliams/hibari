# kokoro-tiny / hibari TODO

Goal: Kokoro-82M distilled into single-voice students that are smaller and faster than Paradee-8M and as close to
the teacher as possible, for weak Mac/Windows CPUs. Product name: **hibari** (skylark). 0.1.0 is public (GitHub
MMWilliams/hibari, PyPI hibari-tts, HF maureesewilliams/hibari). 0.2.0 (five voices) published 2026-10-20 05:45.
Student = training/tiny_model.py: TinyKokoro v3 decoder (4.19M) + ProsodyNet (1.68M); 7.94 MB int8 ONNX per voice.

## In progress 2026-10-20: close the gap on fenrir / fable / emma (session 01GnXdCxRp9e3p4vPLT7uCVX)
- Diagnosis done (scripts/voice_gap.py, voice_swap.py; runs/voice_gap/; write-up docs/VOICE_GAP.md): the whole gap is in
  voiced frames below 4 kHz for every voice; fricatives, silence, transitions, high band are fine. Mel LSD / MCD / harmonic
  amplitude / periodicity / HNR do not separate the weak voices from heart. The noise branch is dead (0 %); the free branch
  carries unvoiced sounds and 23-38 % of voiced 1-4 kHz energy with structured phases (random phases cost 0.1-0.6 UTMOS).
- Lever 1 done: linear-frequency log-STFT L1 (--lin-w 20 --hi-w 0.25), emma 10k of 30k: teacher-forced 4.184 vs 4.174,
  pnet 4.170 vs 4.134, MCD 9.34 vs 9.54 (runs/exp/emma_lin20, scores in runs/exp/gap/results.json): small, within noise.
- Lever 2 done: free-branch share penalty (--free-w 10): share 37 % -> 19 % but UTMOS flat (+0.01/+0.03), MCD worse: no.
  British-only tokens (6.5 % of frames) are not where the gap is (splice test with control).
- Lever 3 done: fresh discriminators: MCD 9.54 -> 8.22 (all 50 better) but UTMOS -0.05/-0.07: UTMOS wants MORE
  adversarial pull, not less. train_tiny.py --gan-w added.
- Lever 4 done: --gan-w 2: UTMOS -0.04/-0.01, MCD +1.2: no. Shipped loss balance is the best point.
- Export finding: per-channel int8 (quantize_dynamic per_channel=True) on the shipped fenrir: UTMOS 4.385 vs 4.349
  (fp32 4.380), WER 2.4 % vs 2.9 %, 171x, 8.03 MB (+0.09): recovers the whole int8 loss. Testing on emma/fable.
- Capacity probe done (runs/exp/emma_cap12, 12 blocks): +0.03 UTMOS at 10k, MCD +0.1, +1.6 MB, ~45 % slower: report only.
- emma v2 full run (lin-w 20, 30k sentences, 30k steps): 50 unseen int8 4.144 / 2.8 % / MCD 11.38 vs shipped 4.161 / 2.6 % /
  11.01: NOT better, shipped emma stays (runs/voices/bf_emma_v2, export/bf_emma_v2).
- [ ] fable v2 running (09:22-10:25), then fenrir v2; fenrir per-channel-only candidate staged from export/am_fenrir_pc.
- [ ] Full runs script: logs/voices/full.sh "--lin-w 20 --hi-w 0.25"
      bf_emma bm_fable am_fenrir (30k steps each, post_voice --no-stage, MCD) -> runs/voices/<v>_v2, export/<v>_v2.
- [ ] Then: capacity probe, 12 decoder blocks (dec.8-11 start near identity from heart's init), 10k steps, report only.
- Second metric for the report: scripts/mcd_eval.py (DTW MCD on the 50 unseen sentences): heart 12.44, kore 10.54,
  fenrir 10.97, fable 13.04, emma 11.01 (v1 11.39, v3s 10.72): MCD does not track UTMOS. logs/voices/cand.sh resumes a
  candidate to 30k and runs post_voice.py --no-stage.
- [ ] If it helps: finish 30k (--resume), post_voice.py --run ... --no-stage, A/B wavs to runs/voices/<voice>_v2/, then
      fenrir and fable; stage in hf_stage/voices_v2 with new file names, registry entries, 0.2.1 prepared, not released.
- [ ] If not: lever 2 = fresh discriminators (no --init-disc), then re-diagnose.

## 0.2.0 published 2026-10-20 (PyPI hibari-tts 0.2.0, GitHub release v0.2.0, HF voices/ + samples/): voices + British G2P + language study
- Voices: heart (af_heart, in the wheel), fenrir (am_fenrir, best US male by teacher UTMOS/WER), emma (bf_emma).
  Each voice = own decoder + ProsodyNet fine-tuned from heart's weights on a 15k-sentence teacher dump
  (data/dist_<voice>, 4.7 GB each): training/train_tiny.py --init 40k.pt --init-disc last.pt, train_prosody.py --init.
  Pipeline: logs/voices/chain.sh (one GPU-0 job at a time, RAM gate) then scripts/post_voice.py (snapshot by UTMOS,
  export, parity, eval_voice.py on 50 unseen sentences, stage_voices.py -> hf_stage/voices + assets/voices.json, demos).
  Results: runs/voices/<voice>/summary.json, ab*_teacher_student.wav, demo1-3.wav.
  emma: decoder 30k steps done 2026-10-20 00:00 (8.6 it/s), snapshot 15k by UTMOS (pnet 4.185 vs teacher 4.394 on
  held-out rows; teacher-forced 4.18-4.21, so the decoder, not prosody, is the gap); ProsodyNet pitch err 0.22 st.
  fenrir: done 01:18 (decoder resumed at 12k after a CUDA OOM caused by dumps sharing GPU 0): snapshot 30k; 50 unseen
  sentences: int8 UTMOS 4.35 vs teacher 4.48, WER 2.9% vs 2.6%, 166x realtime; staged + registered.
  emma v2 (GPU 1, 29.8k rows, 60k steps from 15k.pt): snapshot 40k; 50 unseen: int8 4.16 / 2.6% vs v1 4.14 / 2.7% ->
  kept (runs/voices/bf_emma = v2, bf_emma_v1 = first run; export/bf_emma = v2). emma v3s experiment on GPU 1 from 03:27:
  heart's third stage redone on emma data (init runs/tiny_v2/snap/60k.pt, 60k steps): 60k snapshot int8 4.09 / 2.5% -> lost to v2.
  fable (bm_fable): done 02:54, snapshot 25k, staged + registered (numbers in runs/voices/bm_fable/summary.json).
  kore (af_kore): done 04:09, snapshot 25k; 50 unseen: int8 4.46 / 2.5% vs teacher 4.52 / 2.4%, 170x; staged + registered.
  fable (bm_fable): int8 4.17 / 2.6% vs teacher 4.33 / 2.6%, 167x. All four pass the gate (WER within 0.5, parity, demos).
- Publish status 2026-10-20 05:25: HF voices/ + samples/ + README uploaded and verified by anonymous download (sha256 OK);
  CI green on all 6 jobs; tests 57 pass, wheel test pass; publish workflow green (PyPI 0.2.0 at 05:42); GitHub release
  v0.2.0 with the PyPI wheel; scripts/check_release.sh 0.2.0 OK (fresh venv, all voices downloaded, Dr. Smith + date read right).
- [ ] User: listen to runs/voices/*/ab*.wav and demo*.wav in the morning and give feedback.
- Overnight 2026-10-20 00:25 (user via tts-fast session; logs/fable_task5.md): limits lifted, both GPUs, keep ~6 GB RAM
  free, and PUBLISH 0.2.0 on the user's behalf once validated (tests, wheel test, HF upload + sha check, push, CI,
  PyPI, GitHub release, fresh-venv install check). Plan: GPU 0 fenrir -> bm_fable (teacher 4.32/2.0%) -> af_kore
  (4.51/1.4%); GPU 1 emma second try (30k sentences, 60k steps from 15k.pt, logs/voices/emma_long.sh, kept only if
  better on the same 50 sentences). Lanes: logs/voices/lane.sh; progress in logs/voices/chain.log.
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
