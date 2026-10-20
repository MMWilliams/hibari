# Why fenrir, fable and emma sound worse than heart and kore

Written 2026-10-20 while working on 0.2.1. Scripts: `scripts/voice_gap.py` (measurements), `scripts/voice_swap.py`
(oracle swaps), both on the same 50 held-out training-dump rows per voice, student driven with the teacher's own
durations, pitch and energy so that only the decoder differs. Numbers are in `runs/voice_gap/`. "Gap" below means the
teacher's UTMOS minus the student's on those rows (the shipped numbers on 50 unseen sentences are a little larger
because they also include the ProsodyNet and the int8 export).

## The short version

- The whole loss is in the **voiced** parts of speech (vowels and voiced consonants), **below 4 kHz**. Splicing the
  teacher's voiced frames into the student's output recovers 96 to 101 % of the gap for every voice; splicing the
  teacher's unvoiced frames (s, sh, f, t...), its silences, its phoneme boundaries or its high band recovers nothing.
- Nothing we normally measure sees it. Mel-spectral distance, mel-cepstral distortion, per-harmonic amplitude error,
  harmonic-to-noise ratio, periodicity, pulse shape, energy envelope, silence floor, errors at transitions: all of these
  are about the same for the five voices, and heart (the best voice) is often the worst on them. Spectrograms of the
  worst sentences look identical to the teacher's.
- The decoder is not using its parts the way they were designed. The random-noise branch is dead (0.0 to 0.1 % of
  the energy in every voice). The "free-phase" branch, meant for transients, carries all unvoiced sounds and a quarter to
  a third of the voiced energy between 1 and 4 kHz, with phases that are learned, not random: replacing them with random
  phases costs 0.1 to 0.6 UTMOS. In other words, in the 1 to 4 kHz band the decoder builds part of each harmonic with the
  pulse train and part with a second, free-running source that cannot be phase-locked to the pulses. That is the kind of
  fine structure UTMOS reacts to and the mel losses cannot see: above about 2 kHz for a male voice and 4 kHz for a
  female one, a mel band is wider than the spacing between harmonics, so the harmonic peaks and the valleys between them
  are averaged into one number.
- The damage is local. In the worst sentences, 3-second windows differ by 0.4 to 0.7 UTMOS while the rest of the
  sentence differs by 0.1 to 0.3, and longer sentences lose more (emma: 0.13 on sentences under 7 s, 0.28 on longer
  ones; heart: 0.09 on both). Longer sentences simply contain more of these spots.
- Pitch is not the problem. The teacher's harmonics sit within about 1 Hz (in F0 terms) of the pulse train's harmonics
  for all voices; heart has the largest offsets of the five and the smallest gap. The 40 Hz voicing threshold of the
  pulse train and Kokoro's sine source being on at any F0 above zero do not matter either: the frames where that
  differs are quiet, and swapping in the teacher's unvoiced frames recovers nothing.
- Nothing is wrong with the teacher data: no clipping (peaks 0.45 to 0.75), no odd rows.

## The measurements (50 held-out rows per voice, teacher-forced student)

| voice | UTMOS teacher / student / gap | MCD dB | mel LSD low / mid / high dB | level bias low / mid / high dB | periodicity teacher / student 0-1k, 1-4k | pulse branch share of voiced 1-4 kHz | free-phase share |
|---|---|---|---|---|---|---|---|
| heart | 4.53 / 4.44 / 0.09 | 9.31 | 3.16 / 2.97 / 2.15 | -0.36 / -0.22 / +0.01 | 0.83 / 0.84, 0.35 / 0.33 | 62 % | 38 % |
| kore | 4.52 / 4.48 / 0.04 | 8.57 | 2.80 / 2.76 / 1.92 | -0.21 / -0.16 / -0.08 | 0.81 / 0.82, 0.37 / 0.39 | 71 % | 29 % |
| fenrir | 4.50 / 4.40 / 0.09 | 8.89 | 2.92 / 2.79 / 1.95 | -0.10 / -0.03 / -0.07 | 0.73 / 0.74, 0.32 / 0.34 | 77 % | 23 % |
| fable | 4.33 / 4.21 / 0.13 | 10.18 | 3.13 / 2.99 / 2.12 | -0.10 / -0.18 / +0.00 | 0.66 / 0.68, 0.30 / 0.30 | 68 % | 32 % |
| emma | 4.40 / 4.18 / 0.22 | 8.88 | 2.94 / 2.90 / 2.05 | -0.26 / -0.09 / -0.06 | 0.84 / 0.84, 0.36 / 0.38 | 63 % | 37 % |

Bands: low under 1 kHz, mid 1 to 4 kHz, high 4 to 12 kHz. Mel LSD = root-mean-square difference of the 80-band mel
spectrum in dB; bias = its mean (negative: student quieter). MCD = mel-cepstral distortion, 12 coefficients. Periodicity
= normalised autocorrelation peak at the pitch period after band-passing. Noise-branch share: 0 % everywhere.

Other things checked, none of which separate the weak voices from heart and kore:

- Harmonic amplitudes (peak-picked, 1 to 4 kHz): 3.6 to 4.4 dB RMS error for every voice; the student's valleys
  between harmonics are 0.5 to 1.5 dB deeper than the teacher's for every voice (slightly "cleaner" than the teacher).
- Pulse shape (best-lag waveform correlation of the band under 1 kHz): 0.92 to 0.97 for all voices.
- Energy envelope: 1.1 to 1.3 dB mean absolute error, correlation 0.97 to 0.98 for all; fable's student is 0.6 dB
  too quiet in unvoiced frames, the only envelope difference worth noting.
- Silence: the student's level in the teacher's silent frames is -92 to -106 dBFS, the same as the teacher's; in pause
  interiors and at pause edges the two match within 0.5 dB. Fable's teacher has no pauses at all (its noise floor stays
  above -55 dB).
- Transitions: spectral error at phoneme boundaries is 8 to 12 % above the steady-state error for heart, kore, fenrir and
  emma, 17 % for fable (mel-cepstral: 23 %). Fable is also the only voice with extra error around voicing changes
  (15 %). It is a real but second-order difference.
- Pitch: measured pitch (pyin) deviates from the F0 curve by the same 0.4 to 0.5 semitones for teacher and student (that
  is the tracker's own error); gross errors under 1 %. Teacher harmonic offsets from h x F0: 1.0 to 2.5 Hz.

## What the UTMOS probes said

Scoring the student's own output with one part changed (same held-out rows):

| change | heart | kore | fenrir | fable | emma |
|---|---|---|---|---|---|
| as trained | 4.44 | 4.49 | 4.40 | 4.21 | 4.18 |
| free-phase branch removed | 3.39 | 3.83 | 3.61 | 3.16 | 3.30 |
| free-phase branch with random phases | 3.68 | 4.35 | 4.04 | 3.68 | 3.56 |
| free-phase branch removed below 1 kHz in voiced frames | 4.43 | 4.47 | 4.38 | 4.13 | 4.08 |
| free-phase branch removed below 4 kHz in voiced frames | 4.07 | 4.29 | 4.11 | 3.80 | 3.75 |
| harmonic phase rotation set to zero | 4.28 | 4.24 | 4.06 | 4.06 | 3.86 |
| harmonic amplitudes smoothed over 5 frames | 3.80 | 3.85 | 3.63 | 3.51 | 3.37 |
| 1 % random pitch jitter in the pulse train | 4.43 | 4.46 | 4.38 | 4.18 | 4.07 |
| teacher | 4.53 | 4.52 | 4.50 | 4.33 | 4.40 |

Teacher's frames or bands spliced into the student (share of the gap recovered; negative means the splice itself hurt):

| spliced from the teacher | heart | kore | fenrir | fable | emma |
|---|---|---|---|---|---|
| voiced frames | 100 % | 100 % | 100 % | 96 % | 101 % |
| unvoiced frames | 0 % | -4 % | -2 % | -4 % | -2 % |
| silent frames | -14 % | -5 % | -25 % | -4 % | -12 % |
| voicing transitions | 0 % | -1 % | -2 % | -16 % | -3 % |
| student's high band (4 to 12 kHz) put into the teacher, share of the gap lost | 10 % | 2 % | 8 % | 6 % | 9 % |
| student's low band (under 1 kHz) put into the teacher | 109 % | 199 % | 110 % | 199 % | 98 % |
| student's mid band (1 to 4 kHz) put into the teacher | 83 % | 145 % | 92 % | 211 % | 77 % |

## What this means for fixing it

The decoder's error is a texture error inside voiced sounds between the fundamental and 4 kHz: how each harmonic is
built (pulse train versus free-running source) and how its phase and amplitude move from frame to frame. The losses that
train the decoder (L1 on a 100-band mel spectrogram, a mel spectral-convergence term, and the discriminators) only
constrain this through the discriminators, which were trained on heart and fine-tuned at half the learning rate for
the new voices. Levers that act on exactly this region: a linear-frequency spectral loss that resolves every harmonic
(train_tiny.py --lin-w), and discriminator handling. Levers that act elsewhere (high-band emphasis, silence, F0, data
filtering, more data or steps) are not expected to help, and the last two were already tried for emma.

Results of the lever experiments: see the end of this file.

## Lever experiments (emma, 15 k-sentence dump, 10 k of a 30 k-step schedule from heart's weights, same held-out rows)

| run | UTMOS teacher-forced | UTMOS with ProsodyNet | MCD dB |
|---|---|---|---|
| shipped recipe, 10 k steps (runs/voices/bf_emma_v1) | 4.174 | 4.134 | 9.54 |
| shipped recipe, 15 k steps | 4.175 | 4.185 | 9.47 |
| lever 1: linear-frequency log-STFT L1, weight 20, bins above 4 kHz at 0.25 (--lin-w 20 --hi-w 0.25), 10 k | 4.184 | 4.170 | 9.34 |

Lever 1 improves the second metric (MCD 9.54 to 9.34 dB) and UTMOS by 0.01 to 0.04, which is within the noise of 50
rows and far from the 0.22 gap. Kept as a candidate for the full run only if nothing better turns up.
| lever 2: penalty on the free-phase branch's share of voiced energy below 4 kHz (--free-w 10), 10 k | 4.183 | 4.163 | 9.67 |

Lever 2 did what it was meant to do (free-phase share of the voiced 1 to 4 kHz energy down from 37 % to 19 %, pulse
branch up to 80 %, noise branch awake at 1 %) and it changed nothing: UTMOS +0.01 / +0.03 (27 of 50 sentences better,
a coin flip), MCD worse by 0.14 dB on 41 of 50 sentences. So where the voiced harmonics come from is not what UTMOS
objects to. Also checked: the gap is not in the frames of British phoneme tokens that heart never saw ('ː', 'a', 'ɒ',
'Q': 6.5 % of emma's and fable's frames; splicing the teacher's frames for them helps exactly as little as splicing
the same amount of ordinary frames).
| lever 3: fresh discriminators (no --init-disc, 2 k warm-up), 10 k | 4.106 | 4.086 | 8.22 |

Lever 3 is the most informative miss: with weak, freshly started discriminators the decoder follows the spectral losses
much more closely (MCD 9.54 to 8.22 dB, better on all 50 sentences, mel loss 0.30 to 0.27) and UTMOS falls by 0.05 to
0.07 (worse on 36 of 50 sentences). The two judges pull in opposite directions, and the thing UTMOS rewards is the
texture the adversarial term produces. Heart's discriminators, trained for 60 k steps, are an asset, and the natural
next lever is to weight the adversarial term up (--gan-w).
| lever 4: adversarial term weighted x2 with heart's discriminators (--gan-w 2), 10 k | 4.135 | 4.122 | 10.77 |

Lever 4 is worse on both judges (UTMOS -0.04 / -0.01, MCD +1.2 dB on all 50 sentences): weaker adversarial pull hurts
(lever 3) and stronger hurts too, so the shipped balance between the spectral losses and the discriminators is already
the best of these for UTMOS. The only lever with a consistent gain is lever 1.

## A side finding on the int8 export

fenrir loses 0.03 UTMOS from int8 quantization (fp32 4.380, int8 4.349 on the 50 unseen sentences), more than the other
voices (0.01 to 0.02). Per-channel weight scales in the same dynamic int8 export (export_onnx.py --per-channel) bring
every voice's int8 model to its fp32 level: fenrir 4.385 (+0.037, 27 of 50 sentences better, WER 2.4 % instead of
2.9 %), emma 4.173 (+0.012), fable 4.160 (-0.007, noise), same speed, 8.03 MB instead of 7.94. The int8 outputs are
also measurably closer to the fp32 model (harmonic log-amplitude error 0.04 instead of 0.05, F0 0.4 Hz instead of
0.7 Hz on rows with the same frame count).

## Capacity probe (report only, not shipped)

A decoder with 12 ConvNeXt blocks instead of 8 (the four new blocks start near identity, everything else from heart's
weights; 6.24 M parameters instead of 4.65 M, about 1.6 MB more on disk and about 45 % slower on one CPU thread in the
torch benchmark: RTF 0.0159 vs 0.0110), same emma recipe, 10 k of 30 k steps: UTMOS 4.202 / 4.162 (teacher-forced /
with ProsodyNet) against 4.174 / 4.134 for the shipped-size decoder at the same step, MCD 9.65 vs 9.54 dB. A gain of
0.03, the same size as lever 1's, for half again as much decoder. Capacity is not what is missing either, or not in a
way that 10 k steps can show.

## Full run with lever 1 (emma, 30 k sentences, 30 k steps, --lin-w 20 --hi-w 0.25)

Snapshot by UTMOS on held-out rows: 10 k 4.168, 15 k 4.158, 20 k 4.165, 25 k 4.166, 30 k 4.148 (flat, best at 10 k).
On the 50 unseen sentences: int8 4.144 / WER 2.8 % / MCD 11.38 dB against the shipped emma's 4.161 / 2.6 % / 11.01 dB.
Not better, so the shipped emma stays. The 10 k-step gain did not carry over to unseen sentences; all of these
differences are within about 0.03, the noise of a 50-sentence UTMOS mean.

## Full run with lever 1 (fable, 15 k sentences, 30 k steps)

Held-out UTMOS with ProsodyNet: 10 k 4.147, 15 k 4.157, 20 k 4.162, 25 k 4.164, 30 k 4.166 (the shipped run peaked at
4.176). On the 50 unseen sentences: int8 4.143 / WER 2.5 % / MCD 13.00 dB against the shipped fable's 4.167 / 2.6 % /
13.04 dB. Not better; the shipped fable stays.
