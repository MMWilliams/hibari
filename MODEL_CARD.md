# hibari model card

hibari (Japanese for skylark) is a 6.15 M parameter American English text-to-speech model with one voice.
It is a student of Kokoro-82M (voice af_heart) and runs about 130x realtime on a single CPU core from a
7.9 MB int8 ONNX file, with a 24 kHz output.

## Intended use

- Reading English text aloud on devices without a GPU: laptops, small servers, embedded boards, browsers.
- Applications that need low latency (12 ms for a short sentence on one core) and a small download.
- Research on distillation and small vocoders: the whole pipeline (teacher dump, training, export, evaluation)
  is reproducible from this repository.

Not intended for: voice cloning or impersonation (the voice is fixed and synthetic), languages other than
English, or any use where the listener is not told the speech is synthetic.

## Voice

One female American English voice, distilled from Kokoro-82M's af_heart. There is no speaker or style input.
Speaking rate is adjustable (speed input, 1.0 = teacher pace).

## Model

| part | what | parameters |
|---|---|---|
| ProsodyNet text encoder | 6 shared ALBERT-style attention layers (c 192) over phoneme ids | 1.68 M (with heads) |
| ProsodyNet heads | per-token durations; 80 Hz F0 and energy from 6 dilated ConvNeXt blocks | |
| Decoder text encoder | ConvNeXt front + 4 shared attention layers (c 128) | 0.29 M |
| Decoder | 8 ConvNeXt blocks at 80 Hz (c 256, ff 768) on text features + F0 + energy | 3.41 M |
| Output head | linear to 5 x 601 spectral parts per frame | 0.77 M |

The network stops at a parametric spectrum per 12.5 ms frame: a harmonic branch (log magnitude and phase
rotation applied to the STFT of a band-limited pulse train built from F0), a noise branch, and a free-phase
branch. The host turns it into audio with one inverse STFT (n_fft 1200, hop 300). This keeps the per-sample
work out of the network, which is where Kokoro-82M spends most of its time.

Text goes through hibari's own grapheme-to-phoneme front end (misaki's American English lexicon and rules,
an averaged-perceptron tagger for part-of-speech dependent words, and a 0.7 M parameter character
transformer for out-of-vocabulary words) into Kokoro's phoneme alphabet.

## Training data

All audio was synthesized by Kokoro-82M: 30 k sentences (66 hours) of web text (4 to 35 words each), with the
teacher's durations, F0 and energy saved as targets. No human speech recordings were used. The G2P
components were trained on misaki + spaCy + espeak-ng outputs for the text corpus (300 k sentences for the
tagger, 418 k words for the out-of-vocabulary net).

## Evaluation

50 unseen sentences (never synthesized in training), UTMOS (SpeechMOS utmos22_strong, higher is better) and
word error rate of Whisper base transcripts, one onnxruntime thread. Full sheet: docs/METRICS.md.

| model | file MB | x realtime (1 thread) | UTMOS | WER |
|---|---|---|---|---|
| Kokoro-82M teacher (fp32, torch) | 327 | 1.9 | 4.52 | 2.8% |
| Paradee-8M int8 | 9.04 | 18 | 4.41 | 2.8% |
| hibari int8 | 7.94 | 128 | 4.43 | 2.7% |

Grapheme-to-phoneme: see README (G2P section) for phoneme error rate and exact match against the
reference pipeline, and for the end-to-end UTMOS / WER with hibari's G2P versus the reference G2P feeding
the same model.

## Limitations

- One voice, English only. Non-English words are read with English letter-to-sound rules.
- Trained on declarative web prose: questions and exclamations are read correctly but with a narrower
  prosodic range than the teacher.
- Sentences are synthesized one at a time; there is no cross-sentence prosody.
- Out-of-vocabulary words (about 1 in 5 sentences has one) are read by a small network; names from
  languages other than English can come out anglicized or wrong.
- Numbers are read in American conventions (10/12 is October 12; dates and times are read in US style).
- The int8 export quantizes the attention and linear layers only; its pitch differs from the fp32 export by
  under 1 Hz on average with no measurable quality difference.

## License

Apache-2.0, like Kokoro-82M and misaki. See NOTICE for attributions.
