# hibari model card

hibari (Japanese for skylark) is a family of 6.15 M parameter English text-to-speech models, one per voice.
Each is a student of one Kokoro-82M voice and runs about 130x realtime on a single CPU core from a 7.9 MB
int8 ONNX file, with a 24 kHz output. Voices: heart (af_heart, American female, in the package), fenrir
(am_fenrir, American male) and emma (bf_emma, British female), the last two downloaded on first use.

## Intended use

- Reading English text aloud on devices without a GPU: laptops, small servers, embedded boards, browsers.
- Applications that need low latency (12 ms for a short sentence on one core) and a small download.
- Research on distillation and small vocoders: the whole pipeline (teacher dump, training, export, evaluation)
  is reproducible from this repository.

Not intended for: voice cloning or impersonation (the voices are fixed and synthetic), languages other than
English, or any use where the listener is not told the speech is synthetic.

## Voices

Three fixed voices, each its own model with the architecture below (no speaker or style input; speaking
rate is adjustable through the speed input, 1.0 = teacher pace):

| name | Kokoro-82M voice | accent | gender | how it was trained |
|---|---|---|---|---|
| heart | af_heart | American | female | 30 k sentences (66 h), from scratch (three stages) |
| fenrir | am_fenrir | American | male | 15 k sentences, decoder and ProsodyNet fine-tuned from heart's weights |
| emma | bf_emma | British | female | 15 k sentences, fine-tuned from heart's weights; British G2P |

fenrir was chosen among the Kokoro-82M American male voices by scoring the teacher itself on 20 unseen
sentences (UTMOS / Whisper-base WER): am_fenrir 4.48 / 1.4%, am_puck 4.47 / 2.5%, am_adam 4.46 / 2.3%,
am_echo 4.45 / 2.0%, am_liam 4.44 / 1.4%, am_eric 4.41 / 2.5%, am_michael 4.36 / 2.0%, am_onyx 4.36 / 2.0%
(af_heart 4.52 / 2.0%; bf_emma 4.40 / 2.3%, bf_isabella 4.41 / 2.5%).

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

Text goes through hibari's own grapheme-to-phoneme front end (misaki's American or British English lexicon
and rules, an averaged-perceptron tagger for part-of-speech dependent words, and a 0.7 M parameter character
transformer for out-of-vocabulary words, one per accent) into Kokoro's phoneme alphabet.

## Training data

All audio was synthesized by Kokoro-82M: 30 k sentences (66 hours) of web text (4 to 35 words each) for
heart, the first 15 k of the same list for fenrir and emma, with the teacher's durations, F0 and energy
saved as targets. No human speech recordings were used. The G2P components were trained on misaki + spaCy
+ espeak-ng outputs for the text corpus (300 k sentences for the tagger, 418 k words for the American
out-of-vocabulary net, 383 k for the British one).

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

- Three fixed voices, English only. Non-English words are read with English letter-to-sound rules.
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
