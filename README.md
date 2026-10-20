# hibari

Tiny American English text-to-speech. One 7.9 MB model, about 130x realtime on a single CPU core, text in,
24 kHz audio out. Pure Python package with two dependencies (numpy, onnxruntime); runs on Windows, macOS and
Linux, x64 and arm64. Distilled from Kokoro-82M (voice af_heart), it scores UTMOS 4.43 against the teacher's
4.52 while running 65x faster.

hibari is Japanese for skylark. Model files, checkpoints and audio samples:
https://huggingface.co/maureesewilliams/hibari

## Install

```
pip install hibari
pip install "hibari[fast]"        # adds scipy: 2x faster signal processing
```

Python 3.9 or newer. Everything (model, lexicon, phonemizer) is inside the wheel, nothing is downloaded.

## Command line

```
hibari "Hello there, this is hibari." -o hello.wav
echo "Reads standard input too." | hibari -o out.wav
hibari "Say it out loud." --play                  # winsound / afplay / aplay or paplay
hibari "Faster, please." --speed 1.25 -o fast.wav
hibari "Hello there." --phonemes                  # həlˈO ðˈɛɹ.
hibari --bench                                    # load time, x realtime, G2P time
```

## Python

```python
from hibari import Hibari

tts = Hibari()                        # loads once (about 0.5 s with imports)
audio = tts("Hello there.")           # numpy float32, 24 kHz, in [-1, 1]
tts.save("Hello there.", "hello.wav") # 16-bit WAV

for chunk in tts.stream(long_text):   # one array per sentence, as soon as it is ready
    play(chunk)

tts("Same text, same audio.", seed=0)             # reproducible noise component
tts("Slower.", speed=0.8)                         # 0.5 .. 2.0, 1.0 = teacher pace
tts.phonemize("Hello there.")                     # ['həlˈO ðˈɛɹ.']
tts.from_phonemes("həlˈO ðˈɛɹ.")                  # synthesize phonemes directly
```

`Hibari(threads=4)` uses more onnxruntime threads (the signal processing stays single-threaded, so the gain
is about 30%). Through this API on 20 held-out sentences (160 s of audio, `scripts/bench_package.py`): 175x
realtime on one thread and 225x on four on a Ryzen 9 9900X under Windows; 210x / 300x in WSL Ubuntu on the
same machine. Short sentences cost about 12 ms each (the 128x figure in the table below is on a mix of
short and long sentences on a busier machine). `Hibari(model=path)` loads another export with the same interface (for example the fp32 one).

Custom pronunciations inline, misaki style: `tts("Say [hibari](/hibˈɑɹi/) now.")`.

## Text to phonemes

The model reads misaki-style phoneme strings (the ones Kokoro-82M was trained on). hibari produces them
without spaCy or espeak-ng:

1. Normalization (`hibari/g2p/normalize.py`, tables in `assets/normalize.json`): times (9:30am), dates
   (10/12/2019, Jan. 5th, 2019-10-12), units (5km, 20°C, 3 hrs), abbreviations (Dr., St., approx., vol.),
   currencies the lexicon does not know (¥, Rs, ¢, "$3.2 billion"), roman numerals after chapter/king words,
   ranges (2-3), phone numbers, fractions, symbols.
2. Tokenizer with spaCy's English rules and special cases (`tokenize.py`, `assets/tokenizer_exceptions.json`).
3. Averaged-perceptron part-of-speech tagger trained on spaCy's tags (`tagger.py`, `assets/tagger.json.gz`),
   97.7% agreement with spaCy; it only matters for the lexicon's part-of-speech entries (read, live, record,
   present...) and function-word rules (a, the, to, in, by, used).
4. misaki's lexicon and rules, ported (`lexicon.py`, `assets/lexicon_us_gold.json.gz` and silver): numbers,
   ordinals, years, decimals, $/£/€, percent, stems (-s, -ed, -ing), stress, context (the/to before vowels).
5. Out-of-vocabulary words: a 0.7 M parameter character transformer (`assets/oov.onnx`, 2.8 MB) trained on
   espeak-ng's readings of 418 k words, so unknown words sound like the reference pipeline's.

Accuracy against the reference pipeline (misaki 0.9.4 + spaCy en_core_web_sm + espeak-ng) on 2,500 held-out
corpus sentences (`scripts/g2p/eval_g2p.py g2p_work/eval.jsonl`):

| | phoneme error rate | sentence exact match |
|---|---|---|
| all sentences, normalization on | 0.50% | 88.0% |
| all sentences, normalization off (pure misaki emulation) | 0.27% | 90.9% |
| sentences without out-of-vocabulary words (80%), normalization on / off | 0.15% / 0.05% | 95.4% / 97.5% |
| sentences with out-of-vocabulary words (20%), normalization on | 1.65% | 58.3% |

With normalization off the output is a re-implementation of misaki, so the residual differences are the
tagger (97.5% agreement with spaCy; "that" and "in" are the common cases) and the out-of-vocabulary net
(72% of unseen words identical to espeak-ng, 9% phoneme error rate on them). With normalization on, most
of the added differences are deliberate: dates read as "October twelfth", units and abbreviations spelled
out, times read as times. The hard set in `scripts/g2p/hard_set.txt` shows both side by side
(`g2p_work/hard_set_results.txt` after running `scripts/g2p/eval_g2p.py`). End to end, the same model fed by hibari's G2P and by the reference G2P on 50
unseen sentences (47 identical phoneme strings): UTMOS 4.431 vs 4.431, Whisper WER 2.6% vs 2.7%
(`scripts/g2p/eval_e2e.py`). Speed: 0.5 to 1 ms per sentence (0.25 ms more per out-of-vocabulary word),
about 0.3 s to load the G2P (lexicon 73 ms, tagger 95 ms, OOV net 25 ms).

## ONNX interface (for ports)

`hibari/assets/hibari.onnx`, opset 17, int8 dynamic MatMul weights, CPU execution provider.

| name | shape | type | meaning |
|---|---|---|---|
| ids (input) | [1, L] | int64 | phoneme ids from `assets/vocab.json`, with id 0 at both ends |
| speed (input) | [1] | float32 | durations are divided by this; 1.0 = as trained |
| lh, theta, ln, lr, phi (outputs) | [1, 601, F] | float32 | spectral parts per 12.5 ms frame (F frames, 601 bins) |
| f0 (output) | [1, F] | float32 | pitch in Hz per frame, 0 where unvoiced |

Audio from the outputs (`hibari/dsp.py`, 200 lines of numpy): E = STFT(pulses(f0)) with a Hann window of
1200 and hop 300, divided by 300; then

    S = exp(min(lh, 7)) * exp(i * pi * tanh(theta)) * E  +  exp(min(ln, 7)) * Z  +  exp(min(lr, 7)) * exp(i * phi)

with Z complex Gaussian noise (unit variance per component); the waveform is the inverse STFT of S
(center=True, length F * 300). `pulses` linearly interpolates f0 to 24 kHz and emits
sum_{h <= 11000/f} cos(h * phase) in closed form, zero where f0 < 40 Hz. Everything in the graph is
conv1d, matmul, layernorm, gelu, softmax and the opset-11 ops for the duration expansion, so it runs under
onnxruntime-web, Android and iOS too.

## For researchers

Architecture: `training/tiny_model.py` (TinyKokoro v3 decoder with a pulse source; ProsodyNet). The decoder
and ProsodyNet were trained separately on Kokoro-82M's outputs; the export puts ProsodyNet's durations, F0
and energy and the decoder in one graph. Metrics and conditions: `docs/METRICS.md`; model card:
`MODEL_CARD.md`.

| model | params | file MB | x realtime 1 thr | 4 thr | latency short / long ms | peak RSS MB | UTMOS | WER |
|---|---|---|---|---|---|---|---|---|
| hibari int8 | 6.15 M | 7.94 | 128 | 175 | 12 / 71 | 158 | 4.43 | 2.7% |
| hibari fp32 | 6.15 M | 24.75 | 104 | 167 | 15 / 76 | 171 | 4.43 | 2.7% |
| Paradee-8M int8 | 9.06 M | 9.04 | 18.3 | 27 | 87 / 602 | 176 | 4.41 | 2.8% |
| Kokoro-7M-Distill fp32 | 7.48 M | 30.10 | 17.4 | 27 | 108 / 357 | 889 | | |
| Kokoro-82M teacher | 81.81 M | 327 | 1.9 | 3.9 | 739 / 3880 | 1527 | 4.52 | 2.8% |

Reproduce (research environment: `.venv` with torch, kokoro, misaki, spaCy, espeak-ng; GPU):

```
# 0. text corpus (agentlans/high-quality-english-sentences on Hugging Face)
curl -L -o data/sentences.txt.gz https://huggingface.co/datasets/agentlans/high-quality-english-sentences/resolve/main/train.txt.gz
# 1. teacher dump: 30k sentences through Kokoro-82M with durations, F0, energy  (data/dist; one process per shard)
python training/dump_teacher.py --n 30000 --shard 0 --nshards 3   # and --shard 1, --shard 2
# 2. decoder in three stages, each starting from the previous one's 60k snapshot (snapshots every 10k)
python training/train_tiny.py --out runs/tiny_v1
python training/train_tiny.py --out runs/tiny_v2 --steps 100000 --model '{"c_dec":256,"dec_blocks":8,"dec_ff":768}' --init runs/tiny_v1/snap/60k.pt
python training/train_tiny.py --out runs/tiny_v3 --steps 60000 --model '{"c_dec":256,"dec_blocks":8,"dec_ff":768,"src":true}' --init runs/tiny_v2/snap/60k.pt
#    after step 3, pick the v3 snapshot by UTMOS with ProsodyNet (40k won): scripts/diagnose_gap.py 30k 40k 50k 60k --run runs/tiny_v3 --prosody runs/prosody_v1/best.pt
# 3. ProsodyNet on whole sentences (18 min on one GPU)
python training/train_prosody.py --out runs/prosody_v1
# 4. export: one graph with a speed input, fp32 + int8, parity check against torch
python export/export_onnx.py --ckpt runs/tiny_v3/snap/40k.pt --prosody runs/prosody_v1/best.pt --out export/v3_speed/tiny.onnx
# 5. evaluation: head to head on unseen sentences (UTMOS, Whisper WER), size and speed sheet
#    (Paradee-8M ONNX in upstream/paradee/onnx, Kokoro-7M-Distill in upstream/kokoro-7m, downloaded separately)
python scripts/compare_paradee.py --tiny export/v3_speed --n 50
python scripts/bench_all.py
# 6. G2P: reference dump, tagger, OOV net, evaluation
python scripts/g2p/dump_misaki.py --start 40000 --end 340000 --shard 0 --nshards 4 --out g2p_work/train
python scripts/g2p/dump_misaki.py --start 31000 --end 33500 --out g2p_work/eval
python scripts/g2p/train_tagger.py --out src/hibari/assets/tagger.json.gz
python scripts/g2p/espeak_words.py --words g2p_work/words_lex.txt --out g2p_work/espeak_lex.tsv
python scripts/g2p/make_oov_data.py && python scripts/g2p/train_oov.py --gpu 1 --out src/hibari/assets/oov
python scripts/g2p/eval_g2p.py g2p_work/eval.jsonl && python scripts/g2p/eval_e2e.py --n 50
# 7. package
pip install build && python -m build --wheel && python -m pytest tests
```

Phase 2 (planned): a native, no-Python runtime. The G2P data files are plain JSON (lexicon, tagger weights,
tokenizer special cases, normalization tables) and the OOV net is a small ONNX graph, so they can be loaded
from C, Rust or JavaScript as they are.

## License

Apache-2.0 (LICENSE). Kokoro-82M and misaki are Apache-2.0 (hexgrad); spaCy's tokenizer rules are MIT;
the vendored StyleTTS2 discriminators used in training are MIT. See NOTICE. espeak-ng (GPL) was used
only to label training words for the out-of-vocabulary net and is not included.
