---
license: apache-2.0
language:
- en
library_name: onnx
pipeline_tag: text-to-speech
base_model: hexgrad/Kokoro-82M
tags:
- text-to-speech
- tts
- onnx
- distillation
- kokoro
- cpu
---

# hibari

Tiny English text-to-speech. 7.9 MB int8 ONNX models, 6.15 M parameters each, about 130x realtime on a
single CPU core, 24 kHz audio. Each voice is distilled from one [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
voice: the default (af_heart) scores UTMOS 4.43 against the teacher's 4.52 while running 65x faster. Five
voices: heart (American female, inside the pip package), fenrir (American male), emma (British female),
fable (British male) and kore (American female), the last four in `voices/` here and downloaded by the
package on first use. Comes with its own
text-to-phoneme front end (no spaCy, no espeak-ng), so plain text goes in and audio comes out.

hibari is Japanese for skylark. Code, Python package and training pipeline:
[github.com/MMWilliams/hibari](https://github.com/MMWilliams/hibari).

## Samples

hello: <audio controls src="https://huggingface.co/maureesewilliams/hibari/resolve/main/samples/hello.wav"></audio>

numbers, dates, money: <audio controls src="https://huggingface.co/maureesewilliams/hibari/resolve/main/samples/numbers.wav"></audio>

question: <audio controls src="https://huggingface.co/maureesewilliams/hibari/resolve/main/samples/question.wav"></audio>

story: <audio controls src="https://huggingface.co/maureesewilliams/hibari/resolve/main/samples/story.wav"></audio>

The other voices introducing themselves (same sentence, synthesized by the package):

fenrir (American male): <audio controls src="https://huggingface.co/maureesewilliams/hibari/resolve/main/samples/fenrir.wav"></audio>

emma (British female): <audio controls src="https://huggingface.co/maureesewilliams/hibari/resolve/main/samples/emma.wav"></audio>

fable (British male): <audio controls src="https://huggingface.co/maureesewilliams/hibari/resolve/main/samples/fable.wav"></audio>

kore (American female): <audio controls src="https://huggingface.co/maureesewilliams/hibari/resolve/main/samples/kore.wav"></audio>

## Use

```
pip install hibari-tts
hibari "Hello there, this is hibari." -o hello.wav
```

```python
from hibari import Hibari
tts = Hibari()                         # voice "heart"; Hibari(voice="emma") or "fenrir" download once
audio = tts("Hello there.")            # numpy float32, 24 kHz
tts.save("Hello there.", "hello.wav")
for chunk in tts.stream(long_text):    # one array per sentence, for low latency
    ...
```

```
hibari "Hello there." --voice fenrir -o fenrir.wav
hibari --list-voices
```

The PyPI name is hibari-tts; the import and the command are `hibari`. The package depends only on numpy and onnxruntime and runs on Windows, macOS and Linux (x64 and arm64).
Everything it needs is inside the wheel; this repository holds the same files for people who want to port
or study the model.

## Files

| path | what |
|---|---|
| onnx/hibari.int8.onnx | shipping model of the heart voice, int8 MatMul weights (7.9 MB) |
| onnx/hibari.fp32.onnx | same model in fp32 (24.8 MB) |
| voices/am_fenrir.int8.onnx, voices/bf_emma.int8.onnx, voices/bm_fable.int8.onnx, voices/af_kore.int8.onnx | the fenrir, emma, fable and kore voices, same interface (7.9 MB each; the package downloads these, sha256 in its registry) |
| voices/am_fenrir_v2.int8.onnx | 0.2.1's fenrir: the same model with per-channel int8 weight scales (8.0 MB), UTMOS 4.39 instead of 4.35 on the 50 unseen sentences; the 0.2.0 file stays for older registries |
| voices/*.fp32.onnx | the same in fp32 |
| g2p/ | text-to-phoneme data: misaki lexicons, tagger weights, tokenizer special cases, normalization tables, out-of-vocabulary net (oov.onnx), phoneme vocabulary |
| checkpoints/decoder_v3_40k.safetensors | PyTorch weights of the decoder (TinyKokoro, kwargs in the file metadata) |
| checkpoints/prosody_v1.safetensors | PyTorch weights of ProsodyNet (durations, pitch, energy) |
| samples/ | the audio above |

ONNX interface: input `ids` [1, L] int64 (phoneme ids from g2p/vocab.json, id 0 at both ends) and `speed`
[1] float32; outputs `lh, theta, ln, lr, phi` [1, 601, F] and `f0` [1, F], one frame per 12.5 ms. A short
numpy routine (pulse train from f0, one inverse STFT) turns them into audio; the formula and reference code
are in the GitHub README and `src/hibari/dsp.py`.

To load a checkpoint in the research code (`training/tiny_model.py` on GitHub):

```python
import json
from safetensors import safe_open
from safetensors.torch import load_file
from tiny_model import TinyKokoro
p = "checkpoints/decoder_v3_40k.safetensors"
kw = json.loads(safe_open(p, "pt").metadata()["kw"])
m = TinyKokoro(**kw); m.load_state_dict(load_file(p))
```

## Model

| part | what | parameters |
|---|---|---|
| ProsodyNet | 6 shared attention layers (c 192) over phonemes; durations; 80 Hz F0 and energy from 6 dilated ConvNeXt blocks | 1.68 M |
| Decoder text encoder | ConvNeXt front + 4 shared attention layers (c 128) | 0.29 M |
| Decoder | 8 ConvNeXt blocks at 80 Hz (c 256, ff 768) on text features + F0 + energy | 3.41 M |
| Output head | linear to 5 x 601 spectral parts per frame | 0.77 M |

The network stops at a parametric spectrum per frame: a harmonic branch (magnitude and phase rotation
applied to the STFT of a band-limited pulse train built from F0), a noise branch and a free-phase branch.
The host turns it into audio with one inverse STFT, which keeps per-sample work out of the network.

## Evaluation

50 sentences never synthesized in training; UTMOS (utmos22_strong, higher is better), word error rate of
Whisper base transcripts; speed on one onnxruntime thread, Ryzen 9 9900X.

| model | params | file MB | x realtime 1 thread | latency short / long | peak RAM MB | UTMOS | WER |
|---|---|---|---|---|---|---|---|
| hibari int8 | 6.15 M | 7.94 | 128 | 12 / 71 ms | 158 | 4.43 | 2.7% |
| hibari fp32 | 6.15 M | 24.75 | 104 | 15 / 76 ms | 171 | 4.43 | 2.7% |
| Paradee-8M int8 | 9.06 M | 9.04 | 18.3 | 87 / 602 ms | 176 | 4.41 | 2.8% |
| Kokoro-7M-Distill | 7.48 M | 30.1 | 17.4 | 108 / 357 ms | 889 | | |
| Kokoro-82M (teacher) | 81.8 M | 327 | 1.9 | 739 / 3880 ms | 1527 | 4.52 | 2.8% |

Per voice (same protocol, 50 unseen sentences each, student fed the teacher's phonemes, one thread):

<!-- voices-table -->
| voice | Kokoro voice | teacher UTMOS / WER | hibari int8 UTMOS / WER | UTMOS gap | x realtime (1 thread) | file MB |
|---|---|---|---|---|---|---|
| heart | af_heart (American female) | 4.52 / 2.8% | 4.43 / 2.7% | 0.09 | 128 | 7.94 |
| emma | bf_emma (British female) | 4.40 / 2.9% | 4.16 / 2.6% | 0.24 | 139 | 7.94 |
| fenrir | am_fenrir (American male) | 4.48 / 2.6% | 4.39 / 2.4% | 0.09 | 171 | 8.03 |
| fable | bm_fable (British male) | 4.33 / 2.6% | 4.17 / 2.6% | 0.16 | 167 | 7.94 |
| kore | af_kore (American female) | 4.52 / 2.4% | 4.45 / 2.5% | 0.06 | 170 | 7.94 |

- emma: snapshot 40k picked by UTMOS with the voice's ProsodyNet on 50 held-out training rows (teacher 4.394; 20k 4.107, 30k 4.125, 40k 4.174, 50k 4.174, 60k 4.173).
- fenrir: snapshot 30k picked by UTMOS with the voice's ProsodyNet on 50 held-out training rows (teacher 4.496; 10k 4.382, 15k 4.400, 20k 4.397, 25k 4.403, 30k 4.414).
- fable: snapshot 25k picked by UTMOS with the voice's ProsodyNet on 50 held-out training rows (teacher 4.333; 10k 4.143, 15k 4.167, 20k 4.151, 25k 4.176, 30k 4.161).
- kore: snapshot 25k picked by UTMOS with the voice's ProsodyNet on 50 held-out training rows (teacher 4.521; 10k 4.454, 15k 4.465, 20k 4.466, 25k 4.471, 30k 4.469).
<!-- /voices-table -->

Text to phonemes, against the reference pipeline (misaki 0.9.4 + spaCy + espeak-ng) on 2,500 held-out
sentences: 0.50% phoneme error rate, 88% of sentences identical (95% for sentences with no out-of-vocabulary
word). Feeding the same model from hibari's G2P versus the reference G2P on 50 unseen sentences: UTMOS
4.431 vs 4.431, WER 2.6% vs 2.7%.

## Intended use and limitations

For reading English text aloud on devices without a GPU, where size and latency matter, and for research on
distillation and small vocoders. Not for voice cloning or impersonation, or for use where listeners aren't
told the speech is synthetic.

- Five fixed voices (American female x2 and male, British female and male), English only; no speaker or
  style input. Speed is adjustable.
- Trained on declarative web prose: questions and exclamations have a narrower pitch range than the teacher.
- Sentences are synthesized one at a time, with no cross-sentence prosody.
- Unknown words (about 1 in 5 sentences has one) are read by a small network that gets about 72% of unseen
  names exactly right; foreign names can come out anglicized.
- Numbers, dates and times follow US conventions.

## Training data

All training audio was synthesized by Kokoro-82M from 30 k sentences (66 hours) of
[agentlans/high-quality-english-sentences](https://huggingface.co/datasets/agentlans/high-quality-english-sentences).
No human recordings were used.

## License

Apache-2.0, like Kokoro-82M and misaki. See NOTICE for attributions (Kokoro-82M, kokoro and misaki by
hexgrad; spaCy's tokenizer rules, MIT; StyleTTS2, MIT). espeak-ng (GPL) only labeled training words for the
out-of-vocabulary net and is not included.
