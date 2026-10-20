# Languages beyond English: feasibility study (2026-10-20)

Question: can hibari speak Kokoro-82M's other languages (Spanish, French, Hindi, Italian, Brazilian
Portuguese, Japanese, Mandarin) while staying as lean and fast as the English models and being reliable?
The bar, per language: the teacher itself must be about as intelligible as it is in English; a
permissively licensed text-to-phoneme front end of at most ~2 MB must reproduce the phonemes the teacher
was trained on at <= 1% phoneme error rate; the acoustic model keeps the same size and speed; every
component's license is permissive. Nothing acoustic was trained for other languages in this study.

## 1. Is the teacher reliable in the language?

30 corpus sentences per language (no digits, 5-30 words; Japanese and Mandarin 8-60 characters), the
best-graded Kokoro voice for it, Kokoro's own front end (espeak-ng for e/f/h/i/p, misaki ja/zh), transcribed
by Whisper large-v3 with the language given (`scripts/lang/teacher_quality.py`). English is scored the
same way as the reference point (the 2.8% WER elsewhere in the docs is Whisper base). UTMOS is an
English-trained MOS predictor, shown for scale only: it rates the Spanish, French and Portuguese voices low
although their transcripts are nearly right.

| language | voice (Kokoro grade) | WER | CER | UTMOS | audio |
|---|---|---|---|---|---|
| English (US) | af_heart (A) | 0.6% | 0.2% | 4.52 | 220 s |
| Spanish | ef_dora (ungraded) | 3.4% | 1.0% | 3.62 | 203 s |
| French | ff_siwis (B-) | 5.8% | 2.3% | 3.66 | 214 s |
| Hindi | hf_alpha (C) | 12.4% | 5.5% | 4.29 | 231 s |
| Italian | if_sara (C) | 6.1% | 1.5% | 4.05 | 229 s |
| Portuguese (BR) | pf_dora (ungraded) | 2.3% | 0.4% | 3.68 | 239 s |
| Japanese | jf_alpha (C+) | n/a | 7.9% (5.5% on readings) | 3.81 | 237 s |
| Mandarin | zf_xiaoxiao (D) | n/a | 4.4% (simplified-script sentences) | 3.52 | 209 s |

Notes. Hindi: Whisper's Hindi spelling varies (anusvara vs chandrabindu, है vs हैं, nukta), which inflates
the WER, but there are also real misreadings (उड़ान heard as उर्दान); the voice is grade C. Japanese: raw
CER counts kanji that Whisper chose differently for the same sound (痕/根, 対馬/津島); converting both
sides to readings with pyopenjtalk gives 5.5%, still well above English. Mandarin: the corpus mixes
traditional and simplified script; Kokoro's pipeline expects simplified, and on the mixed set the CER was
16.8%, so the table uses simplified-only sentences, where the remaining errors are real misreadings
(晉武帝 heard as 晋努帝, 州郡 as 周郡) of a grade-D voice. The non-English voices rush and swallow
syllables more often than af_heart does; every sentence of these runs is in `runs/lang/teacher*/`.

Verdict on the teacher alone: Portuguese and Spanish are close to English; Italian and French are usable
but noticeably worse; Hindi, Japanese and Mandarin are not reliable enough to distill from.

## 2. Text to phonemes: what Kokoro uses, and the lean permissive replacement

What Kokoro's pipeline (kokoro 0.9.4 / misaki 0.9.4) uses, measured in this environment:

| languages | Kokoro's G2P | license | size on disk |
|---|---|---|---|
| es fr hi it pt-br | espeak-ng through phonemizer-fork (`misaki.espeak.EspeakG2P`) | GPL-3.0 (both) | espeak-ng data 18.8 MB (all languages) |
| ja | `misaki.ja`: pyopenjtalk (Open JTalk + HTS engine, dictionary downloaded on first use), misaki's `ja_words.txt`, cutlet, fugashi + unidic-lite | MIT / BSD-3 (Open JTalk, NAIST-jdic); unidic-lite dictionary BSD/LGPL/GPL triple license | pyopenjtalk 2.8 MB + dictionary 107 MB (22.6 MB download) + unidic-lite 260 MB |
| zh | `misaki.zh`: jieba (segmentation), pypinyin + pypinyin-dict (readings), cn2an (numbers), misaki's `zh_normalization` (from PaddleSpeech), tone sandhi rules | MIT (jieba, pypinyin, pypinyin-dict, cn2an), Apache-2.0 (misaki, PaddleSpeech) | jieba 43 MB, pypinyin 3.7 MB, pypinyin-dict 113 MB, misaki zh 0.2 MB |

espeak-ng cannot ship (GPL), and the ja/zh stacks are 100 to 400 MB of dictionaries. The replacement has
to reproduce what the teacher was trained on, which is espeak's output including its conventions and
quirks (Spanish: Castilian θ, β/ð/ɣ allophones, secondary stress on every other syllable before the
primary, numbers joined with "i"; Italian: ɛ/ɔ vs e/o, geminates as ː, final unstressed i as ɪ).
Prototypes in `scripts/g2p_lang/` (rules written from the orthography and tuned against espeak's output
on corpus sentences; no espeak code or data is used), measured with `eval_lang.py` against espeak-ng
on held-out corpus sentences including ones with digits:

| language | prototype | size | sentences | PER | exact match | sentences with digits: PER / exact |
|---|---|---|---|---|---|---|
| Spanish | `es.py`: letter rules, stress rules, allophones across words, number reader, 60-word function-word table | 12.5 KB, no lexicon | 1000 | 1.13% | 68.9% | 1.78% / 52% |
| Spanish | same | | 500 | 1.04% | 71.4% | 1.73% / 55% |
| Italian | `it.py`: letter rules, default penultimate stress, geminates, number reader, 80 function words | 9.4 KB, no lexicon | 500 | 7.81% | 0.4% | 7.82% / 0% |

Spanish: the residual 1% is mostly acronyms (espeak spells ADN, ONU, CFC letter by letter), foreign names
(Khan, Twitter, Ludwig), a few function words whose stress depends on phrase position, and number formats
with separators (espeak reads "2.000" as "dos . cero cero cero"). An acronym rule and a ~200-entry
exceptions list would bring it under 1%; that is a day's work, not a lexicon. Italian: the remaining 8%
is lexical: stress position (sˈɛkolo, fˈizika, nˈumero fall to the penultimate rule), open vs closed e/o
and voiced vs voiceless s/z, which espeak takes from its dictionary. A rules-only Italian cannot pass the
bar; it needs a pronunciation lexicon (a 50 k-word list is about 1 MB gzipped) or a small net trained on
espeak's readings like hibari's English OOV net (0.9 MB int8, 74% word accuracy on unseen words, which
would not reach 1% PER on its own either).

The others, from reading espeak's conventions (no prototype built):
- Portuguese (BR): espeak's output is heavily reduced (unstressed a -> æ, final o -> ʊ, final -te/-de ->
  ʧy/ʤy, l -> w in codas, nasal vowels with a combining tilde, an epenthetic ə in clusters: "portuguesa" ->
  pˌoɾətuɡˈezæ) and open/closed e/o are lexical. Rules alone would land around Italian's 5-8%; a lexicon
  for vowel quality plus two to three days of rule work could reach 1-2%.
- French: silent letters, liaison and many exceptions; the permissive route is a lexicon (ipa-dict fr_FR,
  MIT, about 1.5 MB gzipped) with rules for the rest, then matching espeak's conventions (nasal vowels as
  combining tildes in the Kokoro alphabet). Two to four days.
- Hindi: Devanagari is nearly phonemic except schwa deletion, which needs a lexicon or a small model
  (permissive data is scarce; WikiPron is CC BY-SA). Three to five days, and the teacher is the weaker
  problem.
- Japanese: kanji readings need segmentation and a dictionary (tens of MB even trimmed). Fails the 2 MB bar
  unless input is restricted to kana, which users do not type. Not lean.
- Mandarin: a character-to-pinyin table (~7 k characters, ~100 KB) plus a polyphone phrase list (~300 KB)
  plus tone sandhi rules and misaki's pinyin-to-IPA mapping fits in 2 MB; accuracy on polyphones without
  jieba-style segmentation would be a few percent of characters, and the voices are all grade D.

## 3. Text corpora with permissive licenses

[agentlans/high-quality-multilingual-sentences](https://huggingface.co/datasets/agentlans/high-quality-multilingual-sentences)
(CC-BY-4.0; sentences from LinguaNova filtered by a quality classifier): es 40,468, fr 44,277, hi 31,932,
it 19,395, ja 24,708, pt 46,619, zh 41,493 sentences (`data/lang/`). These are enough for a fine-tune
(fenrir and emma used 15 k each) but fewer than heart's 30 k; for more, Tatoeba (CC BY 2.0 FR, short
sentences, every language above) and FineWeb-2 (ODC-By 1.0, web text, every language above). The English
corpus is ODC-By as well.

## 4. Verdicts

| language | teacher WER / CER | G2P approach, measured PER | G2P data | licenses | verdict |
|---|---|---|---|---|---|
| Spanish | 3.4% / 1.0% | rules, 1.1% (acronyms and names left) | 12.5 KB | all permissive (CC-BY text) | **go** |
| Portuguese (BR) | 2.3% / 0.4% | rules + small lexicon; not built, expected 1-2% after 2-3 days | ~0.5 MB | permissive | not yet |
| Italian | 6.1% / 1.5% | rules measured 7.8%; needs a lexicon | ~1 MB | lexicon source to be chosen | not yet |
| French | 5.8% / 2.3% | lexicon (ipa-dict, MIT) + rules; not built | ~1.5 MB | permissive | not yet |
| Hindi | 12.4% / 5.5% | rules + schwa model; not built | < 1 MB | unclear (lexicon) | no (teacher) |
| Japanese | n/a / 5.5% (readings) | dictionary needed, > 20 MB | fails 2 MB | permissive but huge | no |
| Mandarin | n/a / 4.4% | char + phrase tables + sandhi; not built | ~0.5 MB | permissive | no (teacher grade D) |

Model size and speed are the same for every language: the decoder and ProsodyNet do not change, and
Kokoro's phoneme alphabet (hibari's `vocab.json`) already contains every symbol espeak uses for these
languages, so a language is one more 7.9 MB voice file plus its front end.

## 5. Recommendation: Spanish first

What it takes:
1. Front end (1-2 days, CPU): finish `es.py` (acronym rule, exceptions, "1.234,56" style numbers, dates,
   units, currency) and port it into `src/hibari/g2p/es/` with its own normalizer; keep measuring against
   espeak on 2,000 sentences; target <= 0.7% PER. About 20 KB added to the wheel.
2. Teacher data (1 hour, GPU 0): `dump_teacher.py --voice ef_dora --lang e` on 20 k Spanish sentences
   (the dump script already takes a language code; the Spanish pipeline is espeak-based and works here).
   Consider em_alex (male) as a second voice later.
3. Acoustic model (2-6 hours, GPU 0): fine-tune the decoder and ProsodyNet from heart's weights as fenrir
   and emma were (the vocabulary is shared), 30 k steps; if the cross-language fine-tune plateaus below
   the teacher-forced UTMOS seen for English, train the three-stage schedule from scratch (6 hours).
4. Evaluation: Whisper large-v3 Spanish WER and UTMOS, teacher vs student on 50 unseen sentences, as for
   the English voices; bar: WER within 0.5 points of the teacher's 3.4%.
5. Package: a `language` field in the voice registry picks the front end; tests with Spanish sentences
   and numbers; no new dependencies.

Expected: one more voice in the registry, Spanish text in, 24 kHz audio out, same 7.9 MB and ~130x
realtime, about 4 days of work with under a day of GPU time. Portuguese would follow the same path once
its front end exists; Italian and French after a lexicon decision; Hindi, Japanese and Mandarin wait for a
better teacher.
