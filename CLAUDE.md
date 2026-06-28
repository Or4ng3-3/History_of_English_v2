# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Neural reconstruction of Old English (Anglo-Saxon) word forms from modern Germanic cognates using a ByT5 sequence-to-sequence model. This implements the approach from "Ab Antiquo: Neural Proto-Language Reconstruction" — a joint grapheme-phoneme model that takes modern Germanic words with their IPA pronunciations and predicts the ancestral Old English form (both spelling and IPA).

## Pipeline

The project runs in three stages — run them in order:

```
join_sound_and_morph.py  →  train.py  →  test.py (interactive) / measurement_joint.py (evaluation)
```

### 1. Data Preparation

```bash
python join_sound_and_morph.py
```

Reads `germanic_parallel_dataset.csv` (~2,585 rows; columns: Proto_Germanic_Ancestor, Meaning, Target_Old_English, Modern_English, Modern_German, Modern_Dutch, Modern_Swedish), generates IPA transcriptions via `espeak-ng` for modern words and via hand-written historical sound-change rules (`oe_to_ipa()`) for Old English targets. Writes `germanic_joint_ipa_dataset.csv`.

The `oe_to_ipa()` function encodes specific phonological knowledge about Old English: macron vowels → long IPA, palatalization of c/g before front vowels, intervocalic fricative voicing, sc → /ʃ/, cg → /dʒ/, þ/ð → /θ/.

### 2. Training

```bash
python train.py
```

Fine-tunes `google/byt5-small` (a byte-level T5, chosen because it handles IPA characters natively without tokenization issues). Input format: `"English: free [/fɹiː/] | German: frei [/fraɪ/] | Dutch: vrij [/vrɛɪ/] | Swedish: fri [/friː/]"` → target: `"frēo [/freːo/]"`. Trains for 12 epochs, saves only the best checkpoint to `./byt5_grapheme_phoneme_reconstructor/best_joint_model`.

Key training constants:
- `save_total_limit=1` (disk-constrained; only keeps best checkpoint)
- `fp16=False` (FP32 for stability on T4 GPUs)
- 15% validation split from the joint CSV

### 3. Inference / Evaluation

```bash
# Interactive mode — type in cognates and get predictions
python test.py

# Batch evaluation on the dev set — produces edit-distance metrics
python measurement_joint.py
```

`test.py` uses `phonemizer` at inference time to transcribe user-provided words into IPA before feeding them to the model. `measurement_joint.py` evaluates on `actual_validation_set.csv` (388 samples) and outputs per-sample results to `joint_dev_evaluation_results.csv` plus summary statistics: accuracy at edit distance ≤ 0 through ≤ 4 (matching the paper's Table 1 format).

## System Dependency

`phonemizer` requires **espeak-ng** installed on the system:

```bash
sudo apt-get install espeak-ng
```

## Dependencies

No `requirements.txt` exists. Required Python packages:
- `transformers` (ByT5 model + training)
- `datasets` (Hugging Face dataset splitting)
- `torch` (PyTorch backend)
- `pandas` (CSV I/O)
- `phonemizer` (espeak-ng wrapper for IPA transcription)
- `tqdm` (progress bars, evaluation only)

## Model Input/Output Format

The model uses joint grapheme-phoneme representations throughout:

- **Input:** `"<Lang>: <word> [/<IPA>/]"` segments joined by ` | `
- **Output:** `"<Old English word> [/<IPA>/]"`

Any language with value `-` in the dataset is omitted from the input string at format time. At inference, the user can skip a language by pressing Enter.
