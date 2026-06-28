# Repository Guidelines

## Project Structure & Module Organization

This repository implements a neural Old English reconstruction pipeline from modern Germanic cognates. Core Python scripts live at the repository root:

- `join_sound_and_morph.py`: builds `germanic_joint_ipa_dataset.csv` from `germanic_parallel_dataset.csv`.
- `train.py`: fine-tunes `google/byt5-small` and writes the model under `byt5_grapheme_phoneme_reconstructor/`.
- `test.py`: interactive inference against the saved model.
- `measurement_joint.py`: batch evaluation on `actual_validation_set.csv`.

Datasets and generated CSVs are currently root-level files. Evaluation result artifacts may also appear in `eval/`. `CLAUDE.md` contains useful pipeline notes and should be kept consistent with behavior changes.

## Build, Test, and Development Commands

Install Python dependencies in a virtual environment before running scripts. There is no `requirements.txt`; expected packages include `pandas`, `torch`, `transformers`, `datasets`, `phonemizer`, and `tqdm`.

```bash
sudo apt-get install espeak-ng
python join_sound_and_morph.py
python train.py
python test.py
python measurement_joint.py
```

`espeak-ng` is required by `phonemizer`. Run the scripts in pipeline order when regenerating data or models. `train.py` may download Hugging Face model files and can be slow without GPU acceleration.

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation. Keep constants in uppercase, as in `MODEL_PATH` and `JOINT_CSV`. Prefer small helper functions for reusable formatting, transcription, and metric logic. Existing code includes Chinese comments and user-facing messages; preserve them when editing nearby logic unless intentionally localizing a section.

## Testing Guidelines

There is no formal test framework yet. Validate changes by running the narrowest relevant pipeline stage. For data preparation changes, run `python join_sound_and_morph.py` and inspect the generated CSV columns. For inference changes, run `python test.py` after a model exists. For evaluation logic, run `python measurement_joint.py` and check the summary metrics plus the generated results CSV.

## Commit & Pull Request Guidelines

Recent commit history uses short imperative messages and occasional Conventional Commit prefixes, for example `fix: model doesn't exit` and `feat: updated measurement`. Keep commits focused and describe behavior, not implementation trivia.

Pull requests should include a concise description, affected pipeline stage, commands run, and any changed metrics or generated artifacts. Include screenshots only for terminal/UI behavior where they clarify the result.

## Security & Configuration Tips

Do not commit large model checkpoints unless explicitly requested. Avoid hard-coding local absolute paths; scripts currently assume execution from the repository root.
