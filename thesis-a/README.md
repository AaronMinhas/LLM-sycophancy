# Thesis A analysis

Scripts and results for the question-level analysis in my Thesis A report (Macquarie
University, COMP4092): Section 4.2.3 and Appendix A. They read the per-question CSVs in
[`../reproduction/outputs/`](../reproduction/outputs/), so they don't need the raw pickle
files or a GPU, except for the MPS checks.

Requirements: Python 3 with `pandas` and `numpy`. The MPS checks also need the repo's
`.venv` (see [Reproducing](../README.md#reproducing)).

Run everything from the repository root.

| Report section | Command | Result file |
|---|---|---|
| §4.2.3 answer transitions (Table 9) and shared-vulnerability test | `python3 thesis-a/analysis/transitions.py` | `results/transitions.json` |
| Appendix A.1 agreement by suggested letter (Table 14) | `python3 thesis-a/analysis/letters.py` | `results/letters.json` |
| Appendix A.2 hook and precision checks on MPS (Table 15) | `.venv/bin/python thesis-a/analysis/mps_checks.py` | `results/mps_checks.json` |
| Appendix A.3 comparison with the authors' earlier files (Table 16) | `bash thesis-a/fetch_historical.sh` then `python3 thesis-a/analysis/historical.py` | `results/historical.json` |
| Appendix A.4 stratified shared-vulnerability test (Table 17) | `python3 thesis-a/analysis/stratified.py` | `results/stratified.json` |

The JSON files in `results/` are the outputs the report uses. Rerunning a script overwrites
its file, so `git diff thesis-a/results/` shows whether anything changed.

## Notes

- **Seeds.** The permutation tests use `numpy.random.default_rng(0)` with 2,000 shuffles.
  The MPS checks sample questions with seed 2026.
- **MPS checks.** Ran on an Apple M4 Pro (24 GB) on 8 November 2026 with
  Qwen/Qwen2.5-1.5B-Instruct at revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`,
  float16, greedy decoding and one new token, using the environment in
  `requirements-macos-lock.txt`. They need the raw Prefix input and output pickles
  (hashes in [RESULTS.md](../RESULTS.md)). These are the only results here that can't be
  replayed from the CSVs.
- **Authors' earlier files.** `fetch_historical.sh` downloads five output files from the history of
  [kaustpradalab/LLM-sycophancy](https://github.com/kaustpradalab/LLM-sycophancy) into
  `thesis-a/historical/` and checks them against `data/historical.sha256`. They aren't
  committed here. `historical.py` opens them with a restricted unpickler.
- **Subject labels.** `data/mmlu_subjects.csv` gives the MMLU subject for each
  `question_index`. I took them from the subject named in the authors' academic-prompt file
  (`prefix_full_question/mmlu/original_levelled/mmlu_academic_opinion_advanced.pkl` at commit
  `2d4ebb4`), whose questions are in the same order as the files used here (57 subjects).
