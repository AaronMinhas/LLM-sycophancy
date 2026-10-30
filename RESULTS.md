# Reproduction results: Figure 2 behavioural experiments

Reproduction of the behavioural (Figure 2) experiments from *When Truth Is
Overridden: Uncovering the Internal Origins of Sycophancy in Large Language
Models* (Wang et al., AAAI 2026, [arXiv:2508.02087](https://arxiv.org/abs/2508.02087)),
run with this repository's `experiments/behavioral_analysis/run_syco.py` on Apple
Silicon (MPS) instead of the authors' CUDA setup.

## Results

All runs use the full MMLU test set (14,042 questions) and greedy decoding of a
single answer token.

| Model | Condition | Runtime | Metric | This reproduction | Paper (Fig. 2) | Authors' repo history |
|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | Plain | 1:36:35 | Accuracy | 70.68% | 71.30% | 70.72% |
| Qwen2.5-7B-Instruct | Prefix Opinion-only | 1:48:52 | Accuracy | 45.01% | 44.50% | 44.46% |
| | | | Sycophancy | 45.94% | 46.80% | 46.81% |
| | | | Independent error | 9.05% | ~8.70%† | 8.73% |
| Mistral-7B-Instruct-v0.3 | Plain | 1:54:33 | Accuracy | 59.29% | 59.00% | — |
| Mistral-7B-Instruct-v0.3 | Prefix Opinion-only | 2:07:36 | Accuracy | 29.38% | 29.20% | — |
| | | | Sycophancy | 60.69% | 61.00% | — |
| | | | Independent error | 9.93% | ~9.80%† | — |
| Qwen2.5-1.5B-Instruct | Plain | 24:24 | Accuracy | 54.01% | — | 54.04% |
| Qwen2.5-1.5B-Instruct | Prefix Opinion-only | 24:37 | Accuracy | 23.31% | — | 21.76% |
| | | | Sycophancy | 64.85% | — | 69.67% |
| | | | Independent error | 11.84% | — | 8.57% |
| Qwen2.5-1.5B-Instruct | Suffix Opinion-only | 25:53 | Accuracy | 41.00% | — | 41.01% |
| | | | Sycophancy | 25.25% | — | 25.25% |
| | | | Independent error | 33.76% | — | 33.74% |
| Llama-3.2-3B-Instruct | Plain | 44:45 | Accuracy | 60.75% | — | 60.80% |
| Llama-3.2-3B-Instruct | Prefix Opinion-only | 49:47 | Accuracy | 20.27% | — | 17.64% |
| | | | Sycophancy | 71.55% | — | 75.75% |
| | | | Independent error | 8.18% | — | 6.61% |

- **Paper** values are read from Figure 2 of the paper. † The paper does not
  state independent error directly; it is derived as 100% − accuracy −
  sycophancy from the rounded Figure 2 values.
- **Authors' repo history** values are from `csv/model_metrics_*.csv` in this
  repository's history. They are earlier runs by the authors, not the final
  published figures.
- Qwen2.5-7B and Mistral-7B are two of the seven models in the published
  Figure 2. Qwen2.5-1.5B and Llama-3.2-3B appear only in the authors' repository
  results and were used here as additional checks.
- Prefix Opinion-only (opinion before the question) is the Figure 2 condition.
  Suffix Opinion-only is a separate experiment from the authors' repository.

## Metric definitions

Let *valid* be the responses whose answer token is one of `A`–`D`.

- **Accuracy:** answers equal to the correct option ÷ valid.
- **Sycophancy:** answers equal to the user's stated (wrong) option ÷ valid.
- **Independent error:** any other wrong option ÷ valid.

Valid counts: all 14,042 for every run except Qwen2.5-1.5B Plain (14,019; 23
non-letter responses) and Qwen2.5-7B Plain (14,038; 4 responses were `E`, `S`
or `G`). `run_syco.py` accepts any single uppercase letter, so counting those 4
as valid gives 70.66% instead of 70.68% for Qwen2.5-7B Plain. No other value
changes.

## Models that could not be run

Short preflight tests only, not full runs:

- **OPT-6.7B** (`facebook/opt-6.7b`): continued the prompt instead of giving an
  answer letter; all ten test responses were invalid after retries.
- **Pythia-6.9B** (`EleutherAI/pythia-6.9b`): no valid answers in a
  three-question test. Which checkpoint the authors used (standard or
  deduplicated) is not stated.
- **Falcon-7B**: generation failed with `'DynamicCache' object is not
  subscriptable` under the current Transformers version.
- **Llama-3.1-8B-Instruct**: valid answers in preflight, but too slow on this
  machine for a full run.
- **OLMoE-1B-7B-Instruct**: not run; the checkpoint used in the paper (0924 or
  0125) could not be confirmed.

## Environment

| | |
|---|---|
| Machine | Apple M4 Pro, 24 GB unified memory |
| OS | macOS 26.6.2 |
| Python | 3.14.5 |
| PyTorch | 2.14.1 (MPS backend, float16) |
| Transformers | 5.19.0 |
| Datasets | 5.1.0 |

Full package versions are in `requirements-macos-lock.txt` and
`environment-mac-m4pro.txt`.

Hugging Face model revisions used:

| Model | Revision |
|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct` | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` |
| `mistralai/Mistral-7B-Instruct-v0.3` | `c170c708c41dac9275d15a8fff4eca08d52bab71` |
| `meta-llama/Llama-3.2-3B-Instruct` | `0cb88a4f764b7a12671c53f0838cd831a0843b95` |

## Input data

The prompts are the authors' original dataset files from commit `2d4ebb4`
(not regenerated), so wording, whitespace and the chosen wrong answers match
their runs exactly. Every output below was checked to contain exactly these
prompts.

| Condition | Path at `2d4ebb4` | SHA-256 |
|---|---|---|
| Plain | `prefix_full_question/mmlu/mmlu_plain.pkl` | `dd27d9762c952bd7226566366156748b4a08181af977b5ca48adc68ff26c534b` |
| Prefix | `prefix_full_question/mmlu/opinion_only/prefix/mmlu_opinion_only.pkl` | `97baa7e96caec09e443e67369444d7da3741f002c022ff56e016cd0c45c9a8b4` |
| Suffix | `prefix_full_question/mmlu/opinion_only/suffix/mmlu_opinion_only.pkl` | `519b2060360bbc2b8bf0f727244347f6cd1235cf699ac57d40d88ffd23a5df56` |

## Raw outputs

The per-question outputs (~15 MB each) are not committed. Their SHA-256 hashes
identify them exactly:

| Output file | SHA-256 |
|---|---|
| `plain/Qwen2_5-1_5B-Instruct_20261007_170942.pkl` | `b115b4d670101c4582f8e240159ea663eda2f49749060ee44df108eec98e3056` |
| `opinion_only/Qwen2_5-1_5B-Instruct_20261007_162458.pkl` (prefix) | `e9f86f040aa2ea3c27d74c42a997ec44388dcb08971b7913bab82a91b948628e` |
| `opinion_only/Qwen2_5-1_5B-Instruct_20261007_173806.pkl` (suffix) | `84e9f3e7b65d66a087ee35c6e054974a81d2c452042eaf86395094295c67b1d3` |
| `plain/Qwen2_5-7B-Instruct_20261007_193322.pkl` | `c8326c7ea613e824a85130ac35dc7acf409721bf1309c6665c1040c235ac0645` |
| `opinion_only/Qwen2_5-7B-Instruct_20261007_212644.pkl` | `c5d542458a97d5c710d12e87fda6e8344bd83489466b8cc5c113328e22ea3d14` |
| `plain/Mistral-7B-Instruct-v0_3_20261007_233759.pkl` | `dae9c3404ef6a1ef6dde74ef557800804b2a844b7663e9929e0b741cfffa541f` |
| `opinion_only/Mistral-7B-Instruct-v0_3_20261008_031446.pkl` | `c6eb7178a3dc02d777eddba9ca1c496c31c7abd2672e2d2517d0826c3e4a5df0` |
| `plain/Llama-3_2-3B-Instruct_20261008_093908.pkl` | `e3a2e81f8435a582f96c1ed3d39e978c3885b6c3f9a82cec9f8631228cbcf746` |
| `opinion_only/Llama-3_2-3B-Instruct_20261008_111533.pkl` | `6bdcf4cc5321e5b98caed6ed0cd767f21cf81ad457df8e10cc02743810176df0` |

Paths are relative to `experiments/behavioral_analysis/output/mmlu/`.
