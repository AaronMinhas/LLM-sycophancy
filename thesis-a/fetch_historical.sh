#!/bin/bash
# Downloads the authors' earlier output files used by analysis/historical.py and checks their hashes.
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p historical
cd historical

RAW=https://raw.githubusercontent.com/kaustpradalab/LLM-sycophancy
LFS=https://media.githubusercontent.com/media/kaustpradalab/LLM-sycophancy

curl -sSfL -o Qwen2_5-1_5B-Instruct_20250418_123816.pkl \
  $RAW/a89706fe712be16c7dcb9120d949fdfee0d14f5b/output/mmlu/plain/Qwen2_5-1_5B-Instruct_20250418_123816.pkl
curl -sSfL -o Qwen2_5-1_5B-Instruct_20250420_173155.pkl \
  $LFS/8fb835d54cf8515d60e6d7fb936d8efa8e80d500/outputsuffix/mmlu/opinion_only/Qwen2_5-1_5B-Instruct_20250420_173155.pkl
curl -sSfL -o Qwen2_5-1_5B-Instruct_20250426_232046.pkl \
  $RAW/c1b88b86dfb49d6d13c23cd3c0bb8fadf4464fe9/outputprefix/mmlu/opinion_only/Qwen2_5-1_5B-Instruct_20250426_232046.pkl
curl -sSfL -o mmlu_plain_out_Llama-3_2-3B-Instruct_20250409_000701.pkl \
  $RAW/26ede5cccfbfebbb79efa09de55c9b2cc1702c94/output/mmlu/plain/mmlu_plain_out_Llama-3_2-3B-Instruct_20250409_000701.pkl
curl -sSfL -o Llama-3_2-3B-Instruct_20250427_092643.pkl \
  $RAW/c1b88b86dfb49d6d13c23cd3c0bb8fadf4464fe9/outputprefix/mmlu/opinion_only/Llama-3_2-3B-Instruct_20250427_092643.pkl

shasum -a 256 -c ../data/historical.sha256
