"""Export the raw run_syco.py outputs used in RESULTS.md to small per-question CSVs.

Run from the repository root after the runs have finished:

    python reproduction/export_outputs.py
"""

import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "experiments/behavioral_analysis/output/mmlu"
EXPORT_DIR = ROOT / "reproduction/outputs"

# (export name, model, condition, raw output file relative to OUTPUT_DIR)
RUNS = [
    ("qwen2.5-1.5b_plain", "Qwen2.5-1.5B-Instruct", "plain",
     "plain/Qwen2_5-1_5B-Instruct_20261007_170942.pkl"),
    ("qwen2.5-1.5b_prefix", "Qwen2.5-1.5B-Instruct", "prefix",
     "opinion_only/Qwen2_5-1_5B-Instruct_20261007_162458.pkl"),
    ("qwen2.5-1.5b_suffix", "Qwen2.5-1.5B-Instruct", "suffix",
     "opinion_only/Qwen2_5-1_5B-Instruct_20261007_173806.pkl"),
    ("qwen2.5-7b_plain", "Qwen2.5-7B-Instruct", "plain",
     "plain/Qwen2_5-7B-Instruct_20261007_193322.pkl"),
    ("qwen2.5-7b_prefix", "Qwen2.5-7B-Instruct", "prefix",
     "opinion_only/Qwen2_5-7B-Instruct_20261007_212644.pkl"),
    ("mistral-7b-v0.3_plain", "Mistral-7B-Instruct-v0.3", "plain",
     "plain/Mistral-7B-Instruct-v0_3_20261007_233759.pkl"),
    ("mistral-7b-v0.3_prefix", "Mistral-7B-Instruct-v0.3", "prefix",
     "opinion_only/Mistral-7B-Instruct-v0_3_20261008_031446.pkl"),
    ("llama-3.2-3b_plain", "Llama-3.2-3B-Instruct", "plain",
     "plain/Llama-3_2-3B-Instruct_20261008_093908.pkl"),
    ("llama-3.2-3b_prefix", "Llama-3.2-3B-Instruct", "prefix",
     "opinion_only/Llama-3_2-3B-Instruct_20261008_111533.pkl"),
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    for name, model, condition, raw in RUNS:
        raw_path = OUTPUT_DIR / raw
        df = pd.read_pickle(raw_path)

        export = pd.DataFrame({
            "question_index": range(len(df)),
            "correct_answer": df["correct_answer_index"],
            "opinion_answer": df.get("chosen_wrong_answer_index", ""),
            "model_answer": df["model_answer"].fillna(""),
        })
        export.to_csv(EXPORT_DIR / f"{name}.csv", index=False)

        manifest.append({
            "file": f"{name}.csv",
            "model": model,
            "condition": condition,
            "source_output": raw,
            "source_sha256": sha256(raw_path),
        })
        print(f"Wrote {name}.csv ({len(export)} rows)")

    pd.DataFrame(manifest).to_csv(EXPORT_DIR / "manifest.csv", index=False)


if __name__ == "__main__":
    main()
