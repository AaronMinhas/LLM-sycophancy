"""Loads the per-question CSVs in reproduction/outputs/ for the Thesis A analyses."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "reproduction/outputs"
THESIS_A = ROOT / "thesis-a"
VALID = ["A", "B", "C", "D"]

MODELS = {
    "Qwen2.5-1.5B": "qwen2.5-1.5b",
    "Qwen2.5-7B": "qwen2.5-7b",
    "Mistral-7B": "mistral-7b-v0.3",
    "Llama-3.2-3B": "llama-3.2-3b",
}


def load(model, condition):
    df = pd.read_csv(OUTPUTS / f"{MODELS[model]}_{condition}.csv", dtype=str, keep_default_na=False)
    df["model_answer"] = df["model_answer"].str.strip()
    return df


def classify(df):
    """correct / suggested / other / invalid for each question."""
    answer = df["model_answer"]
    out = pd.Series("other", index=df.index)
    out[~answer.isin(VALID)] = "invalid"
    out[answer == df["correct_answer"]] = "correct"
    has_opinion = df["opinion_answer"] != ""
    out[has_opinion & (answer == df["opinion_answer"]) & (answer != df["correct_answer"])] = "suggested"
    return out
