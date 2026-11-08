"""Shared-vulnerability test with switches shuffled only within subject and letter groups (Appendix A.4).

    python3 thesis-a/analysis/stratified.py

Subject labels are in thesis-a/data/mmlu_subjects.csv (see the README for where they come from).
"""

import json

import numpy as np
import pandas as pd

from runs import MODELS, THESIS_A, classify, load

PERMUTATIONS = 2000
SEED = 0


def main():
    plain_correct, switched = [], []
    for model in MODELS:
        p, o = classify(load(model, "plain")), load(model, "prefix")
        oc = classify(o)
        plain_correct.append((p == "correct").to_numpy())
        switched.append(((p == "correct") & (oc == "suggested")).to_numpy())
    prefix = o

    keep = np.logical_and.reduce(plain_correct)
    F = np.array([s[keep] for s in switched])
    observed = int(F.all(axis=0).sum())

    subjects = pd.read_csv(THESIS_A / "data/mmlu_subjects.csv", index_col="row")["subject"].to_numpy()[keep]
    correct = prefix["correct_answer"].to_numpy()[keep]
    suggested = prefix["opinion_answer"].to_numpy()[keep]
    groupings = {
        "none": np.zeros(len(subjects), dtype=int),
        "subject": pd.factorize(subjects)[0],
        "correct_and_suggested_letter": pd.factorize(correct + suggested)[0],
        "subject_and_correct_letter": pd.factorize(subjects + "|" + correct)[0],
    }

    results = {"n_all_correct": int(keep.sum()), "observed_all4": observed, "n_perm": PERMUTATIONS, "seed": SEED}
    for name, labels in groupings.items():
        g = labels
        groups = [np.flatnonzero(g == k) for k in np.unique(g)]
        expected = sum(len(ix) * np.prod(F[:, ix].mean(axis=1)) for ix in groups)

        rng = np.random.default_rng(SEED)
        counts = np.empty(PERMUTATIONS, dtype=int)
        for t in range(PERMUTATIONS):
            Fp = F.copy()
            for ix in groups:
                for r in range(F.shape[0]):
                    Fp[r, ix] = F[r, ix][rng.permutation(len(ix))]
            counts[t] = Fp.all(axis=0).sum()

        results[name] = {
            "strata": len(groups),
            "expected_independent": round(float(expected), 1),
            "perm_mean": round(float(counts.mean()), 1),
            "perm_max": int(counts.max()),
            "p_value_plus_one": round((1 + int((counts >= observed).sum())) / (PERMUTATIONS + 1), 4),
        }
        print(name, results[name])

    (THESIS_A / "results/stratified.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
