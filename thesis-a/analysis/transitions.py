"""Plain to Prefix answer transitions and the shared-vulnerability test (Section 4.2.3).

    python3 thesis-a/analysis/transitions.py
"""

import json

import numpy as np

from runs import MODELS, THESIS_A, classify, load

PERMUTATIONS = 2000
SEED = 0


def main():
    results = {}
    plain_correct, switched = {}, {}

    for model in MODELS:
        plain, prefix = load(model, "plain"), load(model, "prefix")
        assert (plain["correct_answer"] == prefix["correct_answer"]).all()
        p, o = classify(plain), classify(prefix)
        valid_p, valid_o = (p != "invalid").sum(), (o != "invalid").sum()
        pc, pw, pi = p == "correct", p == "other", p == "invalid"

        r = {
            "plain_invalid": int(pi.sum()),
            "prefix_invalid": int((o == "invalid").sum()),
            "plain_acc_valid": round(100 * pc.sum() / valid_p, 2),
            "prefix_acc_valid": round(100 * (o == "correct").sum() / valid_o, 2),
            "prefix_syc_valid": round(100 * (o == "suggested").sum() / valid_o, 2),
            "n_plain_correct": int(pc.sum()),
            "plain_wrong": int(pw.sum()),
        }
        for k in ["correct", "suggested", "other", "invalid"]:
            r[f"pc_to_{k}"] = int((pc & (o == k)).sum())
            r[f"pw_to_{k}"] = int((pw & (o == k)).sum())
        r["plain_invalid_to_correct"] = int((pi & (o == "correct")).sum())
        r["wrong_or_invalid_to_correct"] = r["pw_to_correct"] + r["plain_invalid_to_correct"]
        r["pw_plainans_eq_suggested"] = int((pw & (plain["model_answer"] == prefix["opinion_answer"])).sum())
        r["pct_pc_to_suggested"] = round(100 * r["pc_to_suggested"] / r["n_plain_correct"], 1)
        r["pct_pc_kept"] = round(100 * r["pc_to_correct"] / r["n_plain_correct"], 1)
        results[model] = r

        plain_correct[model] = pc.to_numpy()
        switched[model] = (pc & (o == "suggested")).to_numpy()

    # Questions every model got right in Plain, and which of those switched in each model
    keep = np.logical_and.reduce(list(plain_correct.values()))
    F = np.array([switched[m][keep] for m in MODELS])
    rates = F.mean(axis=1)

    rng = np.random.default_rng(SEED)
    perm = np.array([np.array([rng.permutation(row) for row in F]).all(axis=0).sum() for _ in range(PERMUTATIONS)])

    results["all4_plain_correct"] = int(keep.sum())
    results["all4_flip_to_suggested"] = int(F.all(axis=0).sum())
    results["all4_flip_expected_indep"] = round(float(keep.sum() * np.prod(rates)), 1)
    results["all4_never_flip"] = int((~F.any(axis=0)).sum())
    results["all4_never_flip_expected_indep"] = round(float(keep.sum() * np.prod(1 - rates)), 1)
    results["all4_flip_perm_mean"] = round(float(perm.mean()), 1)
    results["all4_flip_perm_max"] = int(perm.max())
    results["all4_flip_perm_n"] = PERMUTATIONS

    out = THESIS_A / "results/transitions.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
