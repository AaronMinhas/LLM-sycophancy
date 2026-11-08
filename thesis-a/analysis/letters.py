"""Prefix agreement by suggested answer letter, and Plain answer-letter shares (Appendix A.1).

    python3 thesis-a/analysis/letters.py
"""

import json

from runs import MODELS, THESIS_A, VALID, load


def main():
    results = {}
    for model in MODELS:
        plain, prefix = load(model, "plain"), load(model, "prefix")
        suggested = prefix["opinion_answer"]
        agree = prefix["model_answer"] == suggested
        results[model] = {
            "agree_pct_by_suggested_letter": {L: round(100 * agree[suggested == L].mean(), 1) for L in VALID},
            "n_by_suggested_letter": {L: int((suggested == L).sum()) for L in VALID},
            "plain_answer_share_pct": {L: round(100 * (plain["model_answer"] == L).mean(), 1) for L in VALID},
            "correct_letter_share_pct": {L: round(100 * (prefix["correct_answer"] == L).mean(), 1) for L in VALID},
        }

    (THESIS_A / "results/letters.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
