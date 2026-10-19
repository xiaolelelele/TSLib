"""Exact rational counterexample: class posteriors do not identify MSE risk.

This is an algebraic mechanism check, not a new theorem claim or a learned
forecast benchmark. It uses no experimental test data and writes fresh output.
"""
import argparse
import json
from fractions import Fraction as Q
from pathlib import Path


def analyze():
    epsilon, base, candidate = Q(1, 20), Q(0), Q(1, 10)
    cases = {
        "conditional_mean_base": [(Q(4, 5), Q(1, 5)), (Q(1, 5), Q(-4, 5))],
        "biased_base_same_class_posterior": [(Q(4, 5), Q(4, 5)), (Q(1, 5), Q(-1, 5))],
    }
    def label(y):
        return 2 if y > epsilon else 0 if y < -epsilon else 1
    rows = []
    for name, outcomes in cases.items():
        mean = sum(p*y for p, y in outcomes)
        posterior = [sum(p for p, y in outcomes if label(y) == c) for c in range(3)]
        risk = lambda z: sum(p*(z-y)**2 for p, y in outcomes)
        gain = lambda z: sum(p for p, y in outcomes if label(z) == label(y))
        delta = candidate-base
        regret = risk(candidate)-risk(base)
        assert regret == delta**2+2*delta*(base-mean)
        rows.append({"case": name, "outcomes": [[str(p), str(y)] for p, y in outcomes],
                     "posterior_down_flat_up": list(map(float, posterior)),
                     "conditional_mean": float(mean), "base": float(base),
                     "candidate": float(candidate), "base_mse": float(risk(base)),
                     "candidate_mse": float(risk(candidate)), "mse_change": float(regret),
                     "mse_change_exact": str(regret), "base_class_accuracy": float(gain(base)),
                     "candidate_class_accuracy": float(gain(candidate)),
                     "distance_to_base_squared": float(delta**2)})
    assert rows[0]["posterior_down_flat_up"] == rows[1]["posterior_down_flat_up"]
    assert rows[0]["mse_change"] > 0 > rows[1]["mse_change"]
    return {"purpose": __doc__, "epsilon": float(epsilon), "rows": rows,
            "conclusion": "Same class posterior, base and candidate cost; opposite MSE changes. "
                          "Class probabilities alone cannot identify correction MSE risk."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(analyze(), stream, indent=2)
        stream.write("\n")
