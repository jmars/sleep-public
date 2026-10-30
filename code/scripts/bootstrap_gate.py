"""Paired per-question bootstrap for the pooled gates.

A 109-question set at ~12 EM has a sampling spread of several points, so a bare mean
difference cannot decide a 2-point gate. Every question is asked of the SAME base and
the SAME trained student, so the comparison is paired: resample questions with
replacement and report the distribution of (final - base).

Decision rule, fixed before the numbers are read:
  PASS        mean gain >= +2.0 pts AND the 95% CI lower bound > 0
  FAIL        mean gain <= 0
  INSUFFICIENT otherwise (CI spans 0, or mean in (0, 2))

`--metric {em,f1}`: the corpus protocol's transfer frame is scored on token F1 (the base
is near the floor there, so EM is too granular), and the same pairing instrument is used
for both metrics rather than a second implementation.

    python3 scripts/bootstrap_gate.py runs/e5_sleep_qa/summary.json            # own-base
    python3 scripts/bootstrap_gate.py runs/a/summary.json runs/b/summary.json  # b - a
    python3 scripts/bootstrap_gate.py --metric f1 runs/p6_s/summary.json runs/p6_c/summary.json
"""
import argparse
import json
import random
import re
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qlora_sleep.eval_squad import token_f1  # noqa: E402

MIN_GAIN = 2.0


def load(path):
    """Read a run either as `metrics.jsonl` or as `summary.json`.

    `run_multi.py` keeps per-passage predictions in summary.json (the JSONL line
    filter drops list fields), so both shapes have to be readable here.
    """
    text = open(path).read()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if "per_passage" in obj:
        rows = []
        for row in obj["per_passage"]:
            rows.append({"stage": "passage", "predictions": row["predictions"],
                         "golds": row["golds"],
                         # the untrained student's own answers on the same questions: the
                         # paired own-base diff the gate verdict rests on
                         "base_predictions": row.get("base_predictions")})
        return rows
    return [obj]


def per_question(rows, stages=("eval", "passage"), key="predictions"):
    """Per-question (prediction, golds) in file order.

    Both files iterate the same questions in the same order, so index i is the same
    question in each -- that is what makes the comparison paired.
    """
    out = []
    for row in rows:
        if row.get("stage") not in stages or not row.get("predictions"):
            continue
        preds = row.get(key)
        golds = row["golds"]
        if len(preds) != len(golds):
            continue
        for pred, gold in zip(preds, golds):
            out.append((pred, gold))
    return out


def normalise(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def em(pred, golds):
    p = normalise(pred)
    return float(any(p == normalise(g) for g in golds))


def score(pred, golds, metric="em"):
    """One question's score in [0, 1]. Both metrics come from `eval_squad`."""
    if metric == "em":
        return em(pred, golds)
    if metric == "f1":
        return token_f1(pred, golds)
    raise ValueError(f"unknown metric {metric!r}")


def _bootstrap(diffs, n_boot=20000, seed=0):
    """Resample per-question differences; return (mean, lo95, hi95) in points."""
    n = len(diffs)
    mean = sum(diffs) / n
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    return 100.0 * mean, 100.0 * boots[int(0.025 * n_boot)], 100.0 * boots[int(0.975 * n_boot)]


def _align(base_pairs, final_pairs, label=""):
    if len(base_pairs) != len(final_pairs):
        print(f"WARNING: base has {len(base_pairs)} answered questions, final has "
              f"{len(final_pairs)}")
    n = min(len(base_pairs), len(final_pairs))
    if n == 0:
        raise ValueError(f"no paired questions for {label or 'this comparison'}")
    return n


def paired(base_pairs, final_pairs, metric="em", n_boot=20000, seed=0, label=""):
    """Bootstrap the per-question difference `final - base` in points.

    `*_pairs` are lists of `(prediction, golds)` in the same question order. The
    returned `gain_pts` is final minus base, i.e. the SECOND argument's score minus the
    first's -- the sign convention the callers' keys depend on.
    """
    n = _align(base_pairs, final_pairs, label)
    diffs = [
        score(final_pairs[i][0], final_pairs[i][1], metric)
        - score(base_pairs[i][0], base_pairs[i][1], metric)
        for i in range(n)
    ]
    gain, lo, hi = _bootstrap(diffs, n_boot=n_boot, seed=seed)
    base_score = 100.0 * sum(score(p, g, metric) for p, g in base_pairs[:n]) / n
    final_score = 100.0 * sum(score(p, g, metric) for p, g in final_pairs[:n]) / n
    if gain >= MIN_GAIN and lo > 0:
        verdict = "PASS"
    elif gain <= 0:
        verdict = "FAIL"
    else:
        verdict = "INSUFFICIENT"
    result = {
        "n_questions": n,
        f"base_{metric}": base_score,
        f"final_{metric}": final_score,
        "gain_pts": gain,
        "ci95_pts": [lo, hi],
        "n_flipped_up": int(sum(1 for d in diffs if d > 0)),
        "n_flipped_down": int(sum(1 for d in diffs if d < 0)),
        "verdict": verdict,
        "metric": metric,
        "rule": "PASS = gain >= +2.0 pts AND CI95 low > 0; FAIL = gain <= 0; else INSUFFICIENT",
    }
    if label:
        result["label"] = label
    return result


def paired_gain_difference(base_a, final_a, base_b, final_b, metric="em",
                           n_boot=20000, seed=0, label=""):
    """Bootstrap mean[(final_a - base_a) - (final_b - base_b)] in points.

    The dose-efficiency reading needs the DIFFERENCE OF TWO GAINS measured on the same
    questions, not two gains compared as point estimates: per-question gains are paired
    and their difference is resampled.
    """
    n = _align(base_a, final_a, label)
    n = min(n, _align(base_b, final_b, label))
    diffs = []
    for i in range(n):
        gain_a = (score(final_a[i][0], final_a[i][1], metric)
                  - score(base_a[i][0], base_a[i][1], metric))
        gain_b = (score(final_b[i][0], final_b[i][1], metric)
                  - score(base_b[i][0], base_b[i][1], metric))
        diffs.append(gain_a - gain_b)
    gain, lo, hi = _bootstrap(diffs, n_boot=n_boot, seed=seed)
    return {
        "n_questions": n,
        "gain_diff_pts": gain,
        "ci95_pts": [lo, hi],
        "metric": metric,
        "n_favouring_a": int(sum(1 for d in diffs if d > 0)),
        "n_favouring_b": int(sum(1 for d in diffs if d < 0)),
        "label": label,
    }


def main(base_path, final_path=None, n_boot=20000, seed=0, metric="em"):
    if final_path is None:
        # OWN-BASE mode: one summary.json that carries both frames. This is the pairing
        # the gate verdict should rest on -- the untrained student is the same object as
        # the trained one, measured through the same closure on the same card, so no
        # second numeric frame enters the comparison.
        rows = load(base_path)
        base = per_question(rows, key="base_predictions")
        final = per_question(rows)
    else:
        base = per_question(load(base_path))
        final = per_question(load(final_path))
    result = paired(base, final, metric=metric, n_boot=int(n_boot), seed=int(seed))
    print(json.dumps(result, indent=1))
    return result


def cli(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help="one summary.json (own-base) or base then final")
    ap.add_argument("--metric", choices=["em", "f1"], default="em")
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    main(*args.paths[:2], n_boot=args.n_boot, seed=args.seed, metric=args.metric)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
