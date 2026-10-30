#!/usr/bin/env python3
"""Residual test report: dreams written under router perturbation vs standard dreams.

Two sleep-qa arms on the SAME card, same seed, same build/warmup/bias band, same
steps=30, differing only in the routing state installed while the teacher wrote the
dreams:

    R-std   standard-routing teacher (the as-validated E4/E5 dream cache)
    R-pert  the same pass with the teacher's standard top-k slots partly REPLACED by
            non-top-k experts, drawn from the router distribution itself

The gate is the DIFFERENCE OF THE TWO OWN-BASE PAIRED GAINS, bootstrapped per question on
the corpus frame: `(pert_final - pert_base) - (std_final - std_base)`. Two gains compared
as point estimates would ignore the pairing; the per-question pairing is the reason both
arms can be read at 109 questions at all.

Pre-registered readings (brief9):
    MECHANISM_ISOLATED   CI95 excludes 0 in R-pert's favour
    INSUFFICIENT         CI95 contains 0 -- perturbation alone does not move injection
    DESTRUCTIVE          CI95 excludes 0 against R-pert

    python3 scripts/residual_report.py --std runs/r_std/summary.json \
        --pert runs/r_pert/summary.json --out runs/residual_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from bootstrap_gate import load, paired, paired_gain_difference, per_question  # noqa: E402

SIGN_TOL = 1e-6


def _wall_seconds(metrics_path: Path) -> float | None:
    """Span of the run's own telemetry timestamps (the only per-arm clock that survives)."""
    if not metrics_path.exists():
        return None
    stamps = []
    for line in metrics_path.read_text().splitlines():
        if not line.strip():
            continue
        stamp = json.loads(line).get("t")
        if isinstance(stamp, (int, float)):
            stamps.append(float(stamp))
    return (max(stamps) - min(stamps)) if stamps else None


def arm_readings(path: Path, metric: str, n_boot: int, seed: int) -> dict:
    """One arm's own-base paired reading on the corpus frame, plus its held-out guard."""
    summary = json.loads(path.read_text())
    rows = load(path)
    base = per_question(rows, key="base_predictions")
    final = per_question(rows)
    corpus = paired(base, final, metric=metric, n_boot=n_boot, seed=seed,
                    label=f"corpus_orig own-base ({path.parent.name})")
    return {
        "run": summary.get("run"),
        "summary_path": str(path),
        "sft_mode": summary.get("sft_mode"),
        "n_questions": summary.get("n_questions"),
        "steps": (summary.get("dose") or {}).get("steps_mean"),
        "supervised_tokens_total": (summary.get("dose") or {}).get("supervised_tokens_total"),
        f"corpus_{metric}": corpus,
        "pooled_base_em": summary.get("pooled_base_em"),
        "pooled_final_em": summary.get("pooled_final_em"),
        "heldout_post_surgery_em": summary.get("heldout_post_surgery_em"),
        "heldout_final_em": summary.get("heldout_final_em"),
        "heldout_drop": summary.get("heldout_drop"),
        "heldout_n_questions": summary.get("n_heldout_questions"),
        "dream_perturb": summary.get("dream_perturb"),
        "eval_routing_identity": summary.get("eval_routing_identity"),
        "gates": summary.get("gates"),
        # WALL from the metrics timestamps, not summary["timings"]: an arm whose summary was
        # rebuilt by --resume carries the resume pass's wall (2.8 s for a 71-minute arm).
        "wall_s": _wall_seconds(path.parent / "metrics.jsonl"),
        "_pairs": (base, final),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--std", required=True, help="R-std summary.json")
    ap.add_argument("--pert", required=True, help="R-pert summary.json")
    ap.add_argument("--out", default="runs/residual_report.json")
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    report: dict = {
        "protocol": ("residual test: dreams written under router perturbation vs standard "
                     "dreams, gate24 20-passage frame, quiz@0.7, steps=30, own-base paired"),
        "arms": {},
        "contrast": {},
    }
    pairs: dict[str, tuple] = {}
    for key, path in (("R_std", args.std), ("R_pert", args.pert)):
        readings = arm_readings(Path(path), "em", args.n_boot, args.seed)
        pairs[key] = readings.pop("_pairs")
        f1 = paired(pairs[key][0], pairs[key][1], metric="f1", n_boot=args.n_boot,
                    seed=args.seed, label=f"corpus_orig own-base F1 ({key})")
        readings["corpus_f1"] = f1
        report["arms"][key] = readings

    for metric in ("em", "f1"):
        base_std, final_std = pairs["R_std"]
        base_pert, final_pert = pairs["R_pert"]
        # a = R_pert, b = R_std: the returned gain_diff is (pert gain) - (std gain)
        contrast = paired_gain_difference(base_pert, final_pert, base_std, final_std,
                                          metric=metric, n_boot=args.n_boot, seed=args.seed,
                                          label=f"R_pert - R_std corpus frame ({metric})")
        # SIGN ASSERTION: the bootstrap must agree with the direct difference of the two
        # pooled gains. The E5 report shipped a sign defect; recomputing here makes that
        # class structurally impossible rather than merely checked.
        direct = (report["arms"]["R_pert"][f"corpus_{metric}"]["gain_pts"]
                  - report["arms"]["R_std"][f"corpus_{metric}"]["gain_pts"])
        if abs(contrast["gain_diff_pts"] - direct) > SIGN_TOL:
            raise SystemExit(
                f"SIGN/RECOMPUTE MISMATCH on {metric}: bootstrap {contrast['gain_diff_pts']} "
                f"vs direct {direct} -- refusing to publish a report whose contrast does "
                "not recompute"
            )
        contrast["direct_gain_difference_pts"] = direct
        lo, hi = contrast["ci95_pts"]
        if lo > 0:
            verdict = "MECHANISM_ISOLATED"
        elif hi < 0:
            verdict = "DESTRUCTIVE"
        else:
            verdict = "INSUFFICIENT"
        contrast["verdict"] = verdict
        report["contrast"][metric] = contrast

    # SHARED INITIAL CONDITION: the two arms are only comparable if their own-base reads
    # are the same object. Both are built with the same seed/build/warmup/bias band on the
    # same card, so the per-passage base EM lists must agree exactly; a mismatch would make
    # the contrast an initial-condition difference rather than a dream-text difference.
    def _per_passage(path: Path, key: str) -> list:
        return [row[key] for row in json.loads(Path(path).read_text())["per_passage"]]

    base_std = _per_passage(Path(args.std), "base_em")
    base_pert = _per_passage(Path(args.pert), "base_em")
    report["initial_condition"] = {
        "n_passages": len(base_std),
        "identical_base_em_per_passage": base_std == base_pert,
        "n_differing": sum(1 for a, b in zip(base_std, base_pert) if a != b),
        "base_em_std": base_std,
        "base_em_pert": base_pert,
    }
    report["verdict"] = report["contrast"]["em"]["verdict"]
    report["verdict_rule"] = (
        "MECHANISM_ISOLATED = paired CI95 of (R_pert gain - R_std gain) excludes 0 in "
        "R_pert's favour on the corpus frame (EM primary); DESTRUCTIVE = excludes 0 "
        "against it; INSUFFICIENT = contains 0"
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    for key in ("R_std", "R_pert"):
        arm = report["arms"][key]
        c = arm["corpus_em"]
        print(f"[{key}] corpus EM {c['base_em']:.2f} -> {c['final_em']:.2f} "
              f"gain {c['gain_pts']:+.2f} {c['ci95_pts']} | F1 gain "
              f"{arm['corpus_f1']['gain_pts']:+.2f} | heldout {arm['heldout_post_surgery_em']} "
              f"-> {arm['heldout_final_em']} | tokens {arm['supervised_tokens_total']} | "
              f"a1 {None if not arm['dream_perturb'] else arm['dream_perturb'].get('a1_off_standard_fraction')}")
    for metric in ("em", "f1"):
        con = report["contrast"][metric]
        print(f"[contrast {metric}] R_pert - R_std {con['gain_diff_pts']:+.2f} pts "
              f"CI95 {con['ci95_pts']} (direct {con['direct_gain_difference_pts']:+.2f}) "
              f"-> {con['verdict']}")
    ic = report["initial_condition"]
    print(f"[initial condition] base EM identical on {ic['n_passages']} passages: "
          f"{ic['identical_base_em_per_passage']} ({ic['n_differing']} differing)")
    print(f"[verdict] {report['verdict']}  (written to {out})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
