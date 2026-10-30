#!/usr/bin/env python3
"""E5 report: the four E4 conditions re-read on step-matched arms, with dose stated.

Everything here is read from run artifacts; the only computation is the paired
per-question bootstrap, which is the project's existing instrument (`bootstrap_gate`),
called rather than reimplemented.

The arms and what each one is for:

* `e5_sleep_qa`    -- the sleep arm at a FIXED 30 SFT steps (review2 A1's fix);
* `e5_sleep_recap` -- the sleep arm on recap PROSE dreams, whole-text loss: the dose-
  comparable dream arm (review2 A2);
* `e5_cpt`         -- SFT-on-C rerun in the E5 session at the same fixed 30 steps: it is
  the only way to state the cpt arm's supervised-token dose (E4 predates the counter),
  and it doubles as a cross-session reproducibility check against `e4_cpt`.
* `e4_nodream`     -- reused, not rerun: it already trains exactly 30 steps with the same
  build, warmup, bias tune and batch, so A1 does not change it. Its dose is reported as
  unavailable (pre-instrument) and the residual mismatch is stated, not hidden.

    python3 scripts/e5_report.py --out runs/e5_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from bootstrap_gate import main as bootstrap  # noqa: E402

ARMS = {
    "sleep_qa": "e5_sleep_qa",
    "sleep_recap": "e5_sleep_recap",
    "cpt": "e5_cpt",
    "nodream": "e4_nodream",
}


def load(run: str) -> dict:
    return json.loads((Path("runs") / run / "summary.json").read_text())


def cycle_rows(run: str) -> list[dict]:
    path = Path("runs") / run / "metrics.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r for r in rows if r.get("stage") == "cycle"]


def median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])


def arm_summary(run: str) -> dict:
    summary = load(run)
    cycles = cycle_rows(run)
    shares = [p.get("share_heldout") for p in summary["per_passage"]
              if p.get("share_heldout") is not None]
    return {
        "run": run,
        "sft_mode": summary["sft_mode"],
        "dream_variant": summary.get("dream_variant"),
        "dream_temperature": summary.get("dream_temperature"),
        "n_passages": summary["n_passages"],
        "n_questions": summary["n_questions"],
        "pooled_base_em": summary["pooled_base_em"],
        "pooled_final_em": summary["pooled_final_em"],
        "pooled_gain_pts": summary["pooled_gain"],
        "dose": summary.get("dose", {"note": "not recorded by this run (pre-instrument)"}),
        "heldout_post_surgery_em": summary.get("heldout_post_surgery_em"),
        "heldout_final_em": summary.get("heldout_final_em"),
        "heldout_drop": summary.get("heldout_drop"),
        "n_heldout_questions": summary.get("n_heldout_questions"),
        "share_heldout_range": [min(shares), max(shares)] if shares else None,
        "band": [0.05, 0.09],
        "prose_overlap_median_of_passage_medians": median(
            [c["prose_overlap_median"] for c in cycles if c.get("prose_overlap_median") is not None]
        ),
        "answer_support_median_of_passage_values": median(
            [c["answer_support"] for c in cycles if c.get("answer_support") is not None]
        ),
        "pairs_per_passage_mean": (sum(c["n_pairs"] for c in cycles) / len(cycles)
                                  if cycles else None),
        "mean_dream_words": (sum(c["mean_words"] for c in cycles) / len(cycles)
                             if cycles and "mean_words" in cycles[0] else None),
        "literalism_verdicts": sorted({c["verdict"] for c in cycles}) if cycles else None,
        "warmup_text_kind": sorted({r["kind"] for r in
                                    (json.loads(line) for line in
                                     (Path("runs") / run / "metrics.jsonl").read_text().splitlines()
                                     if line.strip())
                                    if r.get("stage") == "warmup_texts"}),
        "gates": summary.get("gates"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/e5_report.json")
    args = ap.parse_args()

    arms = {name: arm_summary(run) for name, run in ARMS.items()}

    boot: dict[str, dict] = {}
    for name, run in ARMS.items():
        boot[f"{name}_vs_own_base"] = bootstrap(f"runs/{run}/summary.json")
    contrasts = [
        ("sleep_qa", "cpt"), ("sleep_recap", "cpt"),
        ("sleep_qa", "nodream"), ("sleep_recap", "nodream"),
        ("sleep_qa", "sleep_recap"), ("cpt", "nodream"),
    ]
    for left, right in contrasts:
        # bootstrap(base, final) reports final - base, so the ARM ORDER is reversed here
        # to make the key `{left}_minus_{right}` mean left - right. Getting this backwards
        # inverts every condition below: with the arguments in key order the recap arm's
        # -8.26 pts against cpt read as +8.26 and both comparator conditions came out PASS.
        key = f"{left}_minus_{right}"
        boot[key] = bootstrap(f"runs/{ARMS[right]}/summary.json",
                              f"runs/{ARMS[left]}/summary.json")
        # the sign is the thing that broke, so it is checked against the arms' own pooled
        # numbers rather than trusted
        expected = arms[left]["pooled_final_em"] - arms[right]["pooled_final_em"]
        if abs(boot[key]["gain_pts"] - expected) > 1e-6:
            raise SystemExit(f"sign/order defect in {key}: bootstrap says "
                             f"{boot[key]['gain_pts']:+.2f} pts, arm pooled EMs say "
                             f"{expected:+.2f} pts")

    min_gain = 2.0
    conditions = {}
    for name in ("sleep_qa", "sleep_recap"):
        own = boot[f"{name}_vs_own_base"]
        vs_cpt = boot[f"{name}_minus_cpt"]
        vs_nodream = boot[f"{name}_minus_nodream"]
        heldout_drop = arms[name]["heldout_drop"]
        conditions[name] = {
            "i_gain_over_own_base": {
                "pass": bool(own["gain_pts"] >= min_gain and own["ci95_pts"][0] > 0),
                "gain_pts": own["gain_pts"], "ci95_pts": own["ci95_pts"],
                "n_flipped_up": own["n_flipped_up"], "n_flipped_down": own["n_flipped_down"],
            },
            "ii_sleep_at_least_cpt_plus_2": {
                "pass": bool(vs_cpt["gain_pts"] >= min_gain),
                "sleep_minus_cpt_pts": vs_cpt["gain_pts"], "ci95_pts": vs_cpt["ci95_pts"],
            },
            "iii_sleep_at_least_nodream_plus_2": {
                "pass": bool(vs_nodream["gain_pts"] >= min_gain),
                "sleep_minus_nodream_pts": vs_nodream["gain_pts"],
                "ci95_pts": vs_nodream["ci95_pts"],
            },
            "iv_non_destruction_heldout_drop_le_2": {
                "pass": bool(heldout_drop is not None and heldout_drop <= 2.0),
                "heldout_drop": heldout_drop,
                "heldout_post_surgery_em": arms[name]["heldout_post_surgery_em"],
                "heldout_final_em": arms[name]["heldout_final_em"],
                "n_heldout_questions": arms[name]["n_heldout_questions"],
            },
        }

    report = {
        "arms": arms,
        # E4's cpt arm, kept as a cross-session reproducibility reading of the arm the
        # E5 rerun replaces (same steps, same build; a difference would be session noise)
        "reference_e4_cpt": (arm_summary("e4_cpt")
                             if (Path("runs") / "e4_cpt" / "summary.json").exists() else None),
        "bootstraps": boot,
        "conditions": conditions,
        "contrast_objective_at_equal_steps": {
            "sleep_qa_minus_sleep_recap": boot["sleep_qa_minus_sleep_recap"],
            "note": ("same build, warmup, bias tune and 30 SFT steps; the arms differ in "
                     "dream prompt, objective and supervised-token dose -- read the dose "
                     "columns before attributing a difference to the objective"),
        },
        "contrast_recap_minus_cpt_review2_A3": boot["sleep_recap_minus_cpt"],
    }
    Path(args.out).write_text(json.dumps(report, indent=2))

    print("arm                base   final   gain   dose(tok)  share_ho        heldout")
    for name, arm in arms.items():
        dose = arm["dose"].get("supervised_tokens_mean")
        share = arm["share_heldout_range"]
        print(f"{name:18s} {arm['pooled_base_em']:6.2f} {arm['pooled_final_em']:6.2f} "
              f"{arm['pooled_gain_pts']:+6.2f} {str(dose):>10s}  "
              f"{('%.3f-%.3f' % (share[0], share[1])) if share else 'n/a':15s} "
              f"{arm['heldout_post_surgery_em']} -> {arm['heldout_final_em']}")
    for name, cond in conditions.items():
        print(f"\n[{name}]")
        for key, value in cond.items():
            print(f"  {key}: {'PASS' if value['pass'] else 'FAIL'} {json.dumps({k: v for k, v in value.items() if k != 'pass'})}")
    print(f"\n[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
