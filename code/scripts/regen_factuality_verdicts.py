#!/usr/bin/env python3
"""Regenerate the E1 factuality verdicts OFFLINE from the metrics rows (review2 B2).

The problem this fixes: `runs/e1_factuality/summary.json` was written by the sweep
BEFORE the `verdict()` fix, so three cells (facts@0.9, recap@0.9, facts@0.7) still read
`g1_verdict: FAIL` while the fixed logic returns NO_PAIRS -- their mean pairs per
passage is 0.00-0.08, i.e. the QA extractor is question-shaped and cannot see a prose
dream variant at all. A future reader scanning the grid would see "recap@0.9 support
0.000 FAIL" and draw exactly the over-read the fix exists to prevent.

Everything here is recomputable from `runs/e1_factuality/metrics.jsonl`
(`factuality_passage` rows, one per cell x passage): no GPU, no generation, no model.
Two things are written back:

* every cell's `g1_verdict` recomputed with the CURRENT `verdict()` (the previous value
  is kept as `g1_verdict_prev`, so the change is auditable rather than silent);
* the prose-grounding reading (`prose_grounding`) computed from the retained dream
  samples, because `answer_support_rate` is structurally blind to prose variants -- the
  reviewer's finding B: recap is grounded paraphrase (content overlap 0.67-1.00) and
  facts is off-topic (0.00-0.13), and BOTH read support 0.0 / NO_PAIRS.

The recomputation is checked against the numbers already in the artifact: any cell whose
recomputed support or pair count disagrees with the stored value is reported as a
MISMATCH and the script exits non-zero, so a quiet divergence cannot pass as a fix.

    python3 scripts/regen_factuality_verdicts.py --run e1_factuality
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

from squad_prep import read_jsonl  # noqa: E402

from qlora_sleep.dream import content_overlap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_factuality import verdict  # noqa: E402  (the ONE verdict implementation)

TOL = 1e-9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="e1_factuality")
    ap.add_argument("--data", default="data/squad_gate24.jsonl")
    ap.add_argument("--min-pairs", type=float, default=10.0)
    ap.add_argument("--pass-support", type=float, default=0.70)
    ap.add_argument("--insufficient-support", type=float, default=0.55)
    args = ap.parse_args()

    run_dir = Path("runs") / args.run
    summary_path = run_dir / "summary.json"
    metrics_path = run_dir / "metrics.jsonl"
    summary = json.loads(summary_path.read_text())
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    passages = {p.get("passage_id", ""): p["context"]
                for p in read_jsonl(args.data)["passages"]}

    cells: dict[tuple, list[dict]] = {}
    for row in rows:
        if row.get("stage") != "factuality_passage":
            continue
        cells.setdefault((row["variant"], float(row["temperature"])), []).append(row)

    mismatches: list[dict] = []
    changes: list[dict] = []
    for cell in summary["grid"]:
        key = (cell["variant"], float(cell["temperature"]))
        got = cells.get(key, [])
        supports = [r["answer_support"] for r in got]
        pairs = [r["n_pairs"] for r in got]
        mean_support = sum(supports) / len(supports) if supports else 0.0
        mean_pairs = sum(pairs) / len(pairs) if pairs else 0.0
        for name, mine, stored in (("mean_answer_support", mean_support,
                                    cell["mean_answer_support"]),
                                   ("mean_pairs_per_passage", mean_pairs,
                                    cell["mean_pairs_per_passage"])):
            if abs(mine - stored) > TOL:
                mismatches.append({"cell": f"{key[0]}@{key[1]}", "field": name,
                                   "recomputed": mine, "stored": stored})

        prev = cell["g1_verdict"]
        new = verdict(mean_support, mean_pairs, pass_support=args.pass_support,
                      insufficient_support=args.insufficient_support,
                      min_pairs=args.min_pairs)
        cell["g1_verdict_prev"] = prev
        cell["g1_verdict"] = new
        cell["verdict_source"] = (
            "recomputed offline from metrics.jsonl factuality_passage rows with the "
            "current verdict() (review2 B2); g1_verdict_prev is the pre-fix value"
        )
        # the prose reading is per dream, pooled over the retained samples of the cell
        overlaps: list[float] = []
        for sample in cell.get("samples", []):
            context = passages.get(sample["passage_id"])
            if context is None:
                continue
            for dream in sample["dreams"]:
                overlaps.append(content_overlap(dream, context))
        if overlaps:
            ordered = sorted(overlaps)
            cell["prose_overlap"] = {
                "n_dreams": len(overlaps),
                "median": ordered[len(ordered) // 2],
                "min": ordered[0],
                "max": ordered[-1],
                "instrument": "content_overlap: content words of the dream that occur in "
                              "the source passage (stop words removed)",
            }
        if prev != new:
            changes.append({"cell": f"{key[0]}@{key[1]}", "prev": prev, "new": new,
                            "mean_pairs_per_passage": mean_pairs})

    order = {"PASS": 0, "INSUFFICIENT": 1, "NO_PAIRS": 2, "FAIL": 3}
    ranked = sorted(summary["grid"],
                    key=lambda c: (order.get(c["g1_verdict"], 9),
                                   -c["mean_answer_support"], -c["mean_pairs_per_passage"]))
    best = ranked[0]
    overall_prev = summary["g1_verdict"]
    if any(c["g1_verdict"] == "PASS" for c in summary["grid"]):
        overall = "PASS"
    elif any(c["g1_verdict"] == "FAIL" for c in summary["grid"]):
        overall = "FAIL"
    else:
        overall = "INSUFFICIENT"
    summary["g1_verdict_prev"] = overall_prev
    summary["g1_verdict"] = overall
    summary["best_cell"] = {
        "variant": best["variant"], "temperature": best["temperature"],
        "mean_answer_support": best["mean_answer_support"],
        "mean_pairs_per_passage": best["mean_pairs_per_passage"],
        "g1_verdict": best["g1_verdict"],
    }
    summary["verdict_regenerated"] = {
        "by": "scripts/regen_factuality_verdicts.py",
        "source": "runs/%s/metrics.jsonl factuality_passage rows" % args.run,
        "cells_changed": changes,
        "mismatches": mismatches,
        "rule": ("verdict(): NO_PAIRS when mean pairs/passage < 2.0 (the extractor cannot "
                 "see this cell), PASS at support >= %.2f and pairs >= %.0f, FAIL below "
                 "%.2f, INSUFFICIENT otherwise"
                 % (args.pass_support, args.min_pairs, args.insufficient_support)),
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"[regen] cells changed: {len(changes)}; mismatches: {len(mismatches)}")
    for change in changes:
        print(f"  {change['cell']}: {change['prev']} -> {change['new']} "
              f"(pairs/passage {change['mean_pairs_per_passage']:.2f})")
    for cell in summary["grid"]:
        prose = cell.get("prose_overlap")
        prose_txt = (f"prose_overlap median {prose['median']:.2f} "
                     f"[{prose['min']:.2f},{prose['max']:.2f}]") if prose else "no samples"
        print(f"  {cell['variant']}@{cell['temperature']}: support "
              f"{cell['mean_answer_support']:.3f}, pairs {cell['mean_pairs_per_passage']:.2f}"
              f" -> {cell['g1_verdict']} | {prose_txt}")
    print(f"[gate] G1 overall {overall_prev} -> {overall} (best "
          f"{best['variant']}@{best['temperature']})")
    if mismatches:
        print("[FAIL] the artifact disagrees with its own metrics rows; not regenerated "
              "cleanly", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
