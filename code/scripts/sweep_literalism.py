#!/usr/bin/env python3
"""M2 gate instrument: dream literalism over real passages.

    python scripts/sweep_literalism.py --data data/squad_base50.jsonl \
        --n-passages 20 --n-dreams 3 --run m2_literalism

Reports the 8-gram containment distribution of teacher dreams against their wake
context, over a temperature x prompt-variant grid. Two things make this a real
instrument rather than a formality:

* the distribution is reported over REAL passages, so the alarm thresholds in
  `configs/base.yaml` are calibrated against something observed, not assumed;
* the grid must MOVE containment. If a 0.7 -> 1.1 temperature sweep and three
  prompt variants leave the median flat, the instrument is not measuring what it
  claims to and the M4 literalism-vs-gain correlation would be built on sand.

The teacher is loaded once and released at the end: on a 16GB card a bf16 teacher
and an NF4 student cannot be resident together.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

from squad_prep import read_jsonl  # noqa: E402

from qlora_sleep.config import load_config  # noqa: E402
from qlora_sleep.dream import (  # noqa: E402
    DreamConfig,
    dream_report,
    generate_dreams,
    literalism_verdict,
)
from qlora_sleep.model import load_tokenizer  # noqa: E402
from qlora_sleep.teacher import load_teacher, release_teacher  # noqa: E402
from qlora_sleep.telemetry import Telemetry  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="single_n1")
    ap.add_argument("--data", default="data/squad_base50.jsonl")
    ap.add_argument("--n-passages", type=int, default=20)
    ap.add_argument("--n-dreams", type=int, default=3)
    ap.add_argument("--temperatures", default="0.7,0.9,1.1")
    ap.add_argument("--variants", default="recap,quiz,facts")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--ngram-n", type=int, default=8)
    ap.add_argument("--run", default="m2_literalism")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_dir = Path("runs") / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    tel = Telemetry(run_dir / "metrics.jsonl")

    payload = read_jsonl(args.data)
    passages = payload["passages"][: args.n_passages]
    if not passages:
        raise SystemExit("no passages in --data")
    temperatures = [float(t) for t in args.temperatures.split(",") if t]
    variants = [v for v in args.variants.split(",") if v]
    n = args.ngram_n

    tok = load_tokenizer(cfg["model"]["id"])
    t = time.time()
    teacher = load_teacher(
        cfg["model"]["id"], device=args.device, attn_implementation=cfg["model"]["attn_implementation"]
    )
    load_s = time.time() - t

    grid: list[dict] = []
    t = time.time()
    for variant in variants:
        for temperature in temperatures:
            containments: list[float] = []
            duplicate_rates: list[float] = []
            dream_lists: list[list[str]] = []
            for i, passage in enumerate(passages):
                dcfg = DreamConfig(
                    n_dreams=args.n_dreams,
                    temperature=temperature,
                    top_p=float(cfg["dream"]["top_p"]),
                    max_new_tokens=args.max_new_tokens,
                    prompt_variant=variant,
                    random_expert=bool(cfg["dream"]["random_expert"]),
                    seed=args.seed + i,
                )
                dreams = generate_dreams(teacher, tok, passage["context"], dcfg)
                report = dream_report(dreams, passage["context"], n=n)
                containments.extend(report["containment"])
                duplicate_rates.append(report["duplicate_rate"])
                dream_lists.append(dreams)
            containments_sorted = sorted(containments)
            mid = containments_sorted[len(containments_sorted) // 2] if containments_sorted else 0.0
            row = {
                "variant": variant,
                "temperature": temperature,
                "n_dreams": len(containments),
                "containment_median": mid,
                "containment_mean": sum(containments) / max(len(containments), 1),
                "containment_p90": containments_sorted[int(0.9 * (len(containments_sorted) - 1))] if containments_sorted else 0.0,
                "duplicate_rate_mean": sum(duplicate_rates) / max(len(duplicate_rates), 1),
                "verdict": literalism_verdict(
                    mid,
                    alarm=float(cfg["gates"]["literalism_alarm"]),
                    insufficient=float(cfg["gates"]["literalism_insufficient"]),
                ),
                "samples": [
                    {"passage_id": p.get("passage_id", ""), "dreams": d}
                    for p, d in list(zip(passages, dream_lists))[:2]
                ],
            }
            grid.append(row)
            tel.log("literalism_cell", **row)
            print(
                f"[{variant:6s} T={temperature:.2f}] median={row['containment_median']:.3f} "
                f"mean={row['containment_mean']:.3f} p90={row['containment_p90']:.3f} "
                f"dup={row['duplicate_rate_mean']:.2f} verdict={row['verdict']}"
            )
    gen_s = time.time() - t
    release_teacher(teacher)

    medians = [row["containment_median"] for row in grid]
    spread = max(medians) - min(medians)
    by_variant: dict[str, float] = {}
    for row in grid:
        by_variant.setdefault(row["variant"], row["containment_median"])
    variant_spread = max(by_variant.values()) - min(by_variant.values())
    summary = {
        "run": args.run,
        "n_passages": len(passages),
        "n_dreams_per_cell": args.n_dreams,
        "temperatures": temperatures,
        "variants": variants,
        "grid": grid,
        "median_spread_across_grid": spread,
        # the instrument must respond to its knobs; a flat grid means it is not
        # measuring literalism at all
        "instrument_moves": spread > 0.02 or variant_spread > 0.02,
        "teacher_load_s": load_s,
        "dream_gen_s": gen_s,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    tel.log("literalism_summary", median_spread_across_grid=spread, instrument_moves=summary["instrument_moves"])
    print(f"[gate] median spread across grid = {spread:.3f} -> instrument_moves={summary['instrument_moves']}")
    print(f"[done] {run_dir / 'summary.json'} wall={load_s + gen_s:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
