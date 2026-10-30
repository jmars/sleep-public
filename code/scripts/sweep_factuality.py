#!/usr/bin/env python3
"""E1: dream FACTUALITY sweep (teacher-only) -- the instrument the M2 sweep never had.

The M2 literalism sweep measured whether dreams copy the passage. It never measured
whether the facts in them are the passage's facts, and that gap is the measured cause of
the M3 failure: of the (question, answer) pairs the SFT was trained on, only 53% had an
answer present in the wake passage, so roughly half of what the student stored was the
teacher's invention.

    python scripts/sweep_factuality.py --data data/squad_gate24.jsonl --n-passages 12 \
        --variants facts,recap --temperatures 0.9,0.7 --n-dreams 8 --run e1_factuality

Per cell (prompt variant x temperature), over `--n-passages` real passages:

* mean `answer_support_rate` -- fraction of extracted answers whose text occurs in C;
* mean usable pairs per passage -- the yield, because a grounded but empty dream stage
  cannot inject anything either;
* the 8-gram containment distribution, so factuality is read next to literalism
  (a variant that grounds by copying C is not a fix).

Decision rule (plan2 G1): PASS if mean support >= 0.70 AND mean pairs/passage >= 10;
INSUFFICIENT at 0.55-0.70; FAIL below 0.55 everywhere -- which escalates to the
orchestrator, because the next lever (few-shot teacher prompting) changes what the
teacher conditions on.

The teacher is loaded once and released at the end (a bf16 teacher and an NF4 student
cannot be co-resident on a small card); no student is built here, so this sweep is
teacher-only and costs one generation pass per cell.
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
    answer_support_rate,
    dream_report,
    extract_qa_from_dreams,
    generate_dreams,
    literalism_verdict,
)
from qlora_sleep.model import load_tokenizer  # noqa: E402
from qlora_sleep.teacher import load_teacher, release_teacher  # noqa: E402
from qlora_sleep.telemetry import Telemetry  # noqa: E402


def verdict(mean_support: float, mean_pairs: float, pass_support: float = 0.70,
            insufficient_support: float = 0.55, min_pairs: float = 10.0,
            min_pairs_measured: float = 2.0) -> str:
    """G1's decision, with an explicit NO_PAIRS outcome.

    `answer_support_rate` is computed over EXTRACTED question/answer pairs, and the
    extractor is question-shaped by construction. A prose variant ("list the facts") that
    emits no questions therefore yields no pairs and reads support 0.0 -- which is "this
    instrument cannot see this cell", NOT "the teacher cannot ground". Reporting it as
    FAIL would over-read the evidence in the direction that matters most (it would
    trigger the escalation branch for a measurement gap).
    """
    if mean_pairs < min_pairs_measured:
        return "NO_PAIRS"
    if mean_support >= pass_support and mean_pairs >= min_pairs:
        return "PASS"
    if mean_support < insufficient_support:
        return "FAIL"
    return "INSUFFICIENT"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="single_n1")
    ap.add_argument("--data", default="data/squad_gate24.jsonl")
    ap.add_argument("--n-passages", type=int, default=12)
    ap.add_argument("--n-dreams", type=int, default=8)
    ap.add_argument("--temperatures", default="0.9,0.7")
    ap.add_argument("--variants", default="facts,recap")
    ap.add_argument("--extra-cells", default="quiz@0.9",
                    help="paired control cell(s), variant@temperature, comma separated")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--ngram-n", type=int, default=8)
    ap.add_argument("--min-pairs", type=float, default=10.0)
    ap.add_argument("--pass-support", type=float, default=0.70)
    ap.add_argument("--insufficient-support", type=float, default=0.55)
    ap.add_argument("--run", default="e1_factuality")
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
    cells = [(v, float(t)) for t in args.temperatures.split(",") if t
             for v in args.variants.split(",") if v]
    for cell in args.extra_cells.split(","):
        if not cell:
            continue
        variant, _, temperature = cell.partition("@")
        pair = (variant, float(temperature or 0.9))
        if pair not in cells:
            cells.append(pair)
    n = args.ngram_n
    print(f"[data] {len(passages)} passages, {args.n_dreams} dreams/cell, cells={cells}", flush=True)
    tel.log("data", n_passages=len(passages), n_dreams=args.n_dreams,
            cells=[f"{v}@{t}" for v, t in cells])

    tok = load_tokenizer(cfg["model"]["id"])
    t = time.time()
    teacher = load_teacher(
        cfg["model"]["id"], device=args.device,
        attn_implementation=cfg["model"]["attn_implementation"],
    )
    load_s = time.time() - t

    grid: list[dict] = []
    t = time.time()
    for variant, temperature in cells:
        supports: list[float] = []
        pair_counts: list[int] = []
        pooled_pairs = 0
        pooled_supported = 0.0
        containments: list[float] = []
        per_passage: list[dict] = []
        samples: list[dict] = []
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
            pairs = extract_qa_from_dreams(dreams)
            support = answer_support_rate(pairs, passage["context"])
            report = dream_report(dreams, passage["context"], n=n)
            supports.append(support)
            pair_counts.append(len(pairs))
            pooled_pairs += len(pairs)
            pooled_supported += support * len(pairs)
            containments.extend(report["containment"])
            row = {
                "variant": variant,
                "temperature": temperature,
                "passage": i,
                "passage_id": passage.get("passage_id", ""),
                "n_pairs": len(pairs),
                "answer_support": support,
                "containment_median": report["containment_median"],
            }
            per_passage.append(row)
            tel.log("factuality_passage", **row)
            if len(samples) < 2:
                samples.append({"passage_id": row["passage_id"], "dreams": dreams,
                                "pairs": [list(p) for p in pairs]})
        containments_sorted = sorted(containments)
        mean_support = sum(supports) / max(len(supports), 1)
        mean_pairs = sum(pair_counts) / max(len(pair_counts), 1)
        pooled_support = pooled_supported / pooled_pairs if pooled_pairs else 0.0
        mid = (containments_sorted[len(containments_sorted) // 2]
               if containments_sorted else 0.0)
        cell = {
            "variant": variant,
            "temperature": temperature,
            "n_passages": len(passages),
            "mean_answer_support": mean_support,
            "pooled_answer_support": pooled_support,
            "mean_pairs_per_passage": mean_pairs,
            "min_pairs_per_passage": min(pair_counts) if pair_counts else 0,
            "total_pairs": pooled_pairs,
            "containment_median": mid,
            "containment_mean": (sum(containments) / max(len(containments), 1)),
            "literalism_verdict": literalism_verdict(
                mid, alarm=float(cfg["gates"]["literalism_alarm"]),
                insufficient=float(cfg["gates"]["literalism_insufficient"]),
            ),
            "g1_verdict": verdict(mean_support, mean_pairs,
                                  pass_support=args.pass_support,
                                  insufficient_support=args.insufficient_support,
                                  min_pairs=args.min_pairs),
            "per_passage": per_passage,
            "samples": samples,
        }
        grid.append(cell)
        tel.log("factuality_cell", **{k: v for k, v in cell.items()
                                      if k not in ("per_passage", "samples")})
        print(f"[{variant:6s} T={temperature:.2f}] support mean={mean_support:.3f} "
              f"pooled={pooled_support:.3f} pairs/passage={mean_pairs:.1f} "
              f"containment_median={mid:.3f} -> {cell['g1_verdict']}", flush=True)
    gen_s = time.time() - t
    release_teacher(teacher)

    order = {"PASS": 0, "INSUFFICIENT": 1, "NO_PAIRS": 2, "FAIL": 3}
    ranked = sorted(grid, key=lambda c: (order.get(c["g1_verdict"], 9),
                                         -c["mean_answer_support"], -c["mean_pairs_per_passage"]))
    best = ranked[0] if ranked else None
    # the sweep's own reading: PASS if any cell passes; FAIL only if a cell that
    # actually produced pairs came in under the floor; anything else is undecided
    if any(c["g1_verdict"] == "PASS" for c in grid):
        overall = "PASS"
    elif any(c["g1_verdict"] == "FAIL" for c in grid):
        overall = "FAIL"
    else:
        overall = "INSUFFICIENT"
    summary = {
        "run": args.run,
        "n_passages": len(passages),
        "n_dreams_per_cell": args.n_dreams,
        "grid": grid,
        "best_cell": None if best is None else {
            "variant": best["variant"], "temperature": best["temperature"],
            "mean_answer_support": best["mean_answer_support"],
            "mean_pairs_per_passage": best["mean_pairs_per_passage"],
            "g1_verdict": best["g1_verdict"],
        },
        # the G1 reading for the WHOLE sweep: a FAIL only if every cell fails, which is
        # the escalation branch (the teacher cannot ground at this scale)
        "g1_verdict": overall,
        "teacher_load_s": load_s,
        "dream_gen_s": gen_s,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    tel.log("factuality_summary", g1_verdict=overall,
            best_variant=None if best is None else best["variant"],
            best_temperature=None if best is None else best["temperature"],
            best_support=None if best is None else best["mean_answer_support"],
            best_pairs=None if best is None else best["mean_pairs_per_passage"])
    print(f"[gate] G1 = {overall} (best {best['variant']}@{best['temperature']} "
          f"support={best['mean_answer_support']:.3f} "
          f"pairs={best['mean_pairs_per_passage']:.1f})" if best else "[gate] G1 = FAIL (no cells)",
          flush=True)
    print(f"[done] {run_dir / 'summary.json'} wall={load_s + gen_s:.1f}s")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
