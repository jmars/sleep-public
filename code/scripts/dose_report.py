#!/usr/bin/env python3
"""Supervised-token DOSE per arm, recomputed offline from the artifacts (review2 A2).

`train_sft` counts the label tokens the loss scored, but the E4 arms were run before that
instrument existed. This script recomputes the same quantity from what those runs left
behind -- the data file, the dream cache, and the config -- so the step-matched E5 arms
and the reused E4 references can be compared on one dose scale.

What it reproduces, exactly:

* cpt  -- examples = [passage], batch 4 -> 1 example per step (the batch is clipped to
  the example count), whole-text labels;
* dreams -- examples = every kept dream after the copy/short filter, batch 4, cycled.

The qa and recap arms SELECT dreams by gradient score (top-k + random), which needs the
model, so their dose is not recomputable offline: those arms carry their own in-training
count and this script reports that, with the reason stated rather than a guessed number.

The token count is taken through the same tokeniser and the same `token_batch` /
`_cycled_batch` code the training loop uses, so a divergence would be a real defect. When
run against an E5 arm (which has `summary["dose"]`), the two readings are compared and a
mismatch is reported and fails the script -- the offline instrument is thereby validated
against the in-training one rather than assumed to agree.

    python3 scripts/dose_report.py --runs e4_cpt e4_nodream e5_sleep_recap --out runs/e5_dose.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import torch  # noqa: E402

from squad_prep import read_jsonl  # noqa: E402

from qlora_sleep.config import load_config  # noqa: E402
from qlora_sleep.router_warmup import token_batch  # noqa: E402
from qlora_sleep.sft import _cycled_batch, prepare_cpt_examples, prepare_sft_examples  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def token_count(tokenizer, texts: list[str], max_len: int) -> int:
    _, attn, labels = token_batch(tokenizer, texts, max_len, "cpu")
    return int((labels != -100).sum())


def arm_dose(tokenizer, examples: list[str], steps: int, batch: int, max_len: int) -> int:
    total = 0
    for step in range(steps):
        batch_texts = _cycled_batch(examples, step, batch)
        total += token_count(tokenizer, batch_texts, max_len)
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data", default="data/squad_gate24.jsonl")
    ap.add_argument("--dreams-cache", default="runs/e4_dreams.pt")
    ap.add_argument("--config", default="single_n1")
    ap.add_argument("--out", default="runs/e5_dose.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["model"]["id"])
    passages = read_jsonl(args.data)["passages"]
    dreams = torch.load(args.dreams_cache, weights_only=False)["dreams"]
    steps = int(cfg["sft"]["steps"])
    batch = int(cfg["sft"]["batch_size"])
    max_len = int(cfg["sft"]["max_len"])
    max_containment = float(cfg["dream"]["max_containment"])
    ngram_n = int(cfg["dream"]["ngram_n"])

    report: dict[str, dict] = {}
    mismatches = []
    for run in args.runs:
        summary_path = Path("runs") / run / "summary.json"
        if not summary_path.exists():
            print(f"[skip] {run}: no summary.json")
            continue
        summary = json.loads(summary_path.read_text())
        mode = summary["sft_mode"]
        if mode in ("qa", "recap") and "dose" in summary:
            report[run] = {
                "sft_mode": mode,
                "steps": summary["dose"]["steps_mean"],
                "batch_size": batch,
                "source": "in-training counter (summary['dose'])",
                "offline_recompute": ("not possible for this arm: dream selection is a "
                                      "gradient pass over the model"),
                "n_passages": len(summary["per_passage"]),
                "supervised_tokens_mean": summary["dose"]["supervised_tokens_mean"],
                "supervised_tokens_min": summary["dose"]["supervised_tokens_min"],
                "supervised_tokens_max": summary["dose"]["supervised_tokens_max"],
            }
            print(f"[{run}] mode={mode} steps={summary['dose']['steps_mean']} "
                  f"dose mean={summary['dose']['supervised_tokens_mean']} "
                  f"(in-training counter)")
            continue
        per_passage = []
        for i, passage in enumerate(passages[: len(summary["per_passage"])]):
            if mode == "cpt":
                examples = prepare_cpt_examples(passage["context"])
            elif mode in ("dreams", "recap"):
                prepared = prepare_sft_examples(
                    list(dreams[i]), passage["context"],
                    max_containment=max_containment, n=ngram_n,
                )
                examples = prepared["examples"]
            else:
                examples = None
            if not examples:
                per_passage.append({"passage": i, "supervised_tokens": None})
                continue
            per_passage.append({
                "passage": i,
                "n_examples": len(examples),
                "supervised_tokens": arm_dose(tok, examples, steps, batch, max_len),
                "mean_words": sum(len(t.split()) for t in examples) / len(examples),
            })
        values = [p["supervised_tokens"] for p in per_passage if p["supervised_tokens"]]
        entry = {
            "sft_mode": mode,
            "steps": steps,
            "batch_size": batch,
            "source": ("in-training counter (summary['dose'])" if "dose" in summary
                       else "offline recompute from artifacts"),
            "n_passages": len(values),
            "supervised_tokens_mean": sum(values) / len(values) if values else None,
            "supervised_tokens_min": min(values) if values else None,
            "supervised_tokens_max": max(values) if values else None,
            "per_passage": per_passage,
        }
        if "dose" in summary and values:
            inrun = summary["dose"]["supervised_tokens_mean"]
            entry["in_training_mean"] = inrun
            entry["offline_mean"] = entry["supervised_tokens_mean"]
            entry["agrees"] = abs(inrun - entry["supervised_tokens_mean"]) <= 0.01 * max(inrun, 1)
            if not entry["agrees"]:
                mismatches.append({"run": run, "in_training": inrun,
                                   "offline": entry["supervised_tokens_mean"]})
        report[run] = entry
        print(f"[{run}] mode={mode} steps={steps} dose mean={entry['supervised_tokens_mean']} "
              f"range={entry['supervised_tokens_min']}-{entry['supervised_tokens_max']} "
              f"({entry['source']})")

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"[out] {args.out}")
    if mismatches:
        print(f"[FAIL] offline dose disagrees with the in-training counter: {mismatches}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
