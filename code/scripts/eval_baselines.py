#!/usr/bin/env python3
"""M0 reference frame: base model, and SFT-on-C (the no-dreaming baseline).

    python scripts/eval_baselines.py --mode base --data data/squad_base50.jsonl
    python scripts/eval_baselines.py --mode sft-on-c --data data/squad_n1.jsonl

Gate reading (plan, M0):
  * base no-context score measured; apply the RE-BASING RULE against the paper's 31.9.
  * the with-context leak probe must clear the configured delta, else the eval leaks.
  * SFT-on-C must show a SMALL delta over base on the same questions. A large delta
    (> 15 points) means context is reaching the no-context path -- fix the eval first.
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
from qlora_sleep.eval_squad import EvalExample, leak_probe, model_generate_fn  # noqa: E402
from qlora_sleep.eval_squad import evaluate  # noqa: E402
from qlora_sleep.model import load_student, load_tokenizer, set_trainable  # noqa: E402
from qlora_sleep.sft import SFTConfig, train_sft  # noqa: E402
from qlora_sleep.telemetry import Telemetry  # noqa: E402


def passages_to_examples(passages: list[dict]) -> list[EvalExample]:
    out = []
    for passage in passages:
        for qa in passage["questions"]:
            out.append(
                EvalExample(
                    question=qa["question"],
                    answers=qa["answers"],
                    context=passage["context"],
                    passage_id=passage.get("passage_id", ""),
                )
            )
    return out


def apply_baseline_lora(model, r: int, alpha: int, targets: list[str]):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    config = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=0.0, bias="none", target_modules=targets,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    set_trainable(model, lora=True, routers=False, alphas=False)
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="single_n1")
    ap.add_argument("--mode", choices=["base", "sft-on-c"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit-passages", type=int, default=0)
    ap.add_argument("--limit-questions", type=int, default=0)
    ap.add_argument("--leak-limit", type=int, default=40)
    ap.add_argument("--run", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_name = args.run or f"m0_{args.mode}"
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    tel = Telemetry(run_dir / "metrics.jsonl")
    t0 = time.time()

    payload = read_jsonl(args.data)
    passages = payload["passages"]
    if args.limit_passages:
        passages = passages[: args.limit_passages]
    examples = passages_to_examples(passages)
    if args.limit_questions:
        examples = examples[: args.limit_questions]
    tel.log("data", n_passages=len(passages), n_questions=len(examples), source=str(args.data))
    print(f"[data] {len(passages)} passages, {len(examples)} questions")

    tok = load_tokenizer(cfg["model"]["id"])
    quant = cfg["model"]["quantisation"]
    model = load_student(
        cfg["model"]["id"],
        quantise=bool(quant["load_in_4bit"]),
        device=args.device,
        skip_modules=tuple(quant["skip_modules"]),
    )
    tel.log("load", mode=args.mode, seconds=time.time() - t0)

    if args.mode == "sft-on-c":
        # the no-dreaming baseline: plain LoRA on attention projections, trained on
        # the wake passage itself (continued-pretraining analogue)
        model = apply_baseline_lora(
            model, r=cfg["lora"]["r"], alpha=cfg["lora"]["alpha"], targets=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        texts = [p["context"] for p in passages]
        sft_cfg = SFTConfig(
            steps=int(cfg["sft"]["steps"]),
            lr=float(cfg["sft"]["lr"]),
            batch_size=1,
            max_len=int(cfg["sft"]["max_len"]),
            train_routers=False,
        )
        print(f"[sft-on-c] training {sft_cfg.steps} steps on {len(texts)} passage(s)")
        t1 = time.time()
        result = train_sft(model, tok, texts, sft_cfg, context=None, log=tel.log)
        tel.log("sft_on_c", seconds=time.time() - t1, first_loss=result["history"][0]["loss"],
                last_loss=result["final"]["loss"])

    gen = model_generate_fn(model, tok, max_new_tokens=int(cfg["eval"]["max_new_tokens"]),
                            batch_size=int(cfg["eval"]["batch_size"]))
    t2 = time.time()
    without = evaluate(gen, examples, "no_context")
    # predictions, not just the score: the pooled gate is bootstrapped over PER-QUESTION
    # outcomes, which needs the individual answers, and a 109-question set at ~12 EM
    # has a sampling spread large enough that a bare mean cannot decide a 2-pt gate.
    tel.log("eval", mode=args.mode, eval_mode="no_context", n=without["n"], em=without["em"], f1=without["f1"],
            seconds=time.time() - t2, predictions=without["predictions"],
            golds=[ex.answers[:2] for ex in examples])
    print(f"[eval] no-context  EM={without['em']:.2f}  F1={without['f1']:.2f}  n={without['n']}")

    leak_examples = examples[: args.leak_limit]
    t3 = time.time()
    probe = leak_probe(gen, leak_examples, min_delta=float(cfg["gates"]["leak_min_delta"]))
    tel.log("leak_probe", n=len(leak_examples), seconds=time.time() - t3, **probe)
    print(f"[leak] with-context EM={probe['with_context_em']:.2f} vs no-context "
          f"{probe['no_context_em']:.2f}  delta={probe['delta']:.2f}  leak={probe['leak']}")

    summary = {
        "mode": args.mode,
        "run": run_name,
        "no_context_em": without["em"],
        "no_context_f1": without["f1"],
        "n_questions": without["n"],
        "leak": probe,
        "wall_clock_s": time.time() - t0,
    }
    if args.mode == "base":
        paper = float(cfg["gates"]["paper_base"])
        tol = float(cfg["gates"]["rebase_tolerance"])
        summary["paper_base"] = paper
        summary["rebase_gap"] = without["em"] - paper
        summary["absolute_gate_demoted"] = abs(without["em"] - paper) > tol
        print(f"[rebase] base={without['em']:.2f} paper={paper} gap={summary['rebase_gap']:+.2f} "
              f"absolute_gate_demoted={summary['absolute_gate_demoted']}")
    if args.mode == "sft-on-c":
        paper = float(cfg["gates"]["paper_no_dreaming_single"])
        summary["paper_no_dreaming"] = paper
        print(f"[sft-on-c] paper no-dreaming={paper} (kind check: small delta over base expected)")

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    tel.log("summary", **{k: v for k, v in summary.items() if not isinstance(v, dict)})
    print(f"[done] {run_dir/'summary.json'}  wall={summary['wall_clock_s']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
