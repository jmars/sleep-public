#!/usr/bin/env python3
"""One sleep cycle on one passage (M3), or the warmup-only gate (M1).

    # M1 gate: router warmup alone must stay within 2 points of base
    python scripts/run_cycle.py --mode warmup-only --data data/squad_n1.jsonl \
        --base-em <m0 base em>

    # M3: the full single-cycle pipeline
    python scripts/run_cycle.py --mode full --data data/squad_n1.jsonl --base-em <m0 base em>

Stage order (plan, M3): teacher dreams -> gradient scoring -> top-k+b -> literalism
and copy-vs-store filters -> router warmup -> QLoRA SFT on C-stripped dreams ->
no-context eval + leak probe. The teacher is released before the student is built:
a bf16 teacher (~14GB) and an NF4 student (~4GB) do not co-reside on a 16GB card.
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
    literalism_verdict,
    score_dreams,
    select_dreams,
)
from qlora_sleep.eval_squad import (  # noqa: E402
    EvalExample,
    build_prompt,
    evaluate,
    format_probe,
    leak_probe,
    model_generate_fn,
)
from qlora_sleep.model import build_student, load_tokenizer, new_expert_start  # noqa: E402
from qlora_sleep.router_warmup import (  # noqa: E402
    WarmupConfig,
    routing_share,
    run_router_warmup,
)
from qlora_sleep.sft import SFTConfig, prepare_sft_examples, train_sft  # noqa: E402
from qlora_sleep.teacher import dream_teacher_stage  # noqa: E402
from qlora_sleep.telemetry import Telemetry  # noqa: E402


def examples_from(passages: list[dict]) -> list[EvalExample]:
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="single_n1")
    ap.add_argument("--mode", choices=["full", "warmup-only"], default="full")
    ap.add_argument("--data", required=True)
    ap.add_argument("--base-em", type=float, default=None, help="M0 base no-context EM on the same questions")
    ap.add_argument("--run", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--leak-limit", type=int, default=40)
    ap.add_argument("--cycles", type=int, default=1, help="sleep cycles over the same passage")
    ap.add_argument("--probe", action="store_true", help="log format/memorisation probes")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_name = args.run or ("m3_cycle" if args.mode == "full" else "m1_warmup")
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    tel = Telemetry(run_dir / "metrics.jsonl")
    timings: dict[str, float] = {}
    t_start = time.time()

    payload = read_jsonl(args.data)
    passages = payload["passages"]
    if not passages:
        raise SystemExit("no passages in --data")
    passage = passages[0]
    context = passage["context"]
    examples = examples_from([passage])
    tel.log("data", passage_id=passage.get("passage_id", ""), n_questions=len(examples),
            context_words=len(context.split()))
    print(f"[data] passage={passage.get('passage_id','')} questions={len(examples)} words={len(context.split())}")

    tok = load_tokenizer(cfg["model"]["id"])

    # ---------------- teacher: load, measure expert load, dream, release ------
    t = time.time()
    load = None
    dreams_by_cycle: list[list[str]] = [[] for _ in range(args.cycles)]
    if args.mode == "full":
        # The teacher is a FROZEN base snapshot for every cycle (v1 design), so all
        # cycles' dreams can be generated in one teacher residency -- which matters on
        # a 16GB card where the teacher and student cannot co-reside. Each cycle gets
        # its own seed: with a fixed seed the frozen teacher regenerates the identical
        # dreams every cycle, so "3 cycles" would be the same 6 texts trained three
        # times over, not three independent consolidation passes.
        dream_cfgs = [
            DreamConfig(
                n_dreams=int(cfg["dream"]["n_dreams"]),
                temperature=float(cfg["dream"]["temperature"]),
                top_p=float(cfg["dream"]["top_p"]),
                max_new_tokens=int(cfg["dream"]["max_new_tokens"]),
                prompt_variant=str(cfg["dream"]["prompt_variant"]),
                random_expert=bool(cfg["dream"]["random_expert"]),
                seed=args.seed + 1000 * c,
            )
            for c in range(args.cycles)
        ]
        # the teacher lives and dies inside this call: a bf16 teacher (~14GB) and the
        # NF4 student cannot be resident together on a 16GB card
        stage = dream_teacher_stage(
            cfg["model"]["id"],
            tok,
            [context] * args.cycles,
            dream_cfgs,
            device=args.device,
            attn_implementation=cfg["model"]["attn_implementation"],
            max_len=int(cfg["sft"]["max_len"]),
        )
        timings["teacher_load_s"] = 0.0
        timings["dream_gen_s"] = time.time() - t
        load = stage["expert_load"]
        dreams_by_cycle = stage["dreams"]
        tel.log("expert_load", shape=list(load.shape),
                top_per_layer=[int(x) for x in load.topk(4, dim=1).indices.flatten()[:8]])
        for c, dreams_c in enumerate(dreams_by_cycle):
            tel.log("dreams", cycle=c, n=len(dreams_c), variant=dream_cfgs[c].prompt_variant,
                    temperature=dream_cfgs[c].temperature,
                    random_expert=dream_cfgs[c].random_expert, dreams=dreams_c)
            print(f"[dream] cycle {c}: {len(dreams_c)} dreams in {timings['dream_gen_s']:.1f}s")
    else:
        timings["teacher_load_s"] = 0.0
        timings["dream_gen_s"] = 0.0

    # ---------------- student -------------------------------------------------
    t = time.time()
    model = build_student(
        cfg["model"]["id"],
        k_new=int(cfg["experts"]["k_new"]),
        lora_r=int(cfg["lora"]["r"]),
        lora_alpha=int(cfg["lora"]["alpha"]),
        lora_dropout=float(cfg["lora"]["dropout"]),
        init_noise_frac=float(cfg["experts"]["init_noise_frac"]),
        mask_value=float(cfg["experts"]["mask_value"]),
        load=load,
        quantise=bool(cfg["model"]["quantisation"]["load_in_4bit"]),
        device=args.device,
    )
    timings["student_build_s"] = time.time() - t
    start = new_expert_start(model)
    tel.log("student", new_expert_start=start, k_new=int(cfg["experts"]["k_new"]),
            trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(f"[student] built in {timings['student_build_s']:.1f}s, new experts from {start}")

    # Eval at each stage. A single end-of-cycle number cannot say WHICH stage broke the
    # model, and "the cycle lost 20 points" is not actionable. Each stage therefore
    # gets its own no-context reading, cheapest first.
    gen = model_generate_fn(model, tok, max_new_tokens=int(cfg["eval"]["max_new_tokens"]),
                            batch_size=int(cfg["eval"]["batch_size"]))
    stage_evals: dict[str, dict] = {}

    def eval_stage(name: str) -> dict:
        result = evaluate(gen, examples, "no_context")
        stage_evals[name] = result
        tel.log("eval", eval_mode="no_context", stage_name=name, n=result["n"],
                em=result["em"], f1=result["f1"],
                # keep the strings, not just the score: at n=5 each question is worth
                # 20 points, so "EM dropped to 0" says nothing until you read what the
                # model actually emitted
                predictions=result["predictions"],
                golds=[ex.answers[:2] for ex in examples])
        print(f"[eval:{name}] no-context EM={result['em']:.2f} F1={result['f1']:.2f} "
              f"preds={result['predictions'][:2]}")
        return result

    # At this point the surgery is done but the new experts are still masked and the
    # LoRA B matrices are zero, so this reading must equal the M0 base reading. If it
    # does not, the surgery changed base behaviour and every later number is suspect.
    eval_stage("post_surgery")

    # ---------------- the sleep cycle(s) -------------------------------------
    selected: list[str] = []
    for cycle in range(args.cycles):
        dreams = dreams_by_cycle[cycle]
        # ---------------- scoring / selection / literalism --------------------
        selected = []
        if args.mode == "full":
            t = time.time()
            scores = score_dreams(model, tok, dreams, max_len=int(cfg["sft"]["max_len"]))
            timings[f"score_s_{cycle}"] = time.time() - t
            keep = select_dreams(scores, int(cfg["dream"]["top_k_keep"]),
                                 int(cfg["dream"]["random_keep"]), seed=args.seed + cycle)
            tel.log("scoring", cycle=cycle, scores=scores, selected=keep)
            print(f"[score] cycle {cycle}: kept {len(keep)}/{len(dreams)} dreams")

            report = dream_report([dreams[i] for i in keep], context, n=int(cfg["dream"]["ngram_n"]))
            verdict = literalism_verdict(
                report["containment_median"],
                alarm=float(cfg["gates"]["literalism_alarm"]),
                insufficient=float(cfg["gates"]["literalism_insufficient"]),
            )
            tel.log("literalism", cycle=cycle, verdict=verdict, **report)
            print(f"[literalism] cycle {cycle}: median containment="
                  f"{report['containment_median']:.3f} verdict={verdict}")

            prepared = prepare_sft_examples(
                [dreams[i] for i in keep],
                context,
                max_containment=float(cfg["dream"]["max_containment"]),
                n=int(cfg["dream"]["ngram_n"]),
            )
            selected = prepared["examples"]
            if prepared["fallback"]:
                # nothing survived: keep the least-copying dreams rather than abort, and say so
                scored = sorted(zip(prepared["containment"], [dreams[i] for i in keep]))
                selected = [text for _, text in scored[:2]]
                tel.log("c_strip_fallback", cycle=cycle, n_used=len(selected))
                print("[c-strip] FALLBACK: every dream exceeded the containment threshold")
            tel.log("c_strip", cycle=cycle, kept=len(selected), dropped=len(prepared["dropped"]),
                    short_dropped=prepared["short_dropped"],
                    containments=prepared["containment"])
            print(f"[c-strip] cycle {cycle}: {len(selected)} training examples, "
                  f"{len(prepared['dropped'])} dropped as copies")
            # how much of the dream content is the passage's content at all -- a teacher
            # that hallucinates cannot inject real facts, whatever the SFT does
            pairs_all = extract_qa_from_dreams([dreams[i] for i in keep])
            tel.log("dream_facts", cycle=cycle, n_pairs=len(pairs_all),
                    answer_support=answer_support_rate(pairs_all, context))
            print(f"[facts] cycle {cycle}: {len(pairs_all)} pairs, "
                  f"answer support {answer_support_rate(pairs_all, context):.2f}")

        # ---------------- router warmup --------------------------------------
        # Warmup is a surgery-time intervention: it exists to escape the masked
        # zero-routing trap that only exists when the new experts are first added.
        # Re-annealing it every cycle would re-force experts the router has already
        # learned to select on merit, so it runs on the first cycle only.
        if cycle == 0:
            warm_texts = selected or [context]
            warm_cfg = WarmupConfig(
                steps=int(cfg["warmup"]["steps"]),
                forced_slots=int(cfg["warmup"]["forced_slots"]),
                lr=float(cfg["warmup"]["lr"]),
                max_len=int(cfg["sft"]["max_len"]),
                seed=args.seed,
            )
            t = time.time()
            warm = run_router_warmup(model, tok, warm_texts, warm_cfg, log=tel.log)
            timings["warmup_s"] = time.time() - t
            print(f"[warmup] {warm_cfg.steps} steps, final share={warm['final'].get('new_share'):.3f}, "
                  f"loss={warm['final'].get('loss'):.3f}")
            eval_stage("post_warmup")

        # ---------------- SFT ------------------------------------------------
        if args.mode == "full":
            sft_cfg = SFTConfig(
                steps=int(cfg["sft"]["steps"]),
                lr=float(cfg["sft"]["lr"]),
                batch_size=int(cfg["sft"]["batch_size"]),
                max_len=int(cfg["sft"]["max_len"]),
                max_containment=float(cfg["sft"]["max_containment"]),
                ngram_n=int(cfg["sft"]["ngram_n"]),
                seed=args.seed + cycle,
            )
            t = time.time()
            sft = train_sft(model, tok, selected, sft_cfg, context=context, log=tel.log)
            timings[f"sft_s_{cycle}"] = time.time() - t
            print(f"[sft] cycle {cycle}: {sft_cfg.steps} steps, loss "
                  f"{sft['history'][0]['loss']:.3f} -> {sft['final']['loss']:.3f}")

        eval_stage(f"cycle{cycle + 1}")

    # ---------------- telemetry + eval ---------------------------------------
    share = routing_share(model, tok, [context], max_len=int(cfg["sft"]["max_len"]))
    tel.log("routing_after", aggregate_new_share=share["aggregate_new_share"],
            per_layer={k: v["new_share"] for k, v in share["layers"].items()})
    print(f"[routing] new-expert share after training={share['aggregate_new_share']:.3f}")

    t = time.time()
    without = eval_stage("post_cycle")
    stage_evals["post_cycle"] = without
    timings["eval_s"] = time.time() - t
    tel.log("eval", eval_mode="no_context", n=without["n"], em=without["em"], f1=without["f1"])
    print(f"[eval] no-context EM={without['em']:.2f} F1={without['f1']:.2f} n={without['n']}")

    probes: dict = {}
    if args.probe:
        # Three readings that separate the failure modes the gate cannot distinguish.
        # 1. memorisation: if the SFT loss on the training dreams is still high, the
        #    student never stored the dreams and nothing downstream can work.
        # 2. format: same questions, three prompt shapes. Answering in the training
        #    shape but not in the gate's shape is a format defect.
        import torch

        model.eval()
        dream_losses: list[float] = []
        for text in selected:
            enc = tok(text, return_tensors="pt", truncation=True,
                      max_length=int(cfg["sft"]["max_len"])).to(args.device)
            with torch.no_grad():
                dream_losses.append(
                    float(model(input_ids=enc["input_ids"], labels=enc["input_ids"],
                                use_cache=False).loss)
                )
        probes["memorisation"] = {
            "n": len(dream_losses),
            "mean_loss": sum(dream_losses) / len(dream_losses) if dream_losses else None,
            "losses": dream_losses,
        }
        tel.log("memorisation", **probes["memorisation"])
        # 3. routing: the new experts are selected on dream text, but are they
        #    selected on the GATE's prompt? If the router only learned the dream
        #    distribution, the trained experts sit out the evaluation.
        eval_prompts = [build_prompt(ex.question, None) for ex in examples]
        probes["routing"] = {
            "on_context": share["aggregate_new_share"],
            "on_eval_prompts": routing_share(
                model, tok, eval_prompts, max_len=int(cfg["sft"]["max_len"])
            )["aggregate_new_share"],
            "on_dreams": routing_share(
                model, tok, selected[:2], max_len=int(cfg["sft"]["max_len"])
            )["aggregate_new_share"] if selected else None,
        }
        tel.log("routing_probe", **probes["routing"])
        probes["format"] = format_probe(gen, examples)
        for name, reading in probes["format"].items():
            tel.log("format_probe", template=name, **reading)
        routing_txt = ", ".join(
            f"{k}={v:.3f}" for k, v in probes["routing"].items() if v is not None
        )
        mean_loss = probes["memorisation"]["mean_loss"]
        print(f"[probe] routing share: {routing_txt}; memorisation mean loss="
              f"{'n/a' if mean_loss is None else f'{mean_loss:.3f}'}; "
              + " ".join(f"{k}:EM={v['em']:.0f}" for k, v in probes["format"].items()))

    probe = leak_probe(gen, examples[: args.leak_limit], min_delta=float(cfg["gates"]["leak_min_delta"]))
    tel.log("leak_probe", **probe)
    print(f"[leak] with-context={probe['with_context_em']:.2f} delta={probe['delta']:.2f} leak={probe['leak']}")

    timings["total_s"] = time.time() - t_start
    summary = {
        "run": run_name,
        "config": args.config,
        "mode": args.mode,
        "passage_id": passage.get("passage_id", ""),
        "n_questions": without["n"],
        "no_context_em": without["em"],
        "no_context_f1": without["f1"],
        "base_em": args.base_em,
        "cycles": args.cycles,
        "new_expert_share": share["aggregate_new_share"],
        "leak": probe,
        "stage_evals": {k: {"em": v["em"], "f1": v["f1"], "n": v["n"]} for k, v in stage_evals.items()},
        "timings": timings,
    }
    if probes:
        summary["probes"] = probes
    gates = {}
    if args.base_em is not None:
        gain = without["em"] - args.base_em
        summary["gain_over_base"] = gain
        if args.mode == "full":
            gates["single_passage_gain"] = gain >= float(cfg["gates"]["single_passage_min_gain"])
        else:
            gates["warmup_within_2pts"] = abs(gain) <= 2.0
        print(f"[gate] gain over base = {gain:+.2f} pts -> {gates}")
    gates["no_leak"] = not probe["leak"]
    summary["gates"] = gates
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    tel.log("summary", **{k: v for k, v in summary.items() if not isinstance(v, dict)})
    print(f"[done] {run_dir/'summary.json'} wall={timings['total_s']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
