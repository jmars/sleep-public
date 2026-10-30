#!/usr/bin/env python3
"""E2: warmup-arm selection -- W1 (forced inclusion only) vs W2 (contrastive).

The measured defect both arms exist to fix: after warmup the new bank was selected MORE
on held-out text than on the text the cycle trained on (contrast -0.100 / -0.032), i.e.
the router had learned to over-attract to the new bank on everything and nothing
distinguished storage-worthy inputs.

W1 = the current forced-inclusion warmup, then the router bias is bisected into a
held-out share band and the routers are frozen.
W2 = forced inclusion plus a differentiable penalty on the new bank's probability MASS
above `share_target` on held-out distractor batches, then the same bias bisection.

Mechanistic difference the selection rests on: the bias is tuned on held-out text only,
so IF W2 learned content selectivity, the trained-text share survives the bias while the
held-out share is capped; W1 caps everywhere equally.

    python scripts/warmup_arms.py --arms w1,w2 --prompt-variant facts --temperature 0.9 \
        --run e2_arms

Reads three checkpoints (post_surgery, post_warmup, post_tune) x four text sets, with the
full 68-expert count vector logged so the top-4 BASE control is recomputable.

G2 per arm: (a) held-out share in band; (b) W2 only: trained-prompt share minus held-out
prompt share >= +0.02; (c) held-out EM drop vs own post-surgery <= 2 pts.
Decision: W2 if a+b+c; W1 if a+c without b; band unreachable in both -> STOP.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import torch  # noqa: E402

from squad_prep import read_jsonl  # noqa: E402

from qlora_sleep.config import load_config  # noqa: E402
from qlora_sleep.dream import (  # noqa: E402
    DreamConfig,
    answer_support_rate,
    extract_qa_from_dreams,
    generate_dreams,
    select_dreams,
)
from qlora_sleep.eval_squad import (  # noqa: E402
    EvalExample,
    build_prompt,
    evaluate,
    model_generate_fn,
)
from qlora_sleep.model import build_student, load_tokenizer  # noqa: E402
from qlora_sleep.router_warmup import (  # noqa: E402
    ShareBandUnreachable,
    WarmupConfig,
    routing_share,
    run_router_warmup,
    tune_new_share,
)
from qlora_sleep.sft import QA_TEMPLATE  # noqa: E402
from qlora_sleep.teacher import load_teacher, release_teacher  # noqa: E402
from qlora_sleep.telemetry import Telemetry  # noqa: E402


def share_row(model, tok, texts: list[str], max_len: int, chunk: int = 16) -> dict:
    counts = None
    start = None
    for i in range(0, len(texts), chunk):
        rep = routing_share(model, tok, texts[i : i + chunk], max_len=max_len)
        start = rep["new_expert_start"]
        c = torch.tensor(rep["counts"], dtype=torch.float64)
        counts = c if counts is None else counts + c
    total = float(counts.sum())
    new = counts[start:]
    base = counts[:start]
    return {
        "n_texts": len(texts),
        "new_share": float(new.sum() / total),
        "uniform_new_share": float(new.numel() / counts.numel()),
        "top4_base_share": float(base.topk(min(4, base.numel())).values.sum() / total),
        "max_base_share": float(base.max() / total),
        "top4_new_share": float(new.topk(min(4, new.numel())).values.sum() / total),
        "counts": [float(v) for v in counts.tolist()],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="single_n1")
    ap.add_argument("--train-data", default="data/squad_n1.jsonl")
    ap.add_argument("--heldout-data", default="data/squad_base50.jsonl")
    ap.add_argument("--heldout-passages", type=int, default=20)
    ap.add_argument("--tune-passages", default="20:28")
    ap.add_argument("--arms", default="w1,w2")
    ap.add_argument("--prompt-variant", default="facts")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--share-band", default="0.05,0.09")
    ap.add_argument("--contrast-min", type=float, default=0.02)
    ap.add_argument("--heldout-max-drop", type=float, default=2.0)
    ap.add_argument("--run", default="e2_arms")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    max_len = int(cfg["sft"]["max_len"])
    band = tuple(float(v) for v in args.share_band.split(","))
    run_dir = Path("runs") / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    tel = Telemetry(run_dir / "metrics.jsonl")
    arms = [a for a in args.arms.split(",") if a]
    t_start = time.time()

    train_passage = read_jsonl(args.train_data)["passages"][0]
    context = train_passage["context"]
    train_examples = [
        EvalExample(question=q["question"], answers=q["answers"], context=context,
                    passage_id=train_passage.get("passage_id", ""))
        for q in train_passage["questions"]
    ]
    heldout = read_jsonl(args.heldout_data)["passages"]
    read_passages = [p for p in heldout[: args.heldout_passages]
                     if p.get("passage_id") != train_passage.get("passage_id")]
    tune_start, _, tune_end = args.tune_passages.partition(":")
    tune_passages = heldout[int(tune_start) : int(tune_end or len(heldout))]
    heldout_examples = [
        EvalExample(question=q["question"], answers=q["answers"], context=p["context"],
                    passage_id=p.get("passage_id", ""))
        for p in read_passages for q in p["questions"]
    ]
    tune_texts = [p["context"] for p in tune_passages] + [
        build_prompt(q["question"]) for p in tune_passages for q in p["questions"]
    ]
    text_sets = {
        "trained_context": [context],
        "trained_prompts": [build_prompt(ex.question) for ex in train_examples],
        "heldout_contexts": [p["context"] for p in read_passages],
        "heldout_prompts": [build_prompt(ex.question) for ex in heldout_examples],
    }
    print(f"[data] trained {len(train_examples)} q; held-out {len(read_passages)} passages / "
          f"{len(heldout_examples)} q; bias-tune on {len(tune_texts)} texts", flush=True)
    tel.log("data", n_train_q=len(train_examples), n_heldout_q=len(heldout_examples),
            n_tune_texts=len(tune_texts), arms=arms, prompt_variant=args.prompt_variant,
            temperature=args.temperature, share_band=list(band))

    # ---------------- teacher: one residency, the dreams all arms warm on -----
    tok = load_tokenizer(cfg["model"]["id"])
    t = time.time()
    teacher = load_teacher(cfg["model"]["id"], device=args.device,
                           attn_implementation=cfg["model"]["attn_implementation"])
    dreams = generate_dreams(
        teacher, tok, context,
        DreamConfig(n_dreams=int(cfg["dream"]["n_dreams"]), temperature=args.temperature,
                    top_p=float(cfg["dream"]["top_p"]),
                    max_new_tokens=int(cfg["dream"]["max_new_tokens"]),
                    prompt_variant=args.prompt_variant,
                    random_expert=bool(cfg["dream"]["random_expert"]), seed=args.seed),
    )
    release_teacher(teacher)
    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    teacher_s = time.time() - t

    keep = select_dreams([0.0] * len(dreams), int(cfg["dream"]["top_k_keep"]),
                         int(cfg["dream"]["random_keep"]), seed=args.seed)
    pairs = extract_qa_from_dreams([dreams[i] for i in keep])
    warm_texts = [QA_TEMPLATE.format(question=q, answer=a) for q, a in pairs]
    support = answer_support_rate(pairs, context)
    print(f"[dream] {len(dreams)} dreams -> {len(pairs)} pairs "
          f"(support {support:.2f}), teacher phase {teacher_s:.0f}s", flush=True)
    tel.log("dreams", dreams=dreams, n_pairs=len(pairs), answer_support=support,
            n_warm_texts=len(warm_texts))

    summary: dict = {"run": args.run, "prompt_variant": args.prompt_variant,
                     "temperature": args.temperature, "share_band": list(band),
                     "n_heldout_questions": len(heldout_examples), "arms": {}}

    for arm in arms:
        t_arm = time.time()
        model = build_student(
            cfg["model"]["id"], k_new=int(cfg["experts"]["k_new"]),
            lora_r=int(cfg["lora"]["r"]), lora_alpha=int(cfg["lora"]["alpha"]),
            lora_dropout=float(cfg["lora"]["dropout"]),
            init_noise_frac=float(cfg["experts"]["init_noise_frac"]),
            mask_value=float(cfg["experts"]["mask_value"]), load=None,
            quantise=bool(cfg["model"]["quantisation"]["load_in_4bit"]), device=args.device,
        )
        gen = model_generate_fn(model, tok, max_new_tokens=int(cfg["eval"]["max_new_tokens"]),
                                batch_size=int(cfg["eval"]["batch_size"]))
        arm_rec: dict = {"checkpoints": {}}

        def checkpoint(name: str) -> dict:
            t_em = evaluate(gen, train_examples, "no_context")
            h_em = evaluate(gen, heldout_examples, "no_context")
            shares = {key: share_row(model, tok, texts, max_len) for key, texts in text_sets.items()}
            rec = {
                "trained_em": t_em["em"], "trained_f1": t_em["f1"],
                "heldout_em": h_em["em"], "heldout_f1": h_em["f1"],
                "shares": shares,
                "contrast_prompts": (shares["trained_prompts"]["new_share"]
                                     - shares["heldout_prompts"]["new_share"]),
                "contrast_contexts": (shares["trained_context"]["new_share"]
                                      - shares["heldout_contexts"]["new_share"]),
            }
            arm_rec["checkpoints"][name] = rec
            tel.log("checkpoint", arm=arm, name=name, trained_em=rec["trained_em"],
                    heldout_em=rec["heldout_em"], contrast_prompts=rec["contrast_prompts"],
                    contrast_contexts=rec["contrast_contexts"],
                    share_heldout=shares["heldout_prompts"]["new_share"])
            for key, s in shares.items():
                tel.log("share", arm=arm, checkpoint=name, text_set=key,
                        **{k: v for k, v in s.items() if k != "counts"}, counts=s["counts"])
            print(f"[{arm} {name}] trained EM={rec['trained_em']:.1f} | heldout EM="
                  f"{rec['heldout_em']:.1f} | new share ctx={shares['trained_context']['new_share']:.3f}"
                  f"/{shares['heldout_contexts']['new_share']:.3f} prompts="
                  f"{shares['trained_prompts']['new_share']:.3f}/"
                  f"{shares['heldout_prompts']['new_share']:.3f} "
                  f"contrast_prompts={rec['contrast_prompts']:+.3f}", flush=True)
            return rec

        checkpoint("post_surgery")

        run_router_warmup(
            model, tok, warm_texts or [context],
            WarmupConfig(
                steps=int(cfg["warmup"]["steps"]),
                forced_slots=int(cfg["warmup"]["forced_slots"]),
                lr=float(cfg["warmup"]["lr"]), max_len=max_len, seed=args.seed,
                distractor_texts=(tune_texts if arm == "w2" else []),
                share_target=float(cfg["warmup"].get("share_target", 0.07)),
                penalty_weight=float(cfg["warmup"].get("penalty_weight", 1.0)),
            ),
            log=lambda **row: tel.log(row.pop("stage", "warmup_step"), arm=arm, **row),
        )
        checkpoint("post_warmup")

        try:
            tune = tune_new_share(
                model, tok, tune_texts, band=band, max_len=max_len,
                log=lambda **row: tel.log(row.pop("stage", "share_tune"), arm=arm, **row),
            )
        except ShareBandUnreachable as exc:
            print(f"[{arm}] SHARE BAND UNREACHABLE = INSUFFICIENT: {exc}", flush=True)
            arm_rec["share_band_unreachable"] = str(exc)
            arm_rec["g2"] = {"band_reached": False}
            summary["arms"][arm] = arm_rec
            del gen, model
            gc.collect()
            torch.cuda.empty_cache()
            continue
        arm_rec["share_tune"] = {k: v for k, v in tune.items() if k != "history"}
        arm_rec["share_tune_history"] = tune["history"]
        final = checkpoint("post_tune")

        post_surgery_em = arm_rec["checkpoints"]["post_surgery"]["heldout_em"]
        drop = post_surgery_em - final["heldout_em"]
        band_ok = band[0] <= tune["share"] <= band[1]
        contrast_ok = final["contrast_prompts"] >= args.contrast_min
        drop_ok = drop <= args.heldout_max_drop
        arm_rec["g2"] = {
            "band_reached": band_ok,
            "share_heldout_tune_texts": tune["share"],
            "mask_value": tune["mask_value"],
            "contrast_prompts": final["contrast_prompts"],
            "contrast_ok": contrast_ok,
            "heldout_drop": drop,
            "heldout_drop_ok": drop_ok,
            # (b) only applies to W2; W1 is not asked to show contrast it was not built for
            "passes": band_ok and drop_ok and (contrast_ok if arm == "w2" else True),
        }
        arm_rec["seconds"] = round(time.time() - t_arm, 1)
        summary["arms"][arm] = arm_rec
        print(f"[{arm} G2] band={band_ok} contrast={contrast_ok} "
              f"({final['contrast_prompts']:+.3f}) drop={drop:+.2f} -> "
              f"{'PASS' if arm_rec['g2']['passes'] else 'FAIL'}", flush=True)
        tel.log("arm_g2", arm=arm, **arm_rec["g2"])
        del gen, model
        gc.collect()
        torch.cuda.empty_cache()

    g2 = {arm: rec.get("g2", {}) for arm, rec in summary["arms"].items()}
    any_band = any(g.get("band_reached") for g in g2.values())
    w2 = g2.get("w2", {})
    w1 = g2.get("w1", {})
    if w2.get("passes"):
        decision = "w2"
    elif w1.get("passes"):
        decision = "w1"
    elif not any_band:
        decision = "STOP_BAND_UNREACHABLE"
    else:
        decision = "STOP_NEITHER_ARM"
    summary["decision"] = decision
    summary["teacher_phase_s"] = teacher_s
    summary["timings"] = {"total_s": time.time() - t_start}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    tel.log("arms_summary", decision=decision, g2=g2)
    print(f"[gate] E2 decision = {decision} (w2 {w2.get('passes')} / w1 {w1.get('passes')})",
          flush=True)
    print(f"[done] {run_dir / 'summary.json'} wall={summary['timings']['total_s']:.0f}s")
    return 0 if decision in ("w1", "w2") else 1


if __name__ == "__main__":
    raise SystemExit(main())
