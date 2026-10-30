#!/usr/bin/env python3
"""The Table-3 corpus protocol: ONE student per arm, aggregate training over N passages.

This is the plan3 runner and it is a different shape from `run_multi.py` on purpose.
`run_multi` gives every passage a FRESH student (20 students, 5 questions each); the
paper's CPT column (sleep.txt:617-620) exposes a SINGLE student to all 200 passages in
one continued-pretraining-style run and evaluates it on all their questions. That is the
frame the full-protocol gate is registered on, and it is the frame that makes
`steps = 30 x n_passages` a corpus dose rather than a per-passage one.

Aggregate training is the paper's own construction ("combine them into an aggregated
synthetic dataset"): every selected example from every passage is pooled, shuffled once
with seed 0, and ONE `train_sft` call trains for `steps = sft.steps x n_passages` with a
single optimiser. Per-passage dose conventions survive by arithmetic -- the qa arm sees
24000 pair visits over ~2200 pairs (~10.9 epochs, E5 measured ~10.5), cpt 120 epochs and
nodream 15 epochs, exactly as the per-passage runs had -- so the arm contrast is data,
not schedule.

Arms (`--sft-mode`), all sharing ONE build, ONE warmup and ONE router-bias tune:

  qa      sleep: gradient-scored dream selection, answer-masked QA pairs in the eval
          template (the paper's Sleep row);
  cpt     SFT-on-C: the passage text itself, whole-text loss, context NOT stripped
          (the paper's "fine-tuned model with no dreaming");
  dreams  no-dream ablation: all 8 dreams, no selection, whole-text loss
          (the paper's "removing Dreaming").

Frames, all read on the arm's OWN base (before training) and after:
  F1 corpus-orig  every question of the corpus, EM primary
  F2 corpus-para  the teacher's paraphrase of each question, gold = the ORIGINAL answers
                  (the format control; primary single-hop frame when >= 800 accepted)
  F3 transfer     compositional two-hop questions, F1 primary
  F4 heldout      never-trained passages, no-context plus a with-context subsample

Spend guards: a fixed 200-question probe every `--probe-every` steps with the
pre-registered abort rule (three consecutive probes more than 10 pts below the arm's own
base AND the latest more than 15 below -> save state, mark the arm DESTROYED, stop), and
a rotating runtime-state checkpoint every `--checkpoint-every` steps so `--resume` costs
minutes rather than an arm.

    python3 scripts/run_corpus.py --sft-mode qa --run p6_s --dreams-cache runs/p6_dreams.pt
"""

from __future__ import annotations

import argparse
import json
import random
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
    dream_report,
    extract_qa_from_dreams,
    literalism_verdict,
    prose_grounding,
    scorable_dreams,
    score_dreams,
    select_dreams,
)
from qlora_sleep.eval_squad import (  # noqa: E402
    EvalExample,
    build_prompt,
    evaluate,
    model_generate_fn,
)
from qlora_sleep.model import (  # noqa: E402
    build_student,
    load_runtime_state,
    load_tokenizer,
    save_runtime_state,
)
from qlora_sleep.router_warmup import (  # noqa: E402
    ShareBandUnreachable,
    WarmupConfig,
    routing_share,
    run_router_warmup,
    tune_new_share,
)
from qlora_sleep.sft import (  # noqa: E402
    QA_TEMPLATE,
    SFTConfig,
    prepare_cpt_examples,
    prepare_qa_examples,
    prepare_sft_examples,
    train_sft,
)
from qlora_sleep.teacher import dream_teacher_stage  # noqa: E402
from qlora_sleep.telemetry import Telemetry  # noqa: E402

# Eval-only frames. Training on any of these voids the protocol: the transfer and
# paraphrase frames are the measurement, and the corpus is what is being trained on.
FROZEN_FRAMES = {"corpus_para.jsonl", "transfer_twohop.jsonl", "squad_base50.jsonl",
                 "squad_gate24.jsonl"}


def refuse_training_data(path: str) -> None:
    """Raise if `path` is an eval-only frame rather than the training corpus."""
    name = Path(path).name
    payload = read_jsonl(path)
    if payload["meta"].get("eval_only") or name in FROZEN_FRAMES:
        raise SystemExit(
            f"refusing {path} as --data: it is a frozen eval frame, not the training corpus"
        )


def examples_from(passage: dict) -> list[EvalExample]:
    return [
        EvalExample(question=qa["question"], answers=qa["answers"],
                    context=passage["context"], passage_id=passage.get("passage_id", ""))
        for qa in passage["questions"]
    ]


def read_frame(path: str | None) -> tuple[list[EvalExample], dict]:
    """Read an eval-only frame built by `scripts/build_frames.py`."""
    if not path or not Path(path).exists():
        return [], {}
    payload = read_jsonl(path)
    examples = [
        EvalExample(question=row["question"], answers=row["answers"],
                    context=row.get("context", ""), passage_id=row.get("passage_id", ""))
        for row in payload["passages"]
    ]
    return examples, payload["meta"]


def share_on(model, tok, texts: list[str], max_len: int, chunk: int = 16) -> dict:
    if not texts:
        return {"n_texts": 0, "new_share": None}
    start = None
    counts = None
    for i in range(0, len(texts), chunk):
        rep = routing_share(model, tok, texts[i : i + chunk], max_len=max_len)
        start = rep["new_expert_start"]
        c = torch.tensor(rep["counts"], dtype=torch.float64)
        counts = c if counts is None else counts + c
    total = float(counts.sum())
    base = counts[:start]
    return {
        "n_texts": len(texts),
        "new_share": float(counts[start:].sum() / total),
        "uniform_new_share": float(counts[start:].numel() / counts.numel()),
        "top4_base_share": float(base.topk(min(4, base.numel())).values.sum() / total),
    }


def _mean(values: list) -> float | None:
    return sum(values) / len(values) if values else None


def warmup_texts(dreams_by_passage: list[list[str]], n_pairs: int, seed: int) -> tuple[list[str], str]:
    """The warmup positive set: ONE seed-0 sample of dream QA pairs, identical per arm.

    Identical across arms by construction (the same cache, the same seed, the same
    sample size), so the E4/E5 property that the build is a shared initial condition --
    and therefore that the arm contrast is the SFT objective -- survives into the corpus
    run. Falls back to dream prose if the variant yields no pairs, and says which.
    """
    pairs: list[tuple[str, str]] = []
    for dreams in dreams_by_passage:
        pairs.extend(extract_qa_from_dreams(dreams))
    if pairs:
        rng = random.Random(seed)
        chosen = pairs if len(pairs) <= n_pairs else rng.sample(pairs, n_pairs)
        return ([QA_TEMPLATE.format(question=q, answer=a) for q, a in chosen], "qa_pairs")
    prose = [d for dreams in dreams_by_passage for d in dreams if d.strip()]
    rng = random.Random(seed)
    chosen = prose if len(prose) <= n_pairs else rng.sample(prose, n_pairs)
    return chosen, "dream_prose"


def build_examples(
    mode: str,
    passages: list[dict],
    dreams_by_passage: list[list[str]],
    model,
    tok,
    cfg: dict,
    seed: int,
    tel: Telemetry,
    max_len: int,
) -> dict:
    """Pool every passage's selected examples into one aggregate training set."""
    pooled: list = []
    per_passage: list[dict] = []
    containments: list[float] = []
    for i, passage in enumerate(passages):
        dreams = list(dreams_by_passage[i])
        kept = list(dreams)
        n_too_short = 0
        prepared: dict = {"pairs": [], "examples": [], "dropped": []}
        if mode == "cpt":
            prepared = {"pairs": [], "examples": prepare_cpt_examples(passage["context"]),
                        "dropped": []}
        elif mode == "dreams":
            prepared = prepare_sft_examples(
                kept, passage["context"],
                max_containment=float(cfg["dream"]["max_containment"]),
                n=int(cfg["dream"]["ngram_n"]),
            )
        else:
            pool_idx = scorable_dreams(dreams, tok)
            n_too_short = len(dreams) - len(pool_idx)
            pool = [dreams[j] for j in pool_idx]
            scores = score_dreams(model, tok, pool, max_len=max_len)
            keep = select_dreams(scores, int(cfg["dream"]["top_k_keep"]),
                                 int(cfg["dream"]["random_keep"]), seed=seed + i)
            kept = [pool[j] for j in keep]
            prepared = prepare_qa_examples(
                extract_qa_from_dreams(kept), passage["context"],
                max_containment=float(cfg["dream"]["max_containment"]),
                n=int(cfg["dream"]["ngram_n"]),
            )
        selected = prepared["pairs"] if mode == "qa" else prepared["examples"]
        report = dream_report(kept, passage["context"], n=int(cfg["dream"]["ngram_n"]))
        verdict = literalism_verdict(
            report["containment_median"],
            alarm=float(cfg["gates"]["literalism_alarm"]),
            insufficient=float(cfg["gates"]["literalism_insufficient"]),
        )
        pairs_all = extract_qa_from_dreams(kept)
        prose = prose_grounding(kept, passage["context"])
        containments.append(report["containment_median"])
        tel.log("cycle", passage=i, verdict=verdict,
                containment_median=report["containment_median"],
                n_kept=len(kept), n_examples=len(selected), n_too_short=n_too_short,
                n_pairs=len(pairs_all),
                answer_support=answer_support_rate(pairs_all, passage["context"]),
                prose_overlap_median=prose["median"],
                n_dropped=len(prepared["dropped"]))
        per_passage.append({"passage": i, "n_examples": len(selected),
                            "n_dropped": len(prepared["dropped"]),
                            "containment_median": report["containment_median"],
                            "verdict": verdict})
        pooled.extend(selected)
    random.Random(seed).shuffle(pooled)
    ordered = sorted(containments)
    n = len(ordered)
    median = (ordered[n // 2] if n % 2 else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])) if n else None
    return {
        "examples": pooled,
        "per_passage": per_passage,
        "literalism": {
            "n_passages": n,
            "median_containment": median,
            "alarm_count": sum(1 for p in per_passage if p["verdict"] == "ALARM"),
            "ok_count": sum(1 for p in per_passage if p["verdict"] == "OK"),
            "insufficient_count": sum(1 for p in per_passage if p["verdict"] == "INSUFFICIENT"),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="corpus_p6")
    ap.add_argument("--data", default="data/squad_corpus200.jsonl")
    ap.add_argument("--sft-mode", choices=["qa", "cpt", "dreams"], required=True)
    ap.add_argument("--dreams-cache", default="runs/p6_dreams.pt")
    ap.add_argument("--para-data", default="data/corpus_para.jsonl")
    ap.add_argument("--transfer-data", default="data/transfer_twohop.jsonl")
    ap.add_argument("--heldout-data", default="data/squad_base50.jsonl")
    ap.add_argument("--heldout-passages", type=int, default=20)
    ap.add_argument("--tune-passages", default="20:28")
    ap.add_argument("--with-context-questions", type=int, default=150)
    ap.add_argument("--share-band", default="0.05,0.09")
    ap.add_argument("--no-share-tune", action="store_true")
    ap.add_argument("--warmup-pairs", type=int, default=40)
    ap.add_argument("--eval-batch", type=int, default=None)
    ap.add_argument("--probe-every", type=int, default=1000)
    ap.add_argument("--probe-questions", type=int, default=200)
    ap.add_argument("--checkpoint-every", type=int, default=500)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dream-variant", default=None)
    ap.add_argument("--dream-temperature", type=float, default=None)
    ap.add_argument("--limit-passages", type=int, default=0)
    ap.add_argument("--run", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    refuse_training_data(args.data)
    cfg = load_config(args.config)
    mode = args.sft_mode
    max_len = int(cfg["sft"]["max_len"])
    band = tuple(float(v) for v in args.share_band.split(","))
    eval_batch = int(args.eval_batch or cfg["eval"]["batch_size"])
    dream_variant = args.dream_variant or str(cfg["dream"]["prompt_variant"])
    dream_temperature = (args.dream_temperature if args.dream_temperature is not None
                         else float(cfg["dream"]["temperature"]))
    train_routers = bool(cfg["sft"].get("train_routers", False))
    run_dir = Path("runs") / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    tel = Telemetry(run_dir / "metrics.jsonl")
    t_start = time.time()

    payload = read_jsonl(args.data)
    passages = payload["passages"]
    if args.limit_passages:
        passages = passages[: args.limit_passages]
    n_questions = sum(len(p["questions"]) for p in passages)
    steps = int(cfg["sft"]["steps"]) * len(passages)
    tel.log("data", data=args.data, n_passages=len(passages), n_questions=n_questions,
            sft_mode=mode, steps=steps, share_band=list(band),
            dream_variant=dream_variant, dream_temperature=dream_temperature,
            train_routers=train_routers, corpus_meta=payload["meta"])
    print(f"[data] {len(passages)} passages, {n_questions} questions, mode={mode}, "
          f"aggregate steps={steps}", flush=True)

    # ---------------- frames (never trained on) -----------------------------
    heldout = read_jsonl(args.heldout_data)["passages"]
    trained_ids = {p.get("passage_id", "") for p in passages}
    read_passages = [p for p in heldout[: args.heldout_passages]
                     if p.get("passage_id", "") not in trained_ids]
    tune_start, _, tune_end = args.tune_passages.partition(":")
    tune_passages = [p for p in heldout[int(tune_start) : int(tune_end or len(heldout))]
                     if p.get("passage_id", "") not in trained_ids]
    heldout_examples = [ex for p in read_passages for ex in examples_from(p)]
    tune_texts = [p["context"] for p in tune_passages] + [
        build_prompt(qa["question"]) for p in tune_passages for qa in p["questions"]
    ]
    f1_examples = [ex for p in passages for ex in examples_from(p)]
    f2_examples, f2_meta = read_frame(args.para_data)
    f3_examples, f3_meta = read_frame(args.transfer_data)
    ctx_examples = f1_examples[: args.with_context_questions]
    print(f"[frames] F1 {len(f1_examples)} q | F2 {len(f2_examples)} q | "
          f"F3 {len(f3_examples)} q | F4 {len(heldout_examples)} q | "
          f"with-context {len(ctx_examples)} q", flush=True)
    tel.log("frames", f1=len(f1_examples), f2=len(f2_examples), f3=len(f3_examples),
            f4=len(heldout_examples), with_context=len(ctx_examples),
            f2_meta=f2_meta, f3_meta=f3_meta)

    # the probe subsample is fixed before training so every probe reads the same questions
    probe_rng = random.Random(args.seed)
    probe_pool = list(f1_examples)
    probe_examples = (probe_pool if len(probe_pool) <= args.probe_questions
                      else probe_rng.sample(probe_pool, args.probe_questions))
    probe_examples.sort(key=lambda ex: (ex.passage_id, ex.question))

    tok = load_tokenizer(cfg["model"]["id"])

    # ---------------- phase 1: dreams (shared teacher pass) ------------------
    t = time.time()
    cache_path = Path(args.dreams_cache)
    if cache_path.exists():
        cached = torch.load(cache_path, weights_only=False)
        dreams_by_passage = cached["dreams"]
        load = cached["expert_load"]
        print(f"[dream] {len(dreams_by_passage)} passages from cache {cache_path}", flush=True)
    else:
        stage = dream_teacher_stage(
            cfg["model"]["id"], tok, [p["context"] for p in passages],
            [
                DreamConfig(
                    n_dreams=int(cfg["dream"]["n_dreams"]),
                    temperature=dream_temperature,
                    top_p=float(cfg["dream"]["top_p"]),
                    max_new_tokens=int(cfg["dream"]["max_new_tokens"]),
                    prompt_variant=dream_variant,
                    random_expert=bool(cfg["dream"]["random_expert"]),
                    seed=args.seed + i,
                )
                for i in range(len(passages))
            ],
            device=args.device,
            attn_implementation=cfg["model"]["attn_implementation"],
            max_len=max_len,
        )
        dreams_by_passage, load = stage["dreams"], stage["expert_load"]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"dreams": dreams_by_passage, "expert_load": load}, cache_path)
    timings = {"dream_phase_s": time.time() - t}
    print(f"[dream] {len(dreams_by_passage)} passages in {timings['dream_phase_s']:.0f}s",
          flush=True)

    # ---------------- phase 2: ONE student, one build ----------------------
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
    gen = model_generate_fn(model, tok, max_new_tokens=int(cfg["eval"]["max_new_tokens"]),
                            batch_size=eval_batch)

    warm, warm_kind = warmup_texts(dreams_by_passage, args.warmup_pairs, args.seed)
    tel.log("warmup_texts", kind=warm_kind, n=len(warm))
    print(f"[warmup] {len(warm)} texts ({warm_kind})", flush=True)
    run_router_warmup(
        model, tok, warm,
        WarmupConfig(
            steps=int(cfg["warmup"]["steps"]),
            forced_slots=int(cfg["warmup"]["forced_slots"]),
            lr=float(cfg["warmup"]["lr"]), max_len=max_len, seed=args.seed,
            share_target=float(cfg["warmup"].get("share_target", 0.07)),
            penalty_weight=float(cfg["warmup"].get("penalty_weight", 1.0)),
        ),
        log=lambda **row: tel.log(row.pop("stage", "warmup_step"), **row),
    )
    share_tune: dict = {"mask_value": None, "share": None, "iterations": 0}
    if not args.no_share_tune and tune_texts:
        try:
            share_tune = tune_new_share(
                model, tok, tune_texts, band=band, max_len=max_len,
                log=lambda **row: tel.log(row.pop("stage", "share_tune"), **row),
            )
        except ShareBandUnreachable as exc:
            print(f"SHARE BAND UNREACHABLE = INSUFFICIENT: {exc}", flush=True)
            tel.log("share_band_unreachable", detail=str(exc))
            (run_dir / "summary.json").write_text(json.dumps(
                {"run": args.run, "sft_mode": mode, "verdict": "INSUFFICIENT",
                 "reason": "new-expert share band unreachable by router bias",
                 "detail": str(exc)}, indent=2))
            return 3
        tel.log("share_tuned", mask_value=share_tune["mask_value"],
                share=share_tune["share"], iterations=share_tune["iterations"])
    print(f"[build] share band {band} -> mask {share_tune['mask_value']} "
          f"share {share_tune['share']}", flush=True)
    timings["build_phase_s"] = time.time() - t

    # ---------------- phase 3: OWN-BASE reads, every frame ----------------
    t = time.time()
    reads: dict[str, dict] = {}
    for name, examples in (("corpus_orig", f1_examples), ("corpus_para", f2_examples),
                           ("transfer", f3_examples), ("heldout", heldout_examples)):
        if not examples:
            continue
        read = evaluate(gen, examples, "no_context")
        reads[name] = {"base": read}
        print(f"[base] {name}: n={read['n']} EM={read['em']:.2f} F1={read['f1']:.2f}", flush=True)
    ctx_base = evaluate(gen, ctx_examples, "with_context") if ctx_examples else None
    probe_base = evaluate(gen, probe_examples, "no_context")
    timings["base_reads_s"] = time.time() - t
    tel.log("base_reads", **{f"{k}_em": v["base"]["em"] for k, v in reads.items()},
            **{f"{k}_f1": v["base"]["f1"] for k, v in reads.items()})

    # ---------------- phase 4: aggregate training ------------------------
    t = time.time()
    built = build_examples(mode, passages, dreams_by_passage, model, tok, cfg,
                           args.seed, tel, max_len)
    examples = built["examples"]
    if not examples:
        print("[fatal] no training examples survived filtering", flush=True)
        (run_dir / "summary.json").write_text(json.dumps(
            {"run": args.run, "sft_mode": mode, "verdict": "NO_EXAMPLES"}, indent=2))
        return 5
    tel.log("aggregate", n_examples=len(examples), n_passages=len(passages), steps=steps,
            literalism=built["literalism"])
    print(f"[aggregate] {len(examples)} examples from {len(passages)} passages, "
          f"{steps} steps, literalism {built['literalism']}", flush=True)

    ckpt_path = run_dir / "runtime_last.pt"
    start_step = 0
    if args.resume and ckpt_path.exists():
        restored = load_runtime_state(model, ckpt_path)
        start_step = int(restored["step"] or 0)
        print(f"[resume] restored {restored['loaded']} tensors, step {start_step}, "
              f"mask {restored['mask_value']}", flush=True)
        tel.log("resume", start_step=start_step, **restored)

    cadence = [v for v in (args.checkpoint_every, args.probe_every) if v]
    callback_every = min(cadence) if cadence else 0
    abort_state = {"streak": 0, "last": None}

    def on_step(step: int) -> dict:
        """Rotating checkpoint + the pre-registered probe/abort rule."""
        if args.checkpoint_every and step % args.checkpoint_every == 0:
            saved = save_runtime_state(model, ckpt_path, step=step)
            tel.log("checkpoint", step=step, n_lora=saved["n_lora"])
        if not (args.probe_every and step % args.probe_every == 0):
            return {}
        read = evaluate(gen, probe_examples, "no_context")
        delta = read["em"] - probe_base["em"]
        abort_state["streak"] = abort_state["streak"] + 1 if delta < -10.0 else 0
        abort = bool(abort_state["streak"] >= 3 and delta < -15.0)
        abort_state["last"] = {"step": step, "em": read["em"], "delta_pts": delta,
                               "streak": abort_state["streak"], "abort": abort}
        print(f"[probe {step}] EM={read['em']:.2f} delta={delta:+.2f} "
              f"streak={abort_state['streak']}{' ABORT' if abort else ''}", flush=True)
        return {"probe_em": read["em"], "probe_delta_pts": delta,
                "probe_streak": abort_state["streak"], "abort": abort}

    sft = train_sft(
        model, tok, examples,
        SFTConfig(steps=steps, lr=float(cfg["sft"]["lr"]),
                  batch_size=int(cfg["sft"]["batch_size"]), max_len=max_len,
                  max_containment=float(cfg["sft"]["max_containment"]),
                  ngram_n=int(cfg["sft"]["ngram_n"]), seed=args.seed,
                  train_routers=train_routers, start_step=start_step),
        # the copy-vs-store filter ran at PREPARATION time against each example's OWN
        # passage (prepare_qa_examples / prepare_sft_examples assert it there); the
        # aggregate batch mixes passages, so a single context cannot express the
        # invariant here and the cpt arm is the control that deliberately sees C.
        context=None,
        log=lambda **row: tel.log(row.pop("stage", "sft_step"), **row),
        probe=on_step,
        probe_every=callback_every,
    )
    timings["train_phase_s"] = time.time() - t
    label_tokens_total = sum(int(r["label_tokens"]) for r in sft["history"])
    tel.log("sft", steps=sft["steps"], start_step=start_step, aborted=sft["aborted"],
            supervised_tokens=sft["supervised_tokens"],
            label_tokens_total=label_tokens_total,
            n_supervised_examples=len(examples),
            first_loss=sft["history"][0]["loss"] if sft["history"] else None,
            last_loss=sft["final"].get("loss"))
    print(f"[sft] {sft['steps']} steps, last loss {sft['final'].get('loss')}, "
          f"{label_tokens_total} label tokens, aborted={sft['aborted']}", flush=True)

    # ---------------- phase 5: final reads --------------------------------
    t = time.time()
    final: dict[str, dict] = {}
    for name in list(reads):
        examples_for = {"corpus_orig": f1_examples, "corpus_para": f2_examples,
                        "transfer": f3_examples, "heldout": heldout_examples}[name]
        read = evaluate(gen, examples_for, "no_context")
        final[name] = read
        print(f"[final] {name}: EM={read['em']:.2f} F1={read['f1']:.2f}", flush=True)
    ctx_final = evaluate(gen, ctx_examples, "with_context") if ctx_examples else None
    trained_texts = [build_prompt(ex.question) for ex in f1_examples[:100]]
    trained_share = share_on(model, tok, trained_texts, max_len=max_len)
    heldout_share = share_on(model, tok, [p["context"] for p in read_passages]
                             + [build_prompt(ex.question) for ex in heldout_examples[:60]],
                             max_len=max_len)
    timings["final_reads_s"] = time.time() - t
    timings["total_s"] = time.time() - t_start

    frames = {}
    for name, entry in reads.items():
        base_read, final_read = entry["base"], final[name]
        frames[name] = {
            "n": final_read["n"],
            "base_em": base_read["em"], "final_em": final_read["em"],
            "base_f1": base_read["f1"], "final_f1": final_read["f1"],
            "base_predictions": base_read["predictions"],
            "predictions": final_read["predictions"],
            "golds": [[a for a in ex.answers] for ex in
                      {"corpus_orig": f1_examples, "corpus_para": f2_examples,
                       "transfer": f3_examples, "heldout": heldout_examples}[name]],
        }
    summary = {
        "run": args.run,
        "protocol": "plan3 corpus (Table 3 CPT column shape: one student, N passages)",
        "config": args.config,
        "sft_mode": mode,
        "dream_variant": dream_variant,
        "dream_temperature": dream_temperature,
        "data": args.data,
        "data_meta": payload["meta"],
        "n_passages": len(passages),
        "n_questions": n_questions,
        "frames": frames,
        "with_context": None if ctx_base is None else {
            "n": ctx_base["n"], "base_em": ctx_base["em"], "final_em": ctx_final["em"],
            "drop_pts": ctx_base["em"] - ctx_final["em"],
        },
        "probe": {"n": len(probe_examples), "base_em": probe_base["em"],
                  "last": abort_state["last"]},
        "training": {
            "steps_requested": steps, "steps_trained": sft["steps"],
            "start_step": start_step, "aborted": sft["aborted"],
            "batch_size": int(cfg["sft"]["batch_size"]), "lr": float(cfg["sft"]["lr"]),
            "train_routers": train_routers,
            "first_loss": sft["history"][0]["loss"] if sft["history"] else None,
            "last_loss": sft["final"].get("loss"),
            "epochs": None,
        },
        "dose": {
            "steps": sft["steps"],
            "supervised_tokens_total": label_tokens_total,
            "supervised_tokens_mean_per_passage": label_tokens_total / max(len(passages), 1),
            "n_examples": len(examples),
            "examples_per_passage": len(examples) / max(len(passages), 1),
            "visits_per_example": (sft["steps"] * int(cfg["sft"]["batch_size"]))
            / max(len(examples), 1),
            "note": ("label tokens the loss scored, summed over all SFT steps (answer-masked "
                     "for qa, whole-text otherwise); visits_per_example is the epoch count"),
        },
        "literalism": built["literalism"],
        "per_passage_examples": built["per_passage"],
        "share": {
            "band": list(band), "mask_value": share_tune["mask_value"],
            "tune_share": share_tune["share"],
            "trained_texts": trained_share, "heldout_texts": heldout_share,
        },
        "eval_batch": eval_batch,
        "timings": timings,
    }
    summary["training"]["epochs"] = summary["dose"]["visits_per_example"]
    arm_status = "DESTROYED" if sft["aborted"] else "COMPLETE"
    summary["arm_status"] = arm_status
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    tel.log("summary", arm_status=arm_status, **{
        f"{k}_{m}": v for k, entry in frames.items() for m, v in
        (("em", entry["final_em"]), ("f1", entry["final_f1"]))
    })
    print(f"[done] {run_dir / 'summary.json'} status={arm_status} "
          f"wall={timings['total_s']:.0f}s", flush=True)
    return 4 if sft["aborted"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
