#!/usr/bin/env python3
"""Sleep cycles over N passages, scored pooled -- the M3v2 gate at usable resolution.

The single-passage gate has 5 questions, so one question is worth 20 points and a
2-point threshold cannot be resolved. This runner keeps the protocol identical (one
cycle per passage, teacher frozen, adapters rebuilt per passage, no-context closed-book
SQuAD) and pools the questions, so the gate reads a number whose granularity is
100 / n_questions.

M3v2 adds the three things the failed gate needed, all of them within one run on one
card so every comparison is own-base paired:

* `--warmup-arm {w1,w2}`: W1 = forced-inclusion warmup only; W2 = forced inclusion plus
  a contrastive penalty that caps the new bank's probability mass on held-out text.
* a router-bias bisection (`tune_new_share`) that puts the new bank's HELD-OUT routing
  share inside a band around the uniform null, applied through SFT and evaluation, with
  the routers then frozen for SFT so the bias is the only routing control.
* `--sft-mode {qa,cpt,dreams,recap}`: the sleep arm (answer-masked pairs), the
  same-substrate SFT-on-C comparator, the no-dream ablation (all dreams, no gradient
  selection, whole-text loss), and the recap-prose sleep arm (gradient selection,
  whole-text loss on recap dreams -- the CPT objective applied to dream text, which is
  the arm whose supervised-token dose is comparable to the cpt arm).
* every arm trains exactly `sft.steps` steps (review2 A1) and its supervised-token dose
  is counted in-training and written to the summary (review2 A2).

Every arm also reads a HELD-OUT frame (never-trained passages and their questions), so
forgetting is measured in the same run rather than assumed:

    python scripts/run_multi.py --data data/squad_gate24.jsonl --limit-passages 20 \
        --cycles 1 --sft-mode qa --warmup-arm w2 --run m3v2_sleep
"""

from __future__ import annotations

import argparse
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
    dream_report,
    extract_qa_from_dreams,
    literalism_verdict,
    prose_grounding,
    score_dreams,
    scorable_dreams,
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
    current_state,
    load_tokenizer,
    routing_identity_check,
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
    steps_for_arm,
    train_sft,
)
from qlora_sleep.teacher import dream_teacher_stage  # noqa: E402
from qlora_sleep.telemetry import Telemetry, frame_row  # noqa: E402


def examples_from(passage: dict) -> list[EvalExample]:
    return [
        EvalExample(question=qa["question"], answers=qa["answers"], context=passage["context"],
                    passage_id=passage.get("passage_id", ""))
        for qa in passage["questions"]
    ]


def share_on(model, tok, texts: list[str], max_len: int, chunk: int = 16) -> dict:
    """Aggregate new-expert routing share over `texts`, chunked to bound VRAM.

    The full 68-expert count vector is summed across chunks and returned, so the top-4
    BASE expert share -- the control that makes `new_share` interpretable -- is
    recomputable from the artifact instead of being taken on trust.
    """
    if not texts:
        return {"n_texts": 0, "new_share": None, "counts": None}
    start = None
    counts = None
    for i in range(0, len(texts), chunk):
        rep = routing_share(model, tok, texts[i : i + chunk], max_len=max_len)
        start = rep["new_expert_start"]
        c = torch.tensor(rep["counts"], dtype=torch.float64)
        counts = c if counts is None else counts + c
    total = float(counts.sum())
    new = counts[start:]
    base = counts[:start]
    top4_base = float(base.topk(min(4, base.numel())).values.sum() / total)
    return {
        "n_texts": len(texts),
        "new_share": float(new.sum() / total),
        "uniform_new_share": float(new.numel() / counts.numel()),
        "top4_base_share": top4_base,
        "max_base_share": float(base.max() / total),
        "counts": [float(v) for v in counts.tolist()],
    }


def _mean(values: list) -> float | None:
    return sum(values) / len(values) if values else None


def warm_texts_for(pairs: list[tuple[str, str]], kept: list[str]) -> list[str]:
    """The warmup positive set: the dream's QA pairs in the EVAL template, every arm.

    Warming on dream prose teaches the new columns to attract on text the gate never
    asks; the pairs are the question shape the evaluation uses. The template is the SFT
    one in both arms, so the warmup recipe is identical across arms and the only
    difference between them stays the SFT objective. Fallback is the kept dreams, never
    the passage: the sleep path must not see C.
    """
    if pairs:
        return [QA_TEMPLATE.format(question=question, answer=answer)
                for question, answer in pairs]
    return list(kept)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="single_n1")
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit-passages", type=int, default=0)
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--base-em", type=float, default=None)
    ap.add_argument("--sft-mode", choices=["dreams", "qa", "cpt", "recap"], default=None)
    ap.add_argument("--warmup-arm", choices=["w1", "w2"], default="w1")
    ap.add_argument("--heldout-data", default="data/squad_base50.jsonl")
    ap.add_argument("--heldout-passages", type=int, default=20)
    ap.add_argument("--tune-passages", default="20:28",
                    help="slice of the held-out file used ONLY to tune the router bias")
    ap.add_argument("--share-band", default="0.05,0.09")
    ap.add_argument("--no-share-tune", action="store_true")
    # eval batch is a THROUGHPUT knob, not a protocol knob, but batching flips ~1 answer
    # of 109 (measured: EM 11.01 at batch 8/32, 11.93 at 16/64), so one value is used for
    # every arm and both frames of a run
    ap.add_argument("--eval-batch", type=int, default=None)
    # degradation step 2 of the pre-agreed order: read the held-out frame on every
    # `--heldout-read-every`-th student rather than all 20
    ap.add_argument("--heldout-read-every", type=int, default=1)
    # the teacher is the SAME frozen snapshot for every arm, so its dreams are a shared
    # input: generate once, reuse across arms
    ap.add_argument("--dreams-cache", default=None)
    # E1 chooses the dream prompt; the choice has to be overridable per run, otherwise
    # the arm that produced it cannot be reproduced from the command line
    ap.add_argument("--with-context", action="store_true",
                    help="also read with-context EM post-surgery and final (G3's third condition); "
                         "costs n_questions extra generations per passage")
    ap.add_argument("--dream-variant", default=None)
    ap.add_argument("--dream-temperature", type=float, default=None)
    # residual test: dreams written while the teacher's router is FORCED off its
    # standard top-k (replaces this many top-k slots with non-top-k experts). Only the
    # dream-generation forward passes are perturbed; eval reads run with no state.
    ap.add_argument("--dream-perturb-slots", type=int, default=None)
    # resume: skip passages whose rows are already in rows.jsonl (see phase 2)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--run", default="m3_multi")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    sft_mode = args.sft_mode or str(cfg["sft"].get("mode", "dreams"))
    max_len = int(cfg["sft"]["max_len"])
    band = tuple(float(v) for v in args.share_band.split(","))
    dream_variant = args.dream_variant or str(cfg["dream"]["prompt_variant"])
    dream_temperature = (args.dream_temperature if args.dream_temperature is not None
                         else float(cfg["dream"]["temperature"]))
    dream_perturb_slots = int(
        args.dream_perturb_slots if args.dream_perturb_slots is not None
        else cfg["dream"].get("perturb_slots", 0)
    )
    train_routers = bool(cfg["sft"].get("train_routers", False))
    eval_batch = int(args.eval_batch or cfg["eval"]["batch_size"])
    run_dir = Path("runs") / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    tel = Telemetry(run_dir / "metrics.jsonl")
    t_start = time.time()

    payload = read_jsonl(args.data)
    passages = payload["passages"]
    if args.limit_passages:
        passages = passages[: args.limit_passages]
    n_questions = sum(len(p["questions"]) for p in passages)
    tel.log("data", n_passages=len(passages), n_questions=n_questions, sft_mode=sft_mode,
            warmup_arm=args.warmup_arm, share_band=list(band), train_routers=train_routers,
            dream_variant=dream_variant, dream_temperature=dream_temperature,
            dream_perturb_slots=dream_perturb_slots)
    print(f"[data] {len(passages)} passages, {n_questions} questions, "
          f"sft_mode={sft_mode} warmup_arm={args.warmup_arm} band={band}", flush=True)

    # ---------------- held-out frames (never trained on) --------------------
    heldout = read_jsonl(args.heldout_data)["passages"] if args.heldout_data else []
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
    read_contexts = [p["context"] for p in read_passages]
    read_prompts = [build_prompt(ex.question) for ex in heldout_examples]
    print(f"[heldout] read {len(read_passages)} passages / {len(heldout_examples)} q; "
          f"bias-tune on {len(tune_passages)} passages / {len(tune_texts)} texts", flush=True)
    tel.log("heldout", n_read_passages=len(read_passages), n_read_questions=len(heldout_examples),
            n_tune_passages=len(tune_passages), n_tune_texts=len(tune_texts))

    tok = load_tokenizer(cfg["model"]["id"])

    # ---------------- phase 1: all dreams in one teacher residency -----------
    # The teacher is the frozen base snapshot for every passage (v1 design), so one
    # residency covers the whole run.
    t = time.time()
    cache_path = Path(args.dreams_cache) if args.dreams_cache else None
    dream_stats: list[dict] = []
    # resolved ONCE: the save below creates the cache file, so re-testing existence later
    # would label a freshly generated dream pass as a cache hit
    cache_hit = cache_path is not None and cache_path.exists()
    if cache_hit:
        cached = torch.load(cache_path, weights_only=False)
        stage = {"dreams": cached["dreams"], "expert_load": cached["expert_load"]}
        # the cache records the routing perturbation its dreams were written under, so an
        # arm can state which dream set it trained on instead of inferring it from argv
        dream_stats = list(cached.get("perturb", {}).get("per_passage", []))
        print(f"[dream] {len(passages)} passages from cache {cache_path}", flush=True)
    else:
        stage = dream_teacher_stage(
            cfg["model"]["id"], tok,
            [p["context"] for p in passages],
            [
                DreamConfig(
                    n_dreams=int(cfg["dream"]["n_dreams"]),
                    temperature=dream_temperature,
                    top_p=float(cfg["dream"]["top_p"]),
                    max_new_tokens=int(cfg["dream"]["max_new_tokens"]),
                    prompt_variant=dream_variant,
                    random_expert=bool(cfg["dream"]["random_expert"]),
                    perturb_slots=dream_perturb_slots,
                    seed=args.seed + i,
                )
                for i in range(len(passages))
            ],
            device=args.device,
            attn_implementation=cfg["model"]["attn_implementation"],
            max_len=max_len,
            stats=dream_stats,
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"dreams": stage["dreams"], "expert_load": stage["expert_load"],
                        "perturb": {"slots": dream_perturb_slots,
                                    "per_passage": dream_stats}}, cache_path)
    # MEASURED caveat: the clone source is the routing load pooled over ALL passages
    # (the teacher stage measures it per call), not per passage. It only selects which
    # base experts get cloned, and every passage draws from the same frozen base, so
    # the choice is a shared initial condition rather than a per-passage one.
    load = stage["expert_load"]
    dreams_by_passage = stage["dreams"]
    timings = {"teacher_phase_s": time.time() - t}
    print(f"[dream] {len(passages)} passages in {timings['teacher_phase_s']:.0f}s", flush=True)

    # PERTURBATION STRENGTH actually applied while the dreams were written. Read from the
    # pass that ran (or from the cache's record of it), never from the requested setting:
    # a perturbation that silently selected nothing would otherwise look like a standard
    # dream pass with a flag set.
    # NOT named _mean: the module already has _mean(values: list) for the dose block, and
    # a same-named nested helper shadows it for the rest of main() (it did -- the R-pert
    # arm's summary write died with "unhashable type: 'list'" at _mean(dose["steps"])).
    def _mean_stat(key: str):
        vals = [s[key] for s in dream_stats if isinstance(s, dict) and key in s]
        return sum(vals) / len(vals) if vals else None

    dream_perturb = {
        "requested_slots": dream_perturb_slots,
        "n_passages_with_stats": len([s for s in dream_stats if s]),
        "a1_off_standard_fraction": _mean_stat("a1_off_standard_fraction"),
        "mean_drawn_prob": _mean_stat("mean_drawn_prob"),
        "mean_displaced_prob": _mean_stat("mean_displaced_prob"),
        "tokens": sum(int(s.get("tokens", 0)) for s in dream_stats if isinstance(s, dict)),
        "source": "cache" if cache_hit else "generated",
    }
    tel.log("dream_perturb", **dream_perturb)
    print(f"[dream] perturbation {dream_perturb}", flush=True)

    # ---------------- phase 2: one fresh student per passage ----------------
    rows: list[dict] = []
    totals = {"base": 0.0, "final": 0.0, "n": 0,
              "heldout_base": 0.0, "heldout_final": 0.0, "heldout_n": 0}
    # DOSE (review2 A2): steps and supervised tokens per passage, aggregated into the
    # summary so an arm's training signal is stated rather than inferred from steps
    dose: dict[str, list] = {"steps": [], "supervised_tokens": []}
    heldout_post_surgery = None
    band_insufficient = False
    eval_identity: dict | None = None

    # --resume: an arm costs ~2 GPU-hours, and until this seat existed a defect at
    # passage 12 threw away the 11 passages already paid for. Completed rows are
    # appended to rows.jsonl as they finish (metrics.jsonl drops list fields, so it
    # cannot carry the per-question predictions the bootstrap pairs on).
    rows_path = run_dir / "rows.jsonl"
    done_rows: dict[int, dict] = {}
    if args.resume and rows_path.exists():
        for line in rows_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done_rows[int(row["passage"])] = row
        metrics_rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()
                        if line.strip()] if (run_dir / "metrics.jsonl").exists() else []
        for m in metrics_rows:
            if m.get("stage") == "heldout_read" and m.get("checkpoint") == "post_surgery":
                heldout_post_surgery = {"em": m["em"], "f1": m["f1"], "n": 0}
        for m in metrics_rows:
            if m.get("stage") == "sft" and int(m["passage"]) in done_rows:
                dose["steps"].append(m["steps"])
                dose["supervised_tokens"].append(m["supervised_tokens"])
        print(f"[resume] {len(done_rows)} passages already complete; "
              f"{len(passages) - len(done_rows)} to run", flush=True)
    for i, passage in enumerate(passages):
        t_p = time.time()
        examples = examples_from(passage)
        if i in done_rows:
            row = done_rows[i]
            rows.append(row)
            totals["base"] += row["base_em"] * row["n_questions"]
            totals["final"] += row["final_em"] * row["n_questions"]
            totals["n"] += row["n_questions"]
            if row.get("heldout_em") is not None and heldout_examples:
                totals["heldout_final"] += row["heldout_em"] * len(heldout_examples)
                totals["heldout_n"] += len(heldout_examples)
            print(f"[p{i}] resumed from rows.jsonl base={row['base_em']:.0f} "
                  f"final={row['final_em']:.0f}", flush=True)
            continue
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
        if eval_identity is None:
            # "the perturbation hook is off at eval" is MEASURED here, not asserted: (a)
            # no routing state is installed on the eval path, and (b) a zero-strength
            # state is bit-identical to no state while a non-zero strength is not (the
            # asymmetry control that keeps the check falsifiable).
            identity = routing_identity_check(model, tok, [passage["context"]],
                                              max_len=max_len)
            eval_identity = {"eval_hook_off": current_state() is None, **identity}
            tel.log("eval_routing_identity", passage=i, **eval_identity)
            print(f"[eval] routing identity identical={identity['identical']} "
                  f"hook_off={eval_identity['eval_hook_off']} "
                  f"perturb_moves_logits={identity['perturbation_changes_logits']}",
                  flush=True)
        # the untrained student is the honest per-passage reference: it carries the
        # same quantisation and surgery as the trained one, so the delta is training
        base_read = evaluate(gen, examples, "no_context")
        stage_reads: dict[str, dict] = {"post_surgery": base_read}
        ctx_post = (evaluate(gen, examples, "with_context") if args.with_context else None)
        selected: list = []
        n_too_short = 0
        share_tune: dict | None = None
        # one held-out reference reading, taken after surgery on the first student and
        # carried for the run: the held-out drop is then measured against a student that
        # has had the same surgery as the trained ones (not against a separate base run)
        if heldout_examples and heldout_post_surgery is None:
            heldout_post_surgery = evaluate(gen, heldout_examples, "no_context")
            tel.log("heldout_read", passage=i, checkpoint="post_surgery",
                    em=heldout_post_surgery["em"], f1=heldout_post_surgery["f1"])

        for cycle in range(args.cycles):
            dreams = dreams_by_passage[i]
            kept = list(dreams)
            prepared: dict = {"pairs": [], "examples": [], "dropped": []}
            report = dream_report(kept, passage["context"], n=int(cfg["dream"]["ngram_n"]))
            verdict = literalism_verdict(
                report["containment_median"],
                alarm=float(cfg["gates"]["literalism_alarm"]),
                insufficient=float(cfg["gates"]["literalism_insufficient"]),
            )
            if sft_mode == "dreams":
                # no-dream ablation: no gradient scoring, no top-k selection, whole-text
                # loss on every dream -- the paper's form minus the distillation
                prepared = prepare_sft_examples(
                    kept, passage["context"],
                    max_containment=float(cfg["dream"]["max_containment"]),
                    n=int(cfg["dream"]["ngram_n"]),
                )
                selected = prepared["examples"]
            elif sft_mode in ("qa", "recap"):
                # Both sleep arms share the dream SELECTION (gradient scoring, top-k +
                # random). They differ only in the objective: masked answers vs whole-text.
                # Sub-floor dreams are removed from the pool BEFORE scoring: the scorer
                # has no target for them, and the recap variant emits a tail of one-line
                # fragments. The count is logged, never swallowed.
                pool_idx = scorable_dreams(dreams, tok)
                n_too_short = len(dreams) - len(pool_idx)
                pool = [dreams[j] for j in pool_idx]
                scores = score_dreams(model, tok, pool, max_len=max_len)
                keep = select_dreams(scores, int(cfg["dream"]["top_k_keep"]),
                                     int(cfg["dream"]["random_keep"]), seed=args.seed + cycle)
                kept = [pool[j] for j in keep]
                report = dream_report(kept, passage["context"], n=int(cfg["dream"]["ngram_n"]))
                verdict = literalism_verdict(
                    report["containment_median"],
                    alarm=float(cfg["gates"]["literalism_alarm"]),
                    insufficient=float(cfg["gates"]["literalism_insufficient"]),
                )
                if sft_mode == "qa":
                    prepared = prepare_qa_examples(
                        extract_qa_from_dreams(kept), passage["context"],
                        max_containment=float(cfg["dream"]["max_containment"]),
                        n=int(cfg["dream"]["ngram_n"]),
                    )
                    selected = prepared["pairs"]
                else:
                    # recap-prose arm: the CPT objective applied to dream text. Same
                    # selection as the qa arm, whole-text loss, so its supervised-token
                    # dose is comparable to the cpt arm (a prose dream and a passage are
                    # both ~120 words) and the arm contrast is about data, not schedule.
                    prepared = prepare_sft_examples(
                        kept, passage["context"],
                        max_containment=float(cfg["dream"]["max_containment"]),
                        n=int(cfg["dream"]["ngram_n"]),
                    )
                    selected = prepared["examples"]
            else:
                # SFT-on-C comparator: the passage itself, identical build/warmup/bias
                prepared = {"pairs": [], "examples": prepare_cpt_examples(passage["context"]),
                            "dropped": []}
                selected = prepared["examples"]
            pairs_all = extract_qa_from_dreams(kept)
            prose = prose_grounding(kept, passage["context"])
            # the dreams themselves are logged: a passage that yields one usable pair
            # cannot be fixed downstream, and without the text that fact is invisible
            tel.log("dreams", passage=i, cycle=cycle, dreams=dreams)
            tel.log("cycle", passage=i, cycle=cycle, verdict=verdict,
                    containment_median=report["containment_median"],
                    n_kept=len(kept), n_examples=len(selected),
                    # dreams the scorer could not score (sub-floor), removed before
                    # selection: logged so a thin dream tail is visible in the artifact
                    n_too_short=n_too_short,
                    n_pairs=len(pairs_all),
                    answer_support=answer_support_rate(pairs_all, passage["context"]),
                    # prose grounding: answer_support_rate is blind to prose dreams (it
                    # reads extracted QA pairs), so without this a grounded paraphrase
                    # and an off-topic hallucination both read 0.0
                    prose_overlap_median=prose["median"],
                    prose_overlap_mean=prose["mean"],
                    prose_overlap_min=prose["min"],
                    prose_overlap_max=prose["max"],
                    mean_words=report["mean_words"],
                    n_dropped=len(prepared["dropped"]))
            if not selected:
                print(f"[p{i}] no training examples survived; skipping cycle", flush=True)
                continue

            if cycle == 0:
                warm = warm_texts_for(pairs_all, kept)
                if not warm:
                    print(f"[p{i}] no warmup texts; skipping cycle", flush=True)
                    continue
                # the warmup positive set is the dream QA pairs in the eval template in
                # every arm; a prose dream variant yields no pairs, so it falls back to
                # the dreams themselves and that difference is logged, not silent
                tel.log("warmup_texts", passage=i,
                        kind="qa_pairs" if pairs_all else "dream_prose", n=len(warm))
                run_router_warmup(
                    model, tok, warm,
                    WarmupConfig(
                        steps=int(cfg["warmup"]["steps"]),
                        forced_slots=int(cfg["warmup"]["forced_slots"]),
                        lr=float(cfg["warmup"]["lr"]), max_len=max_len, seed=args.seed,
                        # W2 only: the distractors are the held-out prompts and prose,
                        # never the passage being trained on
                        distractor_texts=(tune_texts if args.warmup_arm == "w2" else []),
                        share_target=float(cfg["warmup"].get("share_target", 0.07)),
                        penalty_weight=float(cfg["warmup"].get("penalty_weight", 1.0)),
                    ),
                    log=lambda **row: tel.log(row.pop("stage", "warmup_step"), passage=i, **row),
                )
                if not args.no_share_tune and tune_texts:
                    try:
                        share_tune = tune_new_share(
                            model, tok, tune_texts, band=band, max_len=max_len,
                            log=lambda **row: tel.log(row.pop("stage", "share_tune"),
                                                      passage=i, **row),
                        )
                    except ShareBandUnreachable as exc:
                        # the plan's INSUFFICIENT branch: the seat exists but the band is
                        # not reachable. Escalate, do not search hyper-parameters.
                        print(f"[p{i}] SHARE BAND UNREACHABLE = INSUFFICIENT: {exc}", flush=True)
                        tel.log("share_band_unreachable", passage=i, detail=str(exc))
                        band_insufficient = True
                        selected = []
                        break
                    tel.log("share_tuned", passage=i, mask_value=share_tune["mask_value"],
                            share=share_tune["share"], iterations=share_tune["iterations"])
                else:
                    share_tune = {"mask_value": None, "share": None, "iterations": 0}

            batch = int(cfg["sft"]["batch_size"])
            # A1 (review2): EVERY arm trains the same number of steps. The qa arm used
            # to derive its steps from the pair count, which made it 6.2 steps against
            # the comparators' 30 and confounded the arm contrast.
            steps = steps_for_arm(cfg)
            sft = train_sft(
                model, tok, selected,
                SFTConfig(steps=steps, lr=float(cfg["sft"]["lr"]), batch_size=batch,
                          max_len=max_len,
                          max_containment=float(cfg["sft"]["max_containment"]),
                          ngram_n=int(cfg["sft"]["ngram_n"]), seed=args.seed + cycle,
                          train_routers=train_routers),
                # the CPT arm trains ON the passage, so the copy-vs-store filter is off
                context=None if sft_mode == "cpt" else passage["context"],
                log=lambda **row: tel.log(row.pop("stage", "sft_step"), passage=i, **row),
            )
            tel.log("sft", passage=i, cycle=cycle, steps=steps,
                    supervised_tokens=sft["supervised_tokens"],
                    n_supervised_examples=len(selected),
                    first_loss=sft["history"][0]["loss"], last_loss=sft["final"]["loss"])
            dose["steps"].append(steps)
            dose["supervised_tokens"].append(sft["supervised_tokens"])
        if band_insufficient:
            del model
            torch.cuda.empty_cache()
            break

        final = evaluate(gen, examples, "no_context")
        stage_reads["final"] = final
        share = routing_share(model, tok, [passage["context"]], max_len=max_len)
        trained_prompt_texts = [build_prompt(ex.question) for ex in examples]
        trained_share = share_on(model, tok, [passage["context"]] + trained_prompt_texts,
                                 max_len=max_len)
        heldout_share = share_on(model, tok, read_contexts + read_prompts, max_len=max_len)
        read_heldout = bool(heldout_examples) and (i % max(args.heldout_read_every, 1) == 0)
        heldout_read = evaluate(gen, heldout_examples, "no_context") if read_heldout else None
        ctx_final = (evaluate(gen, examples, "with_context") if args.with_context else None)
        row = frame_row(
            i, passage, base_read, final, trained_share, heldout_share,
            golds=[ex.answers for ex in examples],
            heldout_read=heldout_read, share_tune=share_tune,
            seconds=round(time.time() - t_p, 1),
        )
        if ctx_post is not None:
            row["with_context_post_em"] = ctx_post["em"]
            row["with_context_final_em"] = ctx_final["em"]
            row["with_context_drop"] = ctx_post["em"] - ctx_final["em"]
        row["new_share"] = share["aggregate_new_share"]
        rows.append(row)
        totals["base"] += base_read["em"] * final["n"]
        totals["final"] += final["em"] * final["n"]
        totals["n"] += final["n"]
        if heldout_read is not None:
            totals["heldout_final"] += heldout_read["em"] * heldout_read["n"]
            totals["heldout_n"] += heldout_read["n"]
        with rows_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        tel.log("passage", **{k: v for k, v in row.items() if not isinstance(v, list)})
        print(f"[p{i}] n={final['n']} base={base_read['em']:.0f} final={final['em']:.0f} "
              f"share_ho={row['share_heldout']} ho_em={row['heldout_em']} "
              f"preds={final['predictions'][:3]}", flush=True)
        del model
        torch.cuda.empty_cache()

    if band_insufficient:
        summary = {"run": args.run, "verdict": "INSUFFICIENT",
                   "reason": "new-expert share band unreachable by router bias",
                   "n_passages": len(passages), "per_passage": rows}
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return 3

    n = max(totals["n"], 1)
    pooled_base = totals["base"] / n
    pooled_final = totals["final"] / n
    gain = pooled_final - pooled_base
    heldout_ref = None
    if heldout_post_surgery is not None:
        heldout_ref = heldout_post_surgery["em"]
    summary = {
        "run": args.run,
        "config": args.config,
        "sft_mode": sft_mode,
        "warmup_arm": args.warmup_arm,
        "dream_variant": dream_variant,
        "dream_temperature": dream_temperature,
        "cycles": args.cycles,
        "dream_perturb": dream_perturb,
        "eval_routing_identity": eval_identity,
        "n_passages": len(passages),
        "n_questions": totals["n"],
        "dose": {
            "steps_mean": _mean(dose["steps"]),
            "steps_min": min(dose["steps"]) if dose["steps"] else None,
            "steps_max": max(dose["steps"]) if dose["steps"] else None,
            "supervised_tokens_mean": _mean(dose["supervised_tokens"]),
            "supervised_tokens_min": min(dose["supervised_tokens"]) if dose["supervised_tokens"] else None,
            "supervised_tokens_max": max(dose["supervised_tokens"]) if dose["supervised_tokens"] else None,
            "supervised_tokens_total": sum(dose["supervised_tokens"]),
            "note": ("supervised tokens = label tokens the loss scored, summed over the "
                     "passage's SFT steps (answer-masked for qa, whole-text otherwise)"),
        },
        "pooled_base_em": pooled_base,
        "pooled_final_em": pooled_final,
        "pooled_gain": gain,
        "base_em_arg": args.base_em,
        "n_heldout_questions": totals["heldout_n"],
        "eval_batch": eval_batch,
        "heldout_read_every": args.heldout_read_every,
        "heldout_post_surgery_em": heldout_ref,
        "heldout_final_em": (totals["heldout_final"] / totals["heldout_n"]
                             if totals["heldout_n"] else None),
        "timings": {**timings, "total_s": time.time() - t_start},
        "per_passage": rows,
    }
    gates = {}
    if args.base_em is not None:
        summary["gain_over_external_base"] = pooled_final - args.base_em
        gates["pooled_gain_vs_external_base"] = (
            summary["gain_over_external_base"] >= float(cfg["gates"]["single_passage_min_gain"])
        )
    gates["pooled_gain_vs_own_base"] = gain >= float(cfg["gates"]["single_passage_min_gain"])
    if heldout_ref is not None and totals["heldout_n"]:
        drop = heldout_ref - summary["heldout_final_em"]
        summary["heldout_drop"] = drop
        # non-destruction: the tuned bank must not cost the model what it already knew
        gates["heldout_drop_within_2pts"] = drop <= float(cfg["gates"]["heldout_max_drop"])
    summary["gates"] = gates
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    tel.log("summary", pooled_base_em=pooled_base, pooled_final_em=pooled_final,
            pooled_gain=gain, n_questions=totals["n"], heldout_final_em=summary["heldout_final_em"])
    print(f"[gate] pooled base={pooled_base:.2f} final={pooled_final:.2f} "
          f"gain={gain:+.2f} pts over {totals['n']} questions | heldout "
          f"{heldout_ref} -> {summary['heldout_final_em']} over {totals['heldout_n']} q "
          f"-> {gates}", flush=True)
    print(f"[done] {run_dir / 'summary.json'} wall={summary['timings']['total_s']:.0f}s")
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
