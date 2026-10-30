#!/usr/bin/env python3
"""Build the two eval-only frames: the transfer set and the paraphrase control.

Both frames are constructed MECHANICALLY from the corpus and the dream cache, and the
generator never scores its own output. The teacher only RENDERS; admission is decided by
fixed string predicates, so a fluent-but-wrong render cannot admit itself.

TRANSFER (compositional two-hop). Hops come from the same dream cache the arms train on
(quiz@0.7): per passage, pairs = extract_qa_from_dreams(dreams), kept as hop candidates
only if the answer occurs in that passage (the per-pair form of `answer_support_rate`).
A candidate is (passage i: q1 -> a1) + (passage j: q2 -> a2), i != j, where q2 CONTAINS
a1 (word-boundary) -- q2 asks about the entity whose fact lives in C_i, while a2 lives in
C_j and is deliberately ABSENT from C_i, so the question is unanswerable without
combining two passages. The teacher rewrites q2 so a1 is referred to INDIRECTLY.
Admission is t1-t8 below; nothing about them is a judgement call.

PARAPHRASE (the E5 format control). The teacher paraphrases every corpus question; the
gold stays the ORIGINAL answers. This removes surface-string alignment between the
training pairs and the evaluation prompt, which is the confound E5 named.

The HUMAN SPOT-READ GATE writes 20 accepted and 10 rejected samples (with the rejecting
predicate) into the artefact before any arm runs: mechanical predicates can be satisfied
by degenerate text, and the only instrument that sees that is a reader.

Frames are IMMUTABLE once written (`--force` is required to rebuild, and the runner
refuses them as training data).

    python3 scripts/build_frames.py --mine-only              # CPU: mining + yield only
    python3 scripts/build_frames.py --limit-candidates 8     # GPU smoke
    python3 scripts/build_frames.py                          # full build
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

from squad_prep import read_jsonl, write_jsonl  # noqa: E402

from qlora_sleep.dream import (  # noqa: E402
    content_words,
    extract_qa_from_dreams,
    ngram_containment,
    words,
)

MAX_QUESTION_WORDS = 45
MIN_TRANSFER_OVERLAP = 0.5
RELAXED_TRANSFER_OVERLAP = 0.4
MIN_PARAPHRASE_OVERLAP = 0.3
MAX_PARAPHRASE_OVERLAP = 0.9

TRANSFER_EXAMPLES = [
    ("Who wrote the novel that was adapted into the 1975 film?",
     "Ivan Doig",
     "Who wrote the book whose author also wrote the 1975 screenplay adaptation?"),
    ("Which river does the city named after a saint stand on?",
     "the Tagus",
     "Which river does the city named after a saint stand on?"),
    ("What year did the company founded by the engineer release its first tractor?",
     "John Froelich",
     "What year did the company founded by the engineer release its first tractor?"),
]

PARAPHRASE_EXAMPLES = [
    ("What is the capital of the state where the river rises?",
     "In which state capital does the river that starts in the mountains end up?"),
    ("Who was the architect of the cathedral built in 1300?",
     "Which architect designed the cathedral that dates from 1300?"),
]


def norm(text: str) -> str:
    return " ".join(words(text))


def contains(haystack: str, needle: str) -> bool:
    """Word-boundary containment on the normalised word sequence."""
    n = norm(needle)
    if not n:
        return False
    return f" {n} " in f" {norm(haystack)} "


def overlap(reference: str, other: str) -> float:
    """Fraction of `reference`'s content words that occur in `other`."""
    ref = content_words(reference)
    if not ref:
        return 0.0
    return len(ref & content_words(other)) / len(ref)


# --------------------------------------------------------------------------- #
# mining (CPU, no teacher)
# --------------------------------------------------------------------------- #


_SENTENCE_END = ".!?:;\n\"'("


def entity_like(answer: str, context: str) -> bool:
    """Is `answer` a NAMED ENTITY in `context` rather than a number or a common word?

    Why this exists (MEASURED on the first frame build): the bridge predicate was
    `q2 contains a1` on the normalised word sequence, which any frequent word satisfies.
    Of the first 20 accepted two-hop questions, 16 were degenerate -- the "bridge" was
    `two`, `south`, `new`, `black`, `one`, `2010` -- so the composed question hid a
    numeral, and the frame measured single-passage recall rather than composition. The
    base-model askability read agrees: with-context EM was 23.50 with both passages
    against 24.50 with the SECOND passage alone, i.e. the first passage was not needed.

    The test is mechanical and deliberately conservative: no purely numeric token, and at
    least one token occurring CAPITALISED mid-sentence in the passage (proper-noun
    evidence). It rejects numbers and closed-class words without a stoplist to maintain.
    """
    toks = words(answer)
    if not toks:
        return False
    if any(tok.replace(",", "").isdigit() for tok in toks):
        return False
    lowered = {tok.lower() for tok in toks}
    for match in re.finditer(r"[A-Za-z0-9']+", context):
        token = match.group(0)
        if token.lower() not in lowered:
            continue
        if not token[0].isupper():
            continue
        before = context[: match.start()].rstrip()
        if before and before[-1] not in _SENTENCE_END:
            return True
    return False


def supported_pairs(passages: list[dict], dreams_by_passage: list[list[str]]) -> list[list[tuple[str, str]]]:
    """Dream QA pairs per passage, kept only when the answer occurs in that passage."""
    out: list[list[tuple[str, str]]] = []
    for passage, dreams in zip(passages, dreams_by_passage):
        ctx = passage["context"]
        out.append([(q, a) for q, a in extract_qa_from_dreams(dreams) if contains(ctx, a)])
    return out


def mine_twohop(
    passages: list[dict],
    pairs_by_passage: list[list[tuple[str, str]]],
    cap: int = 600,
    seed: int = 0,
) -> dict:
    """All ordered two-hop candidates, capped by a seed-0 sample.

    The bridge is mined, not generated: q2 must literally contain a1 as a word sequence,
    and a2 must be absent from C_i. That is what makes the composed question a genuine
    two-hop rather than a reworded single-hop.
    """
    index: dict[str, list[tuple[int, str, str]]] = {}
    for i, pairs in enumerate(pairs_by_passage):
        for question, answer in pairs:
            index.setdefault(norm(answer), []).append((i, question, answer))
    seen: set[tuple] = set()
    candidates: list[dict] = []
    n_not_entity = 0
    for j, pairs in enumerate(pairs_by_passage):
        ctx_j = passages[j]["context"]
        for q2, a2 in pairs:
            toks = norm(q2).split()
            if not toks:
                continue
            for n in range(1, min(12, len(toks)) + 1):
                for start in range(len(toks) - n + 1):
                    span = " ".join(toks[start : start + n])
                    for i, q1, a1 in index.get(span, ()):
                        if i == j:
                            continue
                        if not contains(passages[i]["context"], a1):
                            continue
                        if not contains(ctx_j, a2):
                            continue
                        # t4: the answer must be unavailable from the first passage alone
                        if contains(passages[i]["context"], a2):
                            continue
                        # the bridge must be an ENTITY, not a numeral or a common word
                        if not entity_like(a1, passages[i]["context"]):
                            n_not_entity += 1
                            continue
                        key = (i, q1, a1, j, q2, a2)
                        if key in seen:
                            continue
                        seen.add(key)
                        candidates.append({
                            "passage_id": passages[i].get("passage_id", ""),
                            "support_passage_id": passages[j].get("passage_id", ""),
                            "hop1_question": q1, "hop1_answer": a1,
                            "hop2_question": q2, "hop2_answer": a2,
                            "hop1_context": passages[i]["context"],
                            "hop2_context": ctx_j,
                        })
    candidates.sort(key=lambda c: (c["passage_id"], c["hop1_question"],
                                   c["support_passage_id"], c["hop2_question"]))
    rng = random.Random(seed)
    sampled = candidates if len(candidates) <= cap else rng.sample(candidates, cap)
    return {"n_candidates": len(candidates), "candidates": sampled, "cap": cap,
            "n_bridge_not_entity": n_not_entity}


# --------------------------------------------------------------------------- #
# rendering prompts
# --------------------------------------------------------------------------- #


def transfer_prompt(candidate: dict) -> str:
    shots = "\n".join(
        f"Question: {q}\nEntity to hide: {entity}\nRewritten: {rewritten}"
        for q, entity, rewritten in TRANSFER_EXAMPLES
    )
    return (
        "Rewrite a question so that the named entity is NOT mentioned, while the question "
        "still asks for the same fact. Refer to the entity indirectly.\n\n"
        f"{shots}\n\n"
        f"Passage: {candidate['hop1_context']}\n\n"
        f"Question: {candidate['hop2_question']}\n"
        f"Entity to hide: {candidate['hop1_answer']}\n"
        "Rewritten:"
    )


def paraphrase_prompt(question: str) -> str:
    shots = "\n".join(f"Question: {q}\nRewritten: {r}" for q, r in PARAPHRASE_EXAMPLES)
    return (
        "Rewrite the question in different words. Keep the meaning and do not answer it.\n\n"
        f"{shots}\n\n"
        f"Question: {question}\nRewritten:"
    )


def clean_render(text: str) -> str:
    """First non-empty line, prefix labels stripped, single line of prose."""
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        for prefix in ("Rewritten:", "Question:", "A:", "Answer:"):
            if line.lower().startswith(prefix.lower()):
                line = line[len(prefix) :].strip()
        return " ".join(line.split())
    return ""


def template_render(candidate: dict) -> str:
    """Deterministic fallback: replace the a1 mention in q2 with a pointer to q1."""
    q2 = candidate["hop2_question"]
    a1 = candidate["hop1_answer"]
    q1 = candidate["hop1_question"]
    pattern_words = norm(a1).split()
    out_words: list[str] = []
    toks = q2.split()
    lowered = [norm(t) for t in toks]
    i = 0
    replaced = False
    while i < len(toks):
        if not replaced and lowered[i : i + len(pattern_words)] == pattern_words:
            # "the the answer to ..." when q2 already had an article before the entity
            if out_words and out_words[-1].lower() in ("the", "a", "an"):
                out_words.pop()
            out_words.append(f'the answer to "{q1}"')
            i += len(pattern_words)
            replaced = True
            continue
        out_words.append(toks[i])
        i += 1
    if not replaced:
        return f'{q2.rstrip("?")} (where the missing name is the answer to "{q1}")?'
    return " ".join(out_words)


# --------------------------------------------------------------------------- #
# mechanical acceptance
# --------------------------------------------------------------------------- #


def accept_transfer(candidate: dict, composed: str, accepted: list[str],
                    min_overlap: float = MIN_TRANSFER_OVERLAP) -> tuple[bool, str]:
    """t1-t8. Returns (accepted, reason) -- the reason is recorded for the spot-read."""
    a1, a2 = candidate["hop1_answer"], candidate["hop2_answer"]
    q2 = candidate["hop2_question"]
    if not composed:
        return False, "t0 empty render"
    if contains(composed, a1):
        return False, "t1 bridge entity leaked into the question"
    if contains(composed, a2):
        return False, "t2 gold answer leaked into the question"
    if not contains(candidate["hop1_context"], a1):
        return False, "t3a bridge entity not in the first passage"
    if not contains(candidate["hop2_context"], a2):
        return False, "t3b gold not in the second passage"
    if contains(candidate["hop1_context"], a2):
        return False, "t4 gold reachable from the first passage alone"
    # t5: the hop-2 FACT must survive. The reference is q2's content words MINUS the
    # bridge entity's own words, because hiding that entity is precisely the required
    # transformation -- scoring against the full set would reject every correct render
    # for deleting the name it was asked to delete (measured on the first pass).
    reference = content_words(q2) - content_words(a1)
    kept = len(reference & content_words(composed)) / len(reference) if reference else 1.0
    if kept < min_overlap:
        return False, f"t5 lost the hop-2 fact (overlap {kept:.2f})"
    if len(composed.split()) > MAX_QUESTION_WORDS:
        return False, "t6 too long"
    if not composed.rstrip().endswith("?"):
        return False, "t6 not a question"
    for other in accepted:
        if ngram_containment(composed, other, 8) > 0.5 or ngram_containment(other, composed, 8) > 0.5:
            return False, "t7 duplicate of an accepted question"
    return True, "accepted"


def accept_paraphrase(question: str, paraphrase: str, golds: list[str]) -> tuple[bool, str]:
    """p1-p3: no answer leak, genuinely reworded, still a question."""
    if not paraphrase:
        return False, "p0 empty render"
    for gold in golds:
        if contains(paraphrase, gold):
            return False, "p1 gold answer leaked into the paraphrase"
    ov = overlap(question, paraphrase)
    if ov < MIN_PARAPHRASE_OVERLAP:
        return False, f"p2 not the same question (overlap {ov:.2f})"
    if ov > MAX_PARAPHRASE_OVERLAP:
        return False, f"p2 barely reworded (overlap {ov:.2f})"
    if len(paraphrase.split()) > MAX_QUESTION_WORDS:
        return False, "p3 too long"
    if not paraphrase.rstrip().endswith("?"):
        return False, "p3 not a question"
    return True, "accepted"


# --------------------------------------------------------------------------- #
# frame writing
# --------------------------------------------------------------------------- #


def write_frame(path: Path, meta: dict, rows: list[dict], force: bool = False) -> Path:
    """Write an eval-only frame; refuse to overwrite one (immutability after freeze)."""
    if path.exists() and not force:
        raise SystemExit(
            f"{path} already exists and frames are immutable once frozen; pass --force to "
            f"rebuild deliberately (this invalidates every number measured on it)"
        )
    write_jsonl({"meta": {"eval_only": True, "frozen": True, **meta}, "passages": rows}, path)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="corpus_p6")
    ap.add_argument("--corpus", default="data/squad_corpus200.jsonl")
    ap.add_argument("--dreams-cache", default="runs/p6_dreams.pt")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--run-dir", default="runs/p6_frames")
    ap.add_argument("--cap", type=int, default=600)
    ap.add_argument("--limit-candidates", type=int, default=0)
    ap.add_argument("--mine-only", action="store_true")
    ap.add_argument("--no-transfer", action="store_true")
    ap.add_argument("--no-paraphrase", action="store_true")
    ap.add_argument("--askability", action="store_true",
                    help="also read base with-context EM on both passages vs each single one")
    ap.add_argument("--template-only", action="store_true",
                    help="render the transfer frame deterministically (no teacher, no GPU)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--teacher-batch", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from qlora_sleep.config import load_config

    cfg = load_config(args.config)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    passages = read_jsonl(args.corpus)["passages"]
    import torch

    cached = torch.load(Path(args.dreams_cache), weights_only=False)
    dreams_by_passage = cached["dreams"]
    if len(dreams_by_passage) != len(passages):
        raise SystemExit(
            f"dream cache has {len(dreams_by_passage)} passages, corpus has {len(passages)}"
        )
    pairs_by_passage = supported_pairs(passages, dreams_by_passage)
    n_pairs = sum(len(p) for p in pairs_by_passage)
    mined = mine_twohop(passages, pairs_by_passage, cap=args.cap, seed=args.seed)
    print(f"[mine] {n_pairs} supported dream pairs over {len(passages)} passages -> "
          f"{mined['n_candidates']} two-hop candidates, {len(mined['candidates'])} kept "
          f"(cap {args.cap}); {mined['n_bridge_not_entity']} bridge spans rejected as "
          f"non-entities", flush=True)
    (run_dir / "mining.json").write_text(json.dumps({
        "n_passages": len(passages), "n_supported_pairs": n_pairs,
        "n_candidates": mined["n_candidates"], "cap": args.cap,
        "n_kept": len(mined["candidates"]),
        "n_bridge_not_entity": mined["n_bridge_not_entity"],
    }, indent=2))
    if args.mine_only:
        print("[mine] --mine-only: no rendering")
        return 0

    candidates = mined["candidates"]
    if args.limit_candidates:
        candidates = candidates[: args.limit_candidates]
    corpus_questions = [
        (p.get("passage_id", ""), qa["question"], qa["answers"], p["context"])
        for p in passages for qa in p["questions"]
    ]

    # ---------------- one teacher residency for both frames ----------------
    import gc

    from qlora_sleep.eval_squad import EvalExample, evaluate, model_generate_fn
    from qlora_sleep.model import load_tokenizer
    from qlora_sleep.teacher import load_teacher

    tok = load_tokenizer(cfg["model"]["id"])
    # A template-only rebuild needs no teacher at all (the render is deterministic), so
    # the residency is skipped rather than paid for: that is what makes tightening the
    # bridge predicate and rebuilding the frame a zero-GPU-minute operation.
    need_teacher = ((not args.template_only and candidates and not args.no_transfer)
                    or not args.no_paraphrase or args.askability)
    teacher = None
    gen = None
    if need_teacher:
        teacher = load_teacher(cfg["model"]["id"], device=args.device,
                               attn_implementation=cfg["model"]["attn_implementation"])
        gen = model_generate_fn(teacher, tok, max_new_tokens=args.max_new_tokens,
                                batch_size=args.teacher_batch)

    def admit(cands: list[dict], renders: list[str], min_overlap: float, render: str) -> dict:
        """Run the t1-t8 acceptance over one pass of renders."""
        accepted_texts: list[str] = []
        rows: list[dict] = []
        log: list[dict] = []
        for candidate, raw in zip(cands, renders):
            composed = clean_render(raw)
            ok, reason = accept_transfer(candidate, composed, accepted_texts,
                                        min_overlap=min_overlap)
            log.append({"accepted": ok, "reason": reason, "composed": composed,
                        "hop1_question": candidate["hop1_question"],
                        "hop2_question": candidate["hop2_question"]})
            if not ok:
                continue
            accepted_texts.append(composed)
            rows.append({
                "passage_id": candidate["passage_id"],
                "support_passage_id": candidate["support_passage_id"],
                "question": composed,
                "answers": [candidate["hop2_answer"]],
                "hop1_question": candidate["hop1_question"],
                "hop1_answer": candidate["hop1_answer"],
                "hop2_question": candidate["hop2_question"],
                "hop2_answer": candidate["hop2_answer"],
                "hop1_context": candidate["hop1_context"],
                "hop2_context": candidate["hop2_context"],
                "render": render,
                "t5_overlap": round(overlap(candidate["hop2_question"], composed), 3),
            })
        return {"rows": rows, "log": log, "n_rejected": len(cands) - len(rows)}

    try:
        transfer_rows, transfer_log, n_rejected, ladder = [], [], 0, "n/a"
        if candidates and not args.no_transfer:
            t = time.time()
            if args.template_only:
                # deterministic render straight into the last rung of the yield ladder
                result = admit(candidates, [template_render(c) for c in candidates],
                               RELAXED_TRANSFER_OVERLAP, "template")
                transfer_rows, transfer_log, n_rejected = (
                    result["rows"], result["log"], result["n_rejected"])
                ladder = "template"
                print(f"[transfer] template-only: {len(transfer_rows)}/{len(candidates)} "
                      f"accepted in {time.time() - t:.0f}s", flush=True)
            else:
                renders = gen([transfer_prompt(c) for c in candidates])
                result = admit(candidates, renders, MIN_TRANSFER_OVERLAP, "teacher_fewshot")
                transfer_rows, transfer_log, n_rejected = (
                    result["rows"], result["log"], result["n_rejected"])
                ladder = "t5>=0.5"
                print(f"[transfer] pass 1 (t5>=0.5): {len(transfer_rows)}/{len(candidates)} "
                      f"accepted in {time.time() - t:.0f}s", flush=True)
                # yield ladder, pre-registered: relax t5 to 0.4 on the SAME renders (no
                # re-render cost), then deterministic template rendering, then
                # INSUFFICIENT.
                if len(transfer_rows) < int(cfg["gates"]["transfer_min_accepted"]):
                    result = admit(candidates, renders, RELAXED_TRANSFER_OVERLAP,
                                   "teacher_fewshot_relaxed")
                    transfer_rows, transfer_log, n_rejected = (
                        result["rows"], result["log"], result["n_rejected"])
                    ladder = "t5>=0.4"
                    print(f"[transfer] pass 2 (t5>=0.4): {len(transfer_rows)} accepted",
                          flush=True)
            if len(transfer_rows) < int(cfg["gates"]["transfer_min_accepted"]):
                result = admit(candidates, [template_render(c) for c in candidates],
                               RELAXED_TRANSFER_OVERLAP, "template")
                transfer_rows, transfer_log, n_rejected = (
                    result["rows"], result["log"], result["n_rejected"])
                ladder = "template"
                print(f"[transfer] pass 3 (template): {len(transfer_rows)} accepted",
                      flush=True)
            print(f"[transfer] {len(transfer_rows)}/{len(candidates)} accepted "
                  f"(ladder {ladder}), {n_rejected} rejected", flush=True)

            if args.askability and transfer_rows:
                # STRATIFICATION telemetry (never an admission filter): can the base
                # answer with BOTH passages in context, and with each single passage?
                # A question already answerable from one passage is not a two-hop for
                # this model, and the transfer EM is reported stratified on this. The
                # reading is PER ROW, not a pooled scalar broadcast to every row (the
                # first build did the latter, which makes the stratum uncomputable).
                import bootstrap_gate as bg

                def read(contexts: list[str]) -> dict:
                    examples = [
                        EvalExample(question=row["question"], answers=row["answers"],
                                    context=ctx)
                        for row, ctx in zip(transfer_rows, contexts)
                    ]
                    return evaluate(gen, examples, "with_context")

                both = read([f"{r['hop1_context']}\n\n{r['hop2_context']}"
                             for r in transfer_rows])
                hop1 = read([r["hop1_context"] for r in transfer_rows])
                hop2 = read([r["hop2_context"] for r in transfer_rows])
                for row, p_both, p_i, p_j in zip(transfer_rows, both["predictions"],
                                                 hop1["predictions"], hop2["predictions"]):
                    golds = row["answers"]
                    row["askability"] = {
                        "both": bg.score(p_both, golds, "em"),
                        "hop1_only": bg.score(p_i, golds, "em"),
                        "hop2_only": bg.score(p_j, golds, "em"),
                        "pred_both": p_both,
                        "pred_hop2_only": p_j,
                    }
                compositional = [r for r in transfer_rows
                                 if r["askability"]["hop2_only"] == 0.0]
                print(f"[transfer] askability with-context EM: both {both['em']:.2f}, "
                      f"hop1 only {hop1['em']:.2f}, hop2 only {hop2['em']:.2f} -> "
                      f"{len(compositional)}/{len(transfer_rows)} rows are NOT answerable "
                      f"from the second passage alone (the compositional stratum)",
                      flush=True)

        para_rows, para_log = [], []
        if not args.no_paraphrase:
            t = time.time()
            renders = gen([paraphrase_prompt(q) for _, q, _, _ in corpus_questions])
            for (pid, question, golds, ctx), raw in zip(corpus_questions, renders):
                paraphrase = clean_render(raw)
                ok, reason = accept_paraphrase(question, paraphrase, golds)
                para_log.append({"accepted": ok, "reason": reason, "paraphrase": paraphrase,
                                 "original": question})
                if ok:
                    para_rows.append({
                        "passage_id": pid, "question": paraphrase, "answers": list(golds),
                        "original_question": question, "context": ctx,
                        "overlap": round(overlap(question, paraphrase), 3),
                    })
            rate = len(para_rows) / max(len(corpus_questions), 1)
            print(f"[paraphrase] {len(para_rows)}/{len(corpus_questions)} accepted "
                  f"({rate:.1%}) in {time.time() - t:.0f}s", flush=True)
    finally:
        teacher = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------------- frames + the human spot-read artefact ----------------
    out_dir = Path(args.out_dir)
    meta_common = {"built_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "corpus": args.corpus,
                   "dreams_cache": args.dreams_cache, "dream_variant": cfg["dream"]["prompt_variant"],
                   "dream_temperature": cfg["dream"]["temperature"], "seed": args.seed}
    written = []
    if transfer_rows:
        meta = {**meta_common, "frame": "transfer_twohop",
                "n_candidates": len(candidates), "n_accepted": len(transfer_rows),
                "n_rejected": n_rejected, "yield_ladder": ladder,
                "acceptance": "t1-t8 mechanical; the generator never scores its output"}
        written.append(write_frame(out_dir / "transfer_twohop.jsonl", meta, transfer_rows,
                                   force=args.force))
    if para_rows:
        rate = len(para_rows) / max(len(corpus_questions), 1)
        primary = ("corpus_para" if len(para_rows) >= int(cfg["gates"]["paraphrase_min_accepted"])
                   else "corpus_orig")
        meta = {**meta_common, "frame": "corpus_para",
                "n_questions": len(corpus_questions), "n_accepted": len(para_rows),
                "accept_rate": round(rate, 4), "acceptance": "p1-p3 mechanical",
                "primary_single_hop_frame": primary}
        written.append(write_frame(out_dir / "corpus_para.jsonl", meta, para_rows,
                                   force=args.force))

    spot = []
    spot.append("HUMAN SPOT-READ GATE -- 20 accepted transfer questions")
    for row in transfer_rows[:20]:
        spot.append(f"\n[ACCEPTED] {row['question']}")
        spot.append(f"  hop1: {row['hop1_question']} -> {row['hop1_answer']}")
        spot.append(f"  hop2: {row['hop2_question']} -> {row['hop2_answer']}")
    spot.append("\n\nHUMAN SPOT-READ GATE -- 10 rejected transfer renders, with reasons")
    for row in [r for r in transfer_log if not r["accepted"]][:10]:
        spot.append(f"\n[REJECTED] {row['reason']}")
        spot.append(f"  render: {row['composed']}")
        spot.append(f"  hop1: {row['hop1_question']}")
        spot.append(f"  hop2: {row['hop2_question']}")
    spot.append("\n\nPARAPHRASE SAMPLES -- 10 accepted")
    for row in para_rows[:10]:
        spot.append(f"\n[ACCEPTED] {row['original_question']}\n  -> {row['question']}")
    spot.append("\n\nPARAPHRASE SAMPLES -- 10 rejected, with reasons")
    for row in [r for r in para_log if not r["accepted"]][:10]:
        spot.append(f"\n[REJECTED] {row['reason']}\n  render: {row['paraphrase']}"
                    f"\n  original: {row['original']}")
    spot_text = "\n".join(spot)
    (run_dir / "spot_read.txt").write_text(spot_text)
    print("\n" + spot_text + "\n")

    summary = {
        "frames_written": [str(p) for p in written],
        "n_transfer_accepted": len(transfer_rows),
        "n_transfer_candidates": len(candidates),
        "transfer_yield_ladder": ladder if transfer_rows else None,
        "n_para_accepted": len(para_rows),
        "para_accept_rate": (len(para_rows) / max(len(corpus_questions), 1)) if para_rows else None,
        "primary_single_hop_frame": ("corpus_para" if para_rows and len(para_rows) >=
                                     int(cfg["gates"]["paraphrase_min_accepted"])
                                     else "corpus_orig"),
        "transfer_gate": ("INSUFFICIENT (yield < 100 accepted)" if transfer_rows and
                          len(transfer_rows) < int(cfg["gates"]["transfer_min_accepted"])
                          else ("MEASURED" if transfer_rows else "NO_CANDIDATES")),
        "spot_read_artifact": str(run_dir / "spot_read.txt"),
        "seconds": round(time.time() - t_start, 1),
    }
    (run_dir / "build.json").write_text(json.dumps(summary, indent=2))
    print(f"[build] {json.dumps(summary, indent=1)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
