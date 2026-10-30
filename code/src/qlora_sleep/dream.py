"""Dream generation, gradient-based scoring/selection, and the literalism instrument.

Literalism is the failure mode where dreams are near-copies of the wake context
C: training the student on them rebuilds continued pretraining rather than
transferring a distilled signal. It is measured, not assumed -- every cycle
publishes the containment distribution, and the pass/fail reading is the plan's
alarm threshold (median containment > 0.5 = ALARM, 0.3-0.5 = INSUFFICIENT).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import torch

from .model import perturb_summary, routing_mode

PROMPT_VARIANTS: dict[str, str] = {
    "recap": (
        "Write a short factual recap of the key facts in the passage below, as notes to yourself.\n\n"
        "Passage: {context}\n\nRecap:"
    ),
    "quiz": (
        "Write three question-and-answer pairs about the passage below. "
        "Use the exact names, numbers and places from the passage.\n\n"
        "Passage: {context}\n\nQuestions and answers:"
    ),
    "facts": (
        "List the facts stated in the passage below as short standalone sentences.\n\n"
        "Passage: {context}\n\nFacts:"
    ),
}

_WORD_RE = re.compile(r"[a-z0-9']+")


# --------------------------------------------------------------------------- #
# literalism instrument
# --------------------------------------------------------------------------- #


def words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def ngram_set(text: str, n: int = 8) -> set[tuple[str, ...]]:
    toks = words(text)
    if len(toks) < n:
        return set()
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def ngram_containment(dream: str, context: str, n: int = 8) -> float:
    """Fraction of the dream's word n-grams that also occur in `context`.

    Convention: a dream shorter than n words has no n-grams and scores 0.0
    (nothing to copy), which is reported rather than silently treated as novel.
    """
    dream_ngrams = ngram_set(dream, n)
    if not dream_ngrams:
        return 0.0
    ctx_ngrams = ngram_set(context, n)
    if not ctx_ngrams:
        return 0.0
    return len(dream_ngrams & ctx_ngrams) / len(dream_ngrams)


def distinct_n(texts: list[str], n: int = 2) -> float:
    """Unique n-grams over total n-grams across `texts` (0.0 when empty)."""
    grams: list[tuple[str, ...]] = []
    for text in texts:
        toks = words(text)
        if len(toks) < n:
            continue
        grams.extend(tuple(toks[i : i + n]) for i in range(len(toks) - n + 1))
    if not grams:
        return 0.0
    return len(set(grams)) / len(grams)


def dream_report(dreams: list[str], context: str, n: int = 8) -> dict:
    containments = [ngram_containment(d, context, n) for d in dreams]
    lengths = [len(words(d)) for d in dreams]
    return {
        "n_dreams": len(dreams),
        "containment": containments,
        "containment_median": _median(containments),
        "containment_mean": sum(containments) / len(containments) if containments else 0.0,
        "containment_max": max(containments) if containments else 0.0,
        "distinct_2": distinct_n(dreams, 2),
        "distinct_3": distinct_n(dreams, 3),
        "mean_words": sum(lengths) / len(lengths) if lengths else 0.0,
        "duplicate_rate": _duplicate_rate(dreams),
    }


_STOP_WORDS = frozenset(
    "a an the and or of to in on at for with by from as is are was were be been being "
    "this that these those it its he she they them his her their who whom which what "
    "when where why how not no but if then than so such also into over under after "
    "before during there here we you i one two first".split()
)


def content_words(text: str) -> set[str]:
    """Distinct content words: the word tokeniser minus stop words and 1-2 letter tokens."""
    return {w for w in words(text) if w not in _STOP_WORDS and len(w) > 2}


def content_overlap(dream: str, context: str) -> float:
    """Fraction of the dream's CONTENT words that occur in `context`.

    This is the prose-grounding instrument. `answer_support_rate` reads extracted
    question/answer pairs, so it is structurally blind to a prose dream variant (measured:
    recap and facts yield 0.0-0.1 pairs per passage). Without a prose reading, a
    grounded paraphrase and an off-topic hallucination both score 0.0 and look identical.

    Convention: a dream with no content words scores 0.0, reported as "unreadable",
    not as "ungrounded" -- the same distinction `verdict()` makes for NO_PAIRS.
    """
    dream_words = content_words(dream)
    if not dream_words:
        return 0.0
    return len(dream_words & content_words(context)) / len(dream_words)


def prose_grounding(dreams: list[str], context: str) -> dict:
    """Content-overlap distribution of `dreams` against `context` (per dream)."""
    overlaps = [content_overlap(d, context) for d in dreams]
    return {
        "n": len(overlaps),
        "median": _median(overlaps),
        "mean": sum(overlaps) / len(overlaps) if overlaps else 0.0,
        "min": min(overlaps) if overlaps else 0.0,
        "max": max(overlaps) if overlaps else 0.0,
    }


def literalism_verdict(median_containment: float, alarm: float = 0.5, insufficient: float = 0.3) -> str:
    if median_containment > alarm:
        return "ALARM"
    if median_containment >= insufficient:
        return "INSUFFICIENT"
    return "OK"


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _duplicate_rate(dreams: list[str]) -> float:
    if not dreams:
        return 0.0
    normalised = [" ".join(words(d)) for d in dreams]
    return 1.0 - len(set(normalised)) / len(normalised)


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #


@dataclass
class DreamConfig:
    n_dreams: int = 8
    temperature: float = 0.9
    top_p: float = 0.95
    max_new_tokens: int = 256
    prompt_variant: str = "quiz"
    random_expert: bool = True
    perturb_slots: int = 0
    """Replace this many standard top-k slots with non-top-k experts while dreaming.

    Dreaming under router perturbation: the teacher's expert set is forced off the
    standard top-k on every generated token, so the dream text is written by a different
    region of the expert bank than the one that will read it back at eval. Strength is
    logged per dream pass (`generate_dreams(..., stats=...)`), never assumed.
    """
    seed: int = 0


def build_prompt(context: str, variant: str = "quiz") -> str:
    if variant not in PROMPT_VARIANTS:
        raise KeyError(f"unknown prompt variant {variant!r}; have {sorted(PROMPT_VARIANTS)}")
    return PROMPT_VARIANTS[variant].format(context=context)


def generate_dreams(
    model,
    tokenizer,
    context: str,
    config: DreamConfig,
    stats: dict | None = None,
) -> list[str]:
    """Sample `config.n_dreams` dreams conditioned on `context`.

    `stats`, when given, is filled with the routing perturbation strength actually
    applied during this pass (`perturb_summary`): the run logs it per passage rather than
    trusting the configured slot count.
    """
    prompt = build_prompt(context, config.prompt_variant)
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"].repeat(config.n_dreams, 1)
    attn = enc["attention_mask"].repeat(config.n_dreams, 1)
    torch.manual_seed(config.seed)
    with torch.no_grad(), routing_mode(
        random_expert=config.random_expert, perturb_slots=config.perturb_slots
    ) as state:
        if config.random_expert or config.perturb_slots > 0:
            state.generator = torch.Generator(device=device).manual_seed(config.seed)
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            do_sample=True,
            temperature=config.temperature,
            top_p=config.top_p,
            max_new_tokens=config.max_new_tokens,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    if stats is not None:
        stats.update(perturb_summary(state))
    prompt_len = input_ids.shape[1]
    dreams = []
    for row in out:
        text = tokenizer.decode(row[prompt_len:], skip_special_tokens=True).strip()
        dreams.append(text)
    return dreams


# --------------------------------------------------------------------------- #
# gradient-based scoring and selection
# --------------------------------------------------------------------------- #


MIN_SCORABLE_TOKENS = 2


def scorable_dreams(
    dreams: list[str], tokenizer, min_tokens: int = MIN_SCORABLE_TOKENS
) -> list[int]:
    """Indices of the dreams the scorer can score.

    A dream that tokenises to fewer than `min_tokens` ids has no next-token target, so
    the gradient score is undefined for it. The recap variant emits such one-line
    dreams occasionally (measured: 17 words mean, but a tail down to a bare fragment),
    and the FIRST full recap arm died on one of them at the scorer. The floor is
    applied HERE, before scoring, so a sub-floor dream never enters the candidate pool
    and the caller can log how many were dropped -- a silently swallowed scorer error
    would hide both the count and the fact that selection ran on a different pool.
    """
    return [
        i
        for i, dream in enumerate(dreams)
        if len(tokenizer(dream, add_special_tokens=True)["input_ids"]) >= min_tokens
    ]


def _sft_loss(model, tokenizer, text: str, max_len: int) -> torch.Tensor:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
    device = next(model.parameters()).device
    input_ids = enc["input_ids"].to(device)
    if input_ids.shape[1] < 2:
        raise ValueError("dream too short to score")
    out = model(input_ids=input_ids, labels=input_ids, use_cache=False)
    return out.loss


def score_dreams(
    model,
    tokenizer,
    dreams: list[str],
    max_len: int = 256,
    trainable: list[torch.nn.Parameter] | None = None,
) -> list[float]:
    """Importance score per dream: L2 norm of the SFT gradient wrt the trainable set."""
    if trainable is None:
        trainable = [p for p in model.parameters() if p.requires_grad]
    params = [p for p in trainable if p.requires_grad]
    if not params:
        raise ValueError("no trainable parameters to differentiate against")
    scores: list[float] = []
    for dream in dreams:
        model.zero_grad(set_to_none=True)
        try:
            loss = _sft_loss(model, tokenizer, dream, max_len)
        except ValueError:
            # BACKSTOP ONLY. The selection path removes sub-floor dreams before it calls
            # this (scorable_dreams), so reaching here means the caller skipped that
            # filter; the dream is scored 0.0 so one bad sample cannot kill an arm, and
            # the caller's n_too_short count is what makes the drop visible.
            scores.append(0.0)
            continue
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        total = 0.0
        for g in grads:
            if g is not None:
                total += float(g.detach().pow(2).sum())
        scores.append(math.sqrt(total))
    model.zero_grad(set_to_none=True)
    return scores


def select_dreams(
    scores: list[float], k: int, b: int, seed: int = 0
) -> list[int]:
    """Top-k by score plus b random from the remainder (paper's diversity term)."""
    n = len(scores)
    if n == 0:
        return []
    k = min(k, n)
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    chosen = order[:k]
    rest = [i for i in order[k:]]
    if b > 0 and rest:
        gen = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(rest), generator=gen).tolist()
        chosen.extend(rest[i] for i in perm[: min(b, len(rest))])
    return sorted(set(chosen))


# --------------------------------------------------------------------------- #
# question/answer extraction
# --------------------------------------------------------------------------- #

# The dream prompts ask for question-and-answer pairs, and a 1B-active model emits
# them in several shapes: "1. <question>? <answer>", "1. <question>?\nAnswer: <answer>",
# "Question: <question> Answer: <answer>". The knowledge in a dream is carried by the
# pair, not by the dream's continuation form: training on the whole dream string
# teaches the model to continue a quiz, and a continuation is not retrievable from a
# bare question. Extraction keeps only the (question, answer) mapping.
_NUMBERED_QA = re.compile(
    r"(?:^|\n)\s*\d+\s*[.)]\s*(?P<q>[^\n?]{3,200}\?)\s*(?P<a>[^\n]{1,200})",
)
# "1. <question>?\nAnswer: <answer>" puts the label INSIDE the numbered capture, so it
# has to be stripped from the answer rather than matched by a second pattern.
_ANSWER_LABEL = re.compile(r"^(?:A|Ans|Answer|Output)\s*[.:]\s*", re.IGNORECASE)
_INLINE_QA = re.compile(
    r"(?:^|\n)\s*(?:Q:|Question:)\s*(?P<q>[^\n?]{3,200}\?)\s*"
    r"(?:A:|Answer:|Ans:|Output:)\s*(?P<a>[^\n]{1,200})",
)


# A question list with no answers ("1. Who is the first?\n2. Who is the last?") is a
# shape this model produces often, and the numbered pattern would read the NEXT
# QUESTION as the answer. Measured on real dreams: without this guard 17 of 43
# "answers" in one cycle were other questions. An answer that is itself a question,
# or that starts with a list marker, is not an answer.
_QUESTIONLIKE_ANSWER = re.compile(r"^\s*(?:\d+\s*[.)]|[-*])\s*")
_ANSWER_IS_QUESTION = re.compile(r"\?")


def _is_answer_like(answer: str) -> bool:
    if _QUESTIONLIKE_ANSWER.match(answer):
        return False
    if _ANSWER_IS_QUESTION.search(answer):
        return False
    return True


def extract_qa_pairs(dream: str, max_answer_words: int = 12, min_question_words: int = 3) -> list[tuple[str, str]]:
    """Pull (question, answer) pairs out of a dream, in either of the two shapes.

    Only short answers are kept: an answer longer than a phrase is the model
    continuing prose, not stating a fact, and a long answer cannot match a SQuAD
    reference. Answers that are themselves questions are dropped (a question list
    without answers yields no pairs, which is the honest reading). Duplicates are
    dropped so a repeated pair is not counted twice.
    """
    pairs: list[tuple[str, str]] = []
    for pattern in (_INLINE_QA, _NUMBERED_QA):
        for match in pattern.finditer(dream):
            question = " ".join(match.group("q").split())
            answer = " ".join(match.group("a").split())
            answer = _ANSWER_LABEL.sub("", answer).strip(" .;:-\t")
            if not answer:
                continue
            if len(answer.split()) > max_answer_words:
                continue
            if not _is_answer_like(answer):
                continue
            if len(question.split()) < min_question_words:
                continue
            if not question.endswith("?"):
                question += "?"
            pairs.append((question, answer))
    seen: set[tuple[str, str]] = set()
    unique: list[tuple[str, str]] = []
    for pair in pairs:
        key = (pair[0].lower(), pair[1].lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(pair)
    return unique


def answer_support_rate(pairs: list[tuple[str, str]], context: str) -> float:
    """Fraction of extracted answers whose text occurs in the wake passage.

    This is a MEASUREMENT, deliberately not a filter: using C to keep only supported
    pairs would give the sleep stage access to the wake passage and void the copy-vs-
    store claim the pipeline rests on. It is reported so a zero gain can be attributed
    -- a teacher whose facts are 30% grounded cannot inject more than 30% real facts,
    whatever the SFT does with them.
    """
    if not pairs:
        return 0.0
    ctx_words = " ".join(words(context))
    supported = 0
    for _, answer in pairs:
        answer_words = " ".join(words(answer))
        if answer_words and answer_words in ctx_words:
            supported += 1
    return supported / len(pairs)


def extract_qa_from_dreams(dreams: list[str], max_answer_words: int = 12) -> list[tuple[str, str]]:
    """(question, answer) pairs across a list of dreams, de-duplicated across dreams."""
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for dream in dreams:
        for question, answer in extract_qa_pairs(dream, max_answer_words=max_answer_words):
            key = (question.lower(), answer.lower())
            if key in seen:
                continue
            seen.add(key)
            out.append((question, answer))
    return out


# --------------------------------------------------------------------------- #
# copy-vs-store filter
# --------------------------------------------------------------------------- #


def filter_dreams(
    dreams: list[str], context: str, max_containment: float = 0.5, n: int = 8
) -> dict:
    """Drop dreams that copy the wake context too closely.

    Returns the kept indices, their texts, the dropped indices and the scores, so
    the caller can report the distribution rather than only the survivors.
    """
    containments = [ngram_containment(d, context, n) for d in dreams]
    kept = [i for i, c in enumerate(containments) if c <= max_containment]
    dropped = [i for i, c in enumerate(containments) if c > max_containment]
    return {
        "kept": kept,
        "dropped": dropped,
        "containment": containments,
        "kept_texts": [dreams[i] for i in kept],
        "dropped_texts": [dreams[i] for i in dropped],
        "fallback": not kept,
    }
