"""Closed-book SQuAD evaluation (SQuAD v1.1 exact-match and token F1).

Scoring convention: normalised exact-match and token F1, best over the reference
answers, no judge model. This is STRICTER than the paper's LLM-judge protocol, so
absolute comparability with the published numbers is approximate and the plan's
re-basing rule (relative deltas) is the primary gate.

The leak probe is the other half of this module: if answering WITH the passage in
context is not clearly better than answering without it, the passage is reaching
the no-context prompt somewhere and every no-context number is inflated.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass

NO_CONTEXT_TEMPLATE = (
    "Answer the question with a short factual answer.\n\n"
    "Question: {question}\nAnswer:"
)
WITH_CONTEXT_TEMPLATE = (
    "Answer the question using the passage below.\n\n"
    "Passage: {context}\n\n"
    "Question: {question}\nAnswer:"
)


def build_prompt(question: str, context: str | None = None) -> str:
    if context is None:
        return NO_CONTEXT_TEMPLATE.format(question=question)
    return WITH_CONTEXT_TEMPLATE.format(context=context, question=question)


# --------------------------------------------------------------------------- #
# SQuAD scoring
# --------------------------------------------------------------------------- #

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = set(string.punctuation)


def normalise_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, golds: list[str]) -> float:
    p = normalise_answer(prediction)
    return float(any(p == normalise_answer(g) for g in golds))


def token_f1(prediction: str, golds: list[str]) -> float:
    p_tokens = normalise_answer(prediction).split()
    best = 0.0
    for gold in golds:
        g_tokens = normalise_answer(gold).split()
        if not p_tokens or not g_tokens:
            best = max(best, float(p_tokens == g_tokens))
            continue
        common = 0
        remaining = list(g_tokens)
        for tok in p_tokens:
            if tok in remaining:
                remaining.remove(tok)
                common += 1
        if common == 0:
            continue
        precision = common / len(p_tokens)
        recall = common / len(g_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def clean_generation(text: str) -> str:
    """First line, no prompt echo, no trailing stop artefacts."""
    line = text.strip().split("\n")[0].strip()
    for marker in ("Question:", "Passage:", "Answer:"):
        if line.startswith(marker):
            line = line[len(marker) :].strip()
    return line


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #


@dataclass
class EvalExample:
    question: str
    answers: list[str]
    context: str
    passage_id: str = ""


def evaluate(generate_fn, examples: list[EvalExample], mode: str = "no_context") -> dict:
    """Score `examples` with an injected `generate_fn(prompts) -> completions`."""
    if mode not in ("no_context", "with_context"):
        raise ValueError(f"unknown eval mode {mode!r}")
    prompts = [
        build_prompt(ex.question, None if mode == "no_context" else ex.context) for ex in examples
    ]
    if not prompts:
        return {"mode": mode, "n": 0, "em": 0.0, "f1": 0.0, "predictions": []}
    raw = generate_fn(prompts)
    if len(raw) != len(prompts):
        raise ValueError(f"generate_fn returned {len(raw)} completions for {len(prompts)} prompts")
    predictions = [clean_generation(text) for text in raw]
    em = sum(exact_match(p, ex.answers) for p, ex in zip(predictions, examples)) / len(examples)
    f1 = sum(token_f1(p, ex.answers) for p, ex in zip(predictions, examples)) / len(examples)
    return {
        "mode": mode,
        "n": len(examples),
        "em": 100.0 * em,
        "f1": 100.0 * f1,
        "predictions": predictions,
    }


PROBE_TEMPLATES = {
    # the gate's own template
    "eval": NO_CONTEXT_TEMPLATE,
    # a bare QA pair, the shape a SQuAD fine-tune would use
    "qa": "Question: {question}\nAnswer:",
    # the shape the dream SFT data actually has ("1. <question>? <answer>")
    "quiz": "1. {question} ",
}


def format_probe(generate_fn, examples: list[EvalExample], templates: dict | None = None) -> dict:
    """Ask the same questions in several prompt formats.

    Separates "the knowledge was never stored" from "the knowledge was stored but
    the prompt the gate uses cannot retrieve it": identical questions, identical
    weights, only the surrounding format changes. A student that answers in the
    training format but not in the eval format has a format defect, not a
    consolidation defect.
    """
    templates = PROBE_TEMPLATES if templates is None else templates
    out: dict = {}
    for name, template in templates.items():
        prompts = [template.format(question=ex.question, context=ex.context) for ex in examples]
        raw = generate_fn(prompts)
        predictions = [clean_generation(text) for text in raw]
        em = sum(exact_match(p, ex.answers) for p, ex in zip(predictions, examples)) / len(examples)
        f1 = sum(token_f1(p, ex.answers) for p, ex in zip(predictions, examples)) / len(examples)
        out[name] = {"em": 100.0 * em, "f1": 100.0 * f1, "predictions": predictions}
    return out


def leak_probe(generate_fn, examples: list[EvalExample], min_delta: float = 10.0) -> dict:
    """with-context accuracy must exceed no-context by `min_delta` points."""
    without = evaluate(generate_fn, examples, "no_context")
    with_ctx = evaluate(generate_fn, examples, "with_context")
    delta = with_ctx["em"] - without["em"]
    return {
        "no_context_em": without["em"],
        "with_context_em": with_ctx["em"],
        "delta": delta,
        "min_delta": min_delta,
        "leak": delta < min_delta,
    }


def model_generate_fn(model, tokenizer, max_new_tokens: int = 16, batch_size: int = 8):
    """Build a `generate_fn` from a causal LM, greedy decoding, first line only."""
    import torch

    device = next(model.parameters()).device
    previous_side = tokenizer.padding_side
    # left padding, otherwise the pad tokens sit between the prompt and the
    # continuation and the generated slice is misaligned for every short row
    tokenizer.padding_side = "left"

    def generate_fn(prompts: list[str]) -> list[str]:
        out_texts: list[str] = []
        model.eval()
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start : start + batch_size]
            enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            with torch.no_grad():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attn,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            prompt_len = input_ids.shape[1]
            for row in generated:
                out_texts.append(tokenizer.decode(row[prompt_len:], skip_special_tokens=True))
        return out_texts

    generate_fn.restore = lambda: setattr(tokenizer, "padding_side", previous_side)
    return generate_fn
