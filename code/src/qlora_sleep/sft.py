"""QLoRA SFT on selected dreams, with the wake context hard-stripped.

The copy-vs-store requirement: the student's training input must never contain
the wake passage. A dream that is a near-copy of the passage would make the
student reproduce the passage (continued pretraining by another name), so
examples are filtered by n-gram containment against the source context and the
survivors are checked again before they reach the optimiser.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .dream import filter_dreams, ngram_containment
from .model import base_router_grad_norm, is_alpha_param, param_groups, set_trainable, zero_base_router_grad
from .router_warmup import token_batch


@dataclass
class SFTConfig:
    steps: int = 30
    lr: float = 1e-4
    batch_size: int = 4
    max_len: int = 256
    grad_clip: float = 1.0
    max_containment: float = 0.5
    ngram_n: int = 8
    seed: int = 0
    train_routers: bool = True
    freeze_base_router: bool = True
    # resume: continue an interrupted aggregate run at the saved step. The step number is
    # absolute, so the example schedule (`_cycled_batch`) is identical to an uninterrupted
    # run of the same length -- a resume is not a new training trajectory.
    start_step: int = 0


# The gate asks a bare question. Training the fact in the shape the gate asks it is
# what makes it retrievable: the dream's own continuation form ("1. <q>? <a>") is a
# format the evaluation never uses, and a model trained only on it answers questions
# by continuing a quiz.
QA_TEMPLATE = "Answer the question with a short factual answer.\n\nQuestion: {question}\nAnswer: {answer}"
QA_PROMPT = "Answer the question with a short factual answer.\n\nQuestion: {question}\nAnswer:"


def steps_for_arm(cfg: dict) -> int:
    """Every arm trains exactly `sft.steps` steps, whatever its data and objective.

    The qa arm used to derive its step count from the pair count
    (`max(4, pairs * epochs // batch)`), which gave it a mean of 6.2 steps against the
    comparators' exact 30 on the E4 frame -- so the arm contrast measured the schedule,
    not the mechanism. A fixed override is the correct step-match; matching the mean with
    an epoch count is not, because the pair count varies per passage (2-31 pairs).
    """
    return int(cfg["sft"]["steps"])


def prepare_qa_examples(
    pairs: list[tuple[str, str]],
    context: str,
    max_containment: float = 0.5,
    n: int = 8,
) -> dict:
    """Turn (question, answer) pairs into masked training examples.

    Each example is `(prompt, answer)` so the loss can be masked to the answer: the
    question is the retrieval key, and spending capacity reproducing it is what made
    the earlier runs memorise dreams without being able to answer anything.
    """
    prompts: list[str] = []
    answers: list[str] = []
    dropped: list[int] = []
    containments: list[float] = []
    for idx, (question, answer) in enumerate(pairs):
        text = QA_TEMPLATE.format(question=question, answer=answer)
        containment = ngram_containment(text, context, n)
        containments.append(containment)
        if containment > max_containment:
            dropped.append(idx)
            continue
        prompts.append(QA_PROMPT.format(question=question))
        answers.append(answer)
    return {
        "pairs": list(zip(prompts, answers)),
        "dropped": dropped,
        "containment": containments,
        "fallback": not prompts,
    }


def prepare_cpt_examples(context: str) -> list[str]:
    """The SFT-on-C comparator: the wake passage itself, whole-text loss, C NOT stripped.

    This is the paper's fine-tuned-no-dreaming / rebuilt-CPT arm, and it is the arm the
    degeneration rule compares against: if the sleep pipeline cannot beat training on
    the passage text directly, the dreams have added nothing. The copy-vs-store filter is
    deliberately bypassed here -- that filter exists so the STUDENT cannot see C on the
    sleep path, and this arm is the control that does see it.
    """
    if not context.strip():
        raise ValueError("CPT arm needs a non-empty passage")
    return [context]


def prepare_sft_examples(
    dreams: list[str],
    context: str,
    max_containment: float = 0.5,
    n: int = 8,
    min_words: int = 8,
) -> dict:
    """Hard filter: drop copies of the context and empty/too-short dreams."""
    result = filter_dreams(dreams, context, max_containment=max_containment, n=n)
    keep: list[str] = []
    for idx in result["kept"]:
        text = dreams[idx]
        if len(text.split()) < min_words:
            continue
        keep.append(text)
    result["examples"] = keep
    result["short_dropped"] = len(result["kept"]) - len(keep)
    result["fallback"] = not keep
    # invariant: nothing that reaches the optimiser still copies the context
    for text in keep:
        c = ngram_containment(text, context, n)
        if c > max_containment:
            raise AssertionError(
                f"context leakage into the SFT set: containment {c:.3f} > {max_containment}"
            )
    return result


def masked_token_batch(tokenizer, pairs: list[tuple[str, str]], max_len: int, device):
    """Tokenise (prompt, answer) pairs with labels masked to the answer.

    Prompt and answer are tokenised separately and concatenated: the boundary is then
    exact, so no prompt token can be scored and no answer token can be masked.
    """
    rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    for prompt, answer in pairs:
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
        ids = (prompt_ids + answer_ids)[:max_len]
        labels = ([-100] * len(prompt_ids) + answer_ids)[:max_len]
        rows.append(ids)
        label_rows.append(labels)
    width = max(len(row) for row in rows)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    input_ids, attn, labels = [], [], []
    for row, label_row in zip(rows, label_rows):
        pad = width - len(row)
        input_ids.append(row + [pad_id] * pad)
        attn.append([1] * len(row) + [0] * pad)
        labels.append(label_row + [-100] * pad)
    return (
        torch.tensor(input_ids, dtype=torch.long, device=device),
        torch.tensor(attn, dtype=torch.long, device=device),
        torch.tensor(labels, dtype=torch.long, device=device),
    )


def train_sft(
    model,
    tokenizer,
    examples: list,
    config: SFTConfig,
    context: str | None = None,
    log=None,
    probe=None,
    probe_every: int = 0,
) -> dict:
    """QLoRA SFT on the prepared examples; trains LoRA and routers, not alpha gates.

    `examples` is either a list of dream strings (loss over the whole text, the
    paper's form) or a list of `(prompt, answer)` pairs (loss masked to the answer,
    the retrievable form).

    `probe` is called every `probe_every` steps under `no_grad` with the model in eval
    mode, and the training mode is restored afterwards, so the probe is a reading and
    not a dynamics change. A probe that returns `{"abort": True}` stops the loop (the
    corpus runner's spend guard: an arm that has destroyed the model is not worth the
    remaining steps, and the destruction is a finding).
    """
    if not examples:
        raise ValueError("no SFT examples survived filtering")
    masked = isinstance(examples[0], tuple)
    if context is not None:
        for example in examples:
            text = f"{example[0]} {example[1]}" if masked else example
            c = ngram_containment(text, context, config.ngram_n)
            if c > config.max_containment:
                raise AssertionError(
                    f"context leakage into the SFT batch: containment {c:.3f}"
                )

    # alpha gates stay grad-enabled as the importance instrument but are NOT handed to
    # the optimiser: `param_groups` excludes them, so the loop never updates them. Their
    # gradients must be zeroed each step by hand, otherwise they accumulate across steps
    # and the per-expert importance reading becomes a stale sum of every previous batch.
    set_trainable(model, lora=True, routers=config.train_routers, alphas=True)
    device = next(model.parameters()).device
    optim = torch.optim.AdamW(param_groups(model, lr_lora=config.lr, lr_router=config.lr))
    alpha_params = [p for name, p in model.named_parameters() if is_alpha_param(name)]
    alpha_accum: list[float] = [0.0] * len(alpha_params)
    model.train()
    torch.manual_seed(config.seed)

    history: list[dict] = []
    probes: list[dict] = []
    aborted = False
    supervised_tokens = 0
    for step in range(config.start_step, config.steps):
        batch = _cycled_batch(examples, step, config.batch_size)
        if masked:
            input_ids, attn, labels = masked_token_batch(tokenizer, batch, config.max_len, device)
        else:
            input_ids, attn, labels = token_batch(tokenizer, batch, config.max_len, device)
        # DOSE: the number of tokens the loss actually scores. Steps are not the dose --
        # an answer-masked qa step scores ~8 tokens where a whole-text step scores ~150,
        # so two arms at 30 steps can differ by ~20x in supervised signal.
        label_tokens = int((labels != -100).sum())
        supervised_tokens += label_tokens
        out = model(input_ids=input_ids, attention_mask=attn, labels=labels, use_cache=False)
        loss = out.loss
        optim.zero_grad(set_to_none=True)
        for p in alpha_params:
            p.grad = None
        loss.backward()
        if config.freeze_base_router:
            # Only the new experts' columns move. The brief's contract is that only new
            # experts train; letting base columns move as well is how a sleep cycle
            # silently becomes a fine-tune that forgets the base distribution.
            base_grad = base_router_grad_norm(model)
            zero_base_router_grad(model)
        else:
            base_grad = base_router_grad_norm(model)
        alpha_step = []
        for i, p in enumerate(alpha_params):
            value = float(p.grad.detach().abs().sum()) if p.grad is not None else 0.0
            alpha_step.append(value)
            alpha_accum[i] += value
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], config.grad_clip
        )
        optim.step()
        row = {
            "step": step,
            "loss": float(loss.detach()),
            "grad_norm": float(grad_norm),
            "base_router_grad_norm": base_grad,
            "alpha_grad": alpha_step,
            # per-step label tokens: the dose sums across a resumed run, where the
            # returned `supervised_tokens` only covers the steps this process trained
            "label_tokens": label_tokens,
        }
        history.append(row)
        if log is not None:
            log(stage="sft_step", **row)
        if probe is not None and probe_every > 0 and (step + 1) % probe_every == 0:
            model.eval()
            try:
                with torch.no_grad():
                    reading = probe(step + 1) or {}
            finally:
                model.train()
            probes.append({"step": step + 1, **reading})
            if log is not None:
                log(stage="sft_probe", **probes[-1])
            if reading.get("abort"):
                aborted = True
                break
    return {
        "history": history,
        "final": history[-1] if history else {},
        "supervised_tokens": supervised_tokens,
        "steps": len(history),
        "probes": probes,
        "aborted": aborted,
        # per-step mean |dL/d alpha_e|: a free, in-training importance reading that the
        # pruning pass can cross-check against its own probe
        "alpha_importance": [v / max(len(history), 1) for v in alpha_accum],
    }


def _cycled_batch(examples: list, step: int, batch_size: int) -> list:
    n = len(examples)
    start = (step * batch_size) % n
    return [examples[(start + i) % n] for i in range(min(batch_size, n))]
