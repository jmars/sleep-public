"""Router warmup and routing telemetry.

The trap this module exists for: OLMoE routes by token-choice top-k. A newly
added expert whose router column is small-initialised is essentially never in
the top-k, so it receives no tokens, so its LoRA receives no gradient. It trains
for zero steps and the evaluation silently reads as SFT noise. Warmup reserves
`forced_slots` of the top-k for new experts and anneals that reservation to
zero, so the router has a chance to learn the new columns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .model import (
    RoutingState,
    base_router_grad_norm,
    moe_blocks,
    new_expert_start,
    param_groups,
    routing_mode,
    set_new_expert_mask,
    set_trainable,
    unlock_new_experts,
    zero_base_router_grad,
)


# --------------------------------------------------------------------------- #
# routing statistics
# --------------------------------------------------------------------------- #


def routing_report(state: RoutingState, new_start: int | None = None) -> dict:
    """Turn a recorded `RoutingState` into per-layer and aggregate statistics."""
    layers: dict[str, dict] = {}
    total_counts = None
    total_probs = None
    n_layers = 0
    for layer_idx in sorted(state.counts):
        counts = state.counts[layer_idx]
        probs = state.prob_sums[layer_idx] / max(state.calls.get(layer_idx, 1), 1)
        acts = state.act_sums[layer_idx] / max(state.calls.get(layer_idx, 1), 1)
        share = None
        if new_start is not None and new_start < counts.numel():
            denom = counts.sum().clamp_min(1e-12)
            share = float(counts[new_start:].sum() / denom)
        layers[str(layer_idx)] = {
            "counts": counts.tolist(),
            "mean_probs": probs.tolist(),
            "act_norms": acts.tolist(),
            "new_share": share,
            "tokens": state.calls.get(layer_idx, 0),
        }
        total_counts = counts if total_counts is None else total_counts + counts
        total_probs = probs if total_probs is None else total_probs + probs
        n_layers += 1
    out: dict = {"layers": layers, "n_layers": n_layers}
    if total_counts is not None:
        out["counts"] = total_counts.tolist()
        out["mean_probs"] = (total_probs / max(n_layers, 1)).tolist()
        if new_start is not None and new_start < total_counts.numel():
            out["new_share"] = float(total_counts[new_start:].sum() / total_counts.sum().clamp_min(1e-12))
    return out


def aggregate_new_share(state: RoutingState, new_start: int) -> float:
    total = 0.0
    denom = 0.0
    for layer_idx, counts in state.counts.items():
        if new_start >= counts.numel():
            continue
        total += float(counts[new_start:].sum())
        denom += float(counts.sum())
    return total / denom if denom > 0 else 0.0


# --------------------------------------------------------------------------- #
# batch helpers
# --------------------------------------------------------------------------- #


def token_batch(tokenizer, texts: list[str], max_len: int, device):
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[attn == 0] = -100
    return input_ids, attn, labels


def lm_loss(model, input_ids, attn, labels) -> torch.Tensor:
    out = model(input_ids=input_ids, attention_mask=attn, labels=labels, use_cache=False)
    return out.loss


def expert_load(model, tokenizer, texts: list[str], max_len: int = 512, device: str = "cpu") -> torch.Tensor:
    """(n_layers, n_experts) routing counts over `texts`."""
    blocks = moe_blocks(model)
    device_t = next(blocks[0].gate.parameters()).device
    model.eval()
    with torch.no_grad(), routing_mode(record=True) as state:
        input_ids, attn, _ = token_batch(tokenizer, texts, max_len, device_t)
        model(input_ids=input_ids, attention_mask=attn, use_cache=False)
    rows = []
    for layer_idx in range(len(blocks)):
        rows.append(state.counts[layer_idx].cpu())
    return torch.stack(rows)


def routing_share(model, tokenizer, texts: list[str], max_len: int = 512) -> dict:
    """New-expert routing share (aggregate and per layer) on `texts`, no grads."""
    start = new_expert_start(model)
    blocks = moe_blocks(model)
    device_t = next(blocks[0].gate.parameters()).device
    model.eval()
    with torch.no_grad(), routing_mode(record=True) as state:
        input_ids, attn, _ = token_batch(tokenizer, texts, max_len, device_t)
        model(input_ids=input_ids, attention_mask=attn, use_cache=False)
    report = routing_report(state, new_start=start)
    report["new_expert_start"] = start
    report["aggregate_new_share"] = aggregate_new_share(state, start)
    return report


# --------------------------------------------------------------------------- #
# share cap: bisect the new-expert router bias into a held-out band
# --------------------------------------------------------------------------- #


class ShareBandUnreachable(RuntimeError):
    """The requested new-expert share band cannot be reached by the router bias.

    This is an INSUFFICIENT result, not a tuning failure: it says the seat exists but
    the band is not reachable on this model, and the caller must escalate rather than
    search the hyper-parameters for longer.
    """


def new_bank_prob_mass(model, input_ids, attention_mask, start: int | None = None) -> torch.Tensor:
    """Differentiable mean softmax probability mass on the NEW router columns.

    Counts (the top-k share used everywhere else) are a discrete selection and carry no
    gradient, so a penalty on them cannot train anything. Probability MASS is
    differentiable through the softmax, and it is the quantity a share cap is about:
    "how much of every token's routing goes to the new bank".

    Hooks the router Linear rather than re-implementing the MoE forward, so the mask is
    applied exactly as `_moe_forward` applies it and nothing about the forward pass is
    duplicated.
    """
    blocks = moe_blocks(model)
    if start is None:
        start = new_expert_start(model)
    masses: list[torch.Tensor] = []
    handles = []
    for block in blocks:
        def hook(module, args, output, block=block):
            logits = output
            mask = getattr(block, "router_mask", None)
            if mask is not None:
                logits = logits + mask
            probs = F.softmax(logits.float(), dim=-1)
            masses.append(probs[:, start:].sum(dim=-1).mean())

        handles.append(block.gate.register_forward_hook(hook))
    try:
        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if not masses:
        raise ValueError("no MoE router fired; is the model patched and expanded?")
    return torch.stack(masses).mean()


def tune_new_share(
    model,
    tokenizer,
    texts: list[str],
    band: tuple[float, float] = (0.05, 0.09),
    lo: float = -12.0,
    hi: float = 0.0,
    max_iters: int = 8,
    max_len: int = 512,
    log=None,
) -> dict:
    """Bisect the new-expert router bias until the held-out routing share is in `band`.

    The pathology this addresses is measured: the new bank holds 0.24-0.27 of every
    input's top-k slots against a 0.059 uniform null, so whatever the SFT stores is
    applied globally and overrides the base answer instead of adding to it. The mask
    buffer is an additive bias on the new columns, so it is the one control that caps
    the bank's share without touching its weights -- and it stays applied through SFT
    and evaluation, so there is no train/eval routing skew.

    `texts` must be HELD-OUT text (never trained on): the bias is the only routing
    control left after the routers are frozen, so it has to be tuned on text the cycle
    cannot have memorised, or the band would be met by memorisation.

    Raises `ShareBandUnreachable` when the band cannot be reached within `max_iters`.
    """
    if not texts:
        raise ValueError("share tuning needs held-out texts")
    low, high = band
    share = routing_share(model, tokenizer, texts, max_len=max_len)["aggregate_new_share"]
    history: list[dict] = [{"mask_value": hi, "share": share, "note": "unbiased"}]
    if log is not None:
        log(stage="share_tune", **history[0])
    if low <= share <= high:
        return {"mask_value": hi, "share": share, "reached": True, "iterations": 0,
                "band": [low, high], "history": history}

    for iteration in range(max_iters):
        mid = 0.5 * (lo + hi)
        set_new_expert_mask(model, mid)
        share = routing_share(model, tokenizer, texts, max_len=max_len)["aggregate_new_share"]
        row = {"mask_value": mid, "share": share, "iteration": iteration}
        history.append(row)
        if log is not None:
            log(stage="share_tune", **row)
        if low <= share <= high:
            return {"mask_value": mid, "share": share, "reached": True,
                    "iterations": iteration + 1, "band": [low, high], "history": history}
        # a more negative bias lowers the share: `lo` is the too-low side
        if share > high:
            hi = mid
        else:
            lo = mid

    set_new_expert_mask(model, 0.0)
    raise ShareBandUnreachable(
        f"new-expert share band {band} unreachable by router bias in {max_iters} bisection "
        f"steps; readings {[round(row['share'], 4) for row in history]}"
    )


# --------------------------------------------------------------------------- #
# warmup
# --------------------------------------------------------------------------- #


def warmup_schedule(step: int, warmup_steps: int, forced_slots: int) -> int:
    """Linear anneal of the number of reserved slots to zero at `warmup_steps`."""
    if warmup_steps <= 0 or step >= warmup_steps:
        return 0
    frac = 1.0 - step / warmup_steps
    return int(math.ceil(forced_slots * frac - 1e-9))


@dataclass
class WarmupConfig:
    steps: int = 25
    forced_slots: int = 2
    lr: float = 1e-3
    max_len: int = 256
    batch_size: int = 1
    seed: int = 0
    # W2 contrastive arm: negative (distractor) batches penalise the new bank's
    # probability mass above `share_target`, so the router is told NOT-HERE as well as
    # HERE. With no distractor texts the warmup is the W1 arm (forced inclusion only).
    distractor_texts: list[str] = field(default_factory=list)
    share_target: float = 0.07
    penalty_weight: float = 1.0


def run_router_warmup(
    model,
    tokenizer,
    texts: list[str],
    config: WarmupConfig,
    device: str = "cpu",
    log=None,
) -> dict:
    """Train the routers (only) with forced inclusion annealing to zero.

    Order matters: the router mask is lifted FIRST (so the new columns have real
    probabilities and therefore real gradients), then `forced_slots` of the top-k
    slots are reserved for the highest-probability new experts, annealed linearly to
    zero over the warmup. After the last step the new experts compete on merit.

    LoRA and the alpha gates are frozen for the whole warmup, so the only thing
    that can change is which experts the router picks.

    `texts` is the POSITIVE set -- the passage's question/answer pairs, which is the
    shape the evaluation asks in. Warming on dream prose instead teaches the new
    columns to attract on text the gate never sees. With `distractor_texts` set, each
    step also takes one negative batch (held-out prompts and prose) and penalises the
    new bank's probability mass above `share_target`: the W2 arm.
    """
    if not texts:
        raise ValueError("warmup needs at least one text")
    set_trainable(model, lora=False, routers=True, alphas=False)
    start = new_expert_start(model)
    # Unmask before training: a masked column's probability is ~0, its mixture
    # weight is ~0 and its gradient is ~0, so forcing a masked expert in would train
    # nothing at all.
    unlock_new_experts(model)
    blocks = moe_blocks(model)
    device_t = next(blocks[0].gate.parameters()).device
    optim = torch.optim.AdamW(param_groups(model, lr_lora=config.lr, lr_router=config.lr))
    model.train()
    torch.manual_seed(config.seed)

    history: list[dict] = []
    for step in range(config.steps):
        forced = warmup_schedule(step, config.steps, config.forced_slots)
        chunk = texts[step % len(texts) :][: config.batch_size] or texts[:1]
        input_ids, attn, labels = token_batch(tokenizer, chunk, config.max_len, device_t)
        with routing_mode(forced_slots=forced, record=True) as state:
            loss = lm_loss(model, input_ids, attn, labels)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        # Base columns stay exactly as the frozen base left them: the warmup exists to
        # teach NEW experts to be selected, not to relearn the base router. Without this
        # line a 25-step warmup at lr 1e-3 moves base router weights by ~2.5% absolute
        # per element and collapses no-context accuracy (measured: EM 20 -> 0).
        zero_base_router_grad(model)
        share_pos = aggregate_new_share(state, start)

        share_neg = None
        penalty = 0.0
        if config.distractor_texts and config.penalty_weight > 0:
            neg_chunk = (config.distractor_texts[step % len(config.distractor_texts):]
                         [: config.batch_size]) or config.distractor_texts[:1]
            neg_ids, neg_attn, _ = token_batch(tokenizer, neg_chunk, config.max_len, device_t)
            # record=True as well: the negative batch's share is the reading that says
            # whether the penalty did anything, and it is measured on the same forward.
            with routing_mode(record=True) as neg_state:
                mass = new_bank_prob_mass(model, neg_ids, neg_attn, start=start)
            # relu, so text the router already routes AWAY from the new bank contributes
            # nothing: the penalty is a cap on over-attraction, not a pressure to keep
            # the bank cold everywhere.
            pen = config.penalty_weight * torch.relu(mass - config.share_target)
            pen.backward()
            zero_base_router_grad(model)
            penalty = float(pen.detach())
            share_neg = aggregate_new_share(neg_state, start)

        # The diagnostic that matters: the gradient on the NEW router columns. The
        # overall grad norm also moves when only base columns move, so it cannot
        # distinguish a working warmup from a broken one.
        new_router_grad = 0.0
        for block in blocks:
            g = block.gate.weight.grad
            if g is not None:
                new_router_grad += float(g[start:].detach().pow(2).sum())
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optim.step()
        row = {
            "step": step,
            "forced_slots": forced,
            "router_mask": 0.0,
            "loss": float(loss.detach()),
            "grad_norm": float(grad_norm),
            "new_router_grad_norm": math.sqrt(new_router_grad),
            "base_router_grad_norm": base_router_grad_norm(model),
            "new_share": share_pos,
            "share_pos": share_pos,
            "share_neg": share_neg,
            "penalty": penalty,
        }
        history.append(row)
        if log is not None:
            log(stage="warmup_step", **row)
    unlock_new_experts(model)
    return {"history": history, "final": history[-1] if history else {}}
