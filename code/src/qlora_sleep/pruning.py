"""Expert importance vs utilisation, and pruning on importance.

Utilisation (raw router frequency) is the wrong pruning signal: a rare expert
that carries a fact nothing else knows is low-utilisation and load-bearing, and
pruning it deletes the memory the sleep cycle just built. Importance here is
behavioural: force the candidate expert into the routing as an EXTRA slot (so
nothing is displaced), and measure how much the loss changes. An expert that
helps scores positive; an inert or harmful expert scores ~0 or negative.

The plan's formula, |dL/d(alpha_e)|, is computed as well and reported, but it is
not the pruning metric: it conflates helpful and harmful influence and it is
inflated by an expert whose output is merely large and wrong. Measured at
M1 on the canary fixture: the alpha-gradient ranks a random harmful expert above
an inert one, while the loss delta does not. See `tests/test_importance.py`.
"""

from __future__ import annotations

import torch

from .model import moe_blocks, new_expert_start, routing_mode


def _loss(model, tokenizer, texts: list[str], max_len: int) -> torch.Tensor:
    from .router_warmup import token_batch

    device = next(model.parameters()).device
    input_ids, attn, labels = token_batch(tokenizer, texts, max_len, device)
    out = model(input_ids=input_ids, attention_mask=attn, labels=labels, use_cache=False)
    return out.loss


def expert_importance(
    model,
    tokenizer,
    texts: list[str],
    max_len: int = 256,
    include_alpha_grad: bool = True,
) -> dict:
    """Per-(layer, new expert) importance, utilisation and the alpha-gradient diagnostic.

    Returns tensors shaped (n_layers, k_new).
    """
    blocks = moe_blocks(model)
    n_layers = len(blocks)
    start = new_expert_start(model)
    k_new = blocks[0].num_new_experts
    device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        base_loss = float(_loss(model, tokenizer, texts, max_len))

    gates = [b.expert_gates for b in blocks]
    # The probe differentiates through the alpha gates, which the optimisers never
    # receive; enabling their grad here does not make them trainable.
    for gate in gates:
        gate.requires_grad_(True)

    delta = torch.zeros(n_layers, k_new)
    alpha_grad = torch.zeros(n_layers, k_new)
    for e in range(k_new):
        with torch.no_grad(), routing_mode(force_new_expert=e, record=True) as state:
            forced_loss = float(_loss(model, tokenizer, texts, max_len))
        delta[:, e] = base_loss - forced_loss
        if include_alpha_grad:
            model.zero_grad(set_to_none=True)
            with routing_mode(force_new_expert=e):
                loss = _loss(model, tokenizer, texts, max_len)
            grads = torch.autograd.grad(loss, gates, allow_unused=True)
            for li, g in enumerate(grads):
                if g is not None:
                    alpha_grad[li, e] = float(g.detach().abs()[e])
            model.zero_grad(set_to_none=True)

    # utilisation on the same texts, no intervention
    with torch.no_grad(), routing_mode(record=True) as state:
        _loss(model, tokenizer, texts, max_len)
    util = torch.zeros(n_layers, k_new)
    for li in range(n_layers):
        counts = state.counts[li]
        total = counts.sum().clamp_min(1e-12)
        util[li] = (counts[start:] / total).to(util.dtype)

    return {
        "importance": delta,
        "utilisation": util,
        "alpha_grad": alpha_grad,
        "base_loss": base_loss,
        "new_expert_start": start,
        "k_new": k_new,
    }


def select_prune(importance: torch.Tensor, n_prune: int) -> list[tuple[int, int]]:
    """(layer, expert) pairs with the LOWEST importance, lowest first."""
    flat = importance.flatten()
    if n_prune <= 0:
        return []
    n_prune = min(n_prune, flat.numel())
    order = torch.argsort(flat)[:n_prune]
    k_new = importance.shape[1]
    return [(int(i) // k_new, int(i) % k_new) for i in order]


def select_prune_utilisation(utilisation: torch.Tensor, n_prune: int) -> list[tuple[int, int]]:
    """Control: prune by raw router frequency. Exists to be shown wrong."""
    flat = utilisation.flatten()
    if n_prune <= 0:
        return []
    n_prune = min(n_prune, flat.numel())
    order = torch.argsort(flat)[:n_prune]
    k_new = utilisation.shape[1]
    return [(int(i) // k_new, int(i) % k_new) for i in order]


def prune_experts(model, pairs: list[tuple[int, int]], reset_lora: bool = True) -> int:
    """Neutralise pruned experts: zero the router columns and (optionally) the adapters.

    Base experts are never touched: only indices at or beyond `new_expert_start`.
    """
    blocks = moe_blocks(model)
    start = new_expert_start(model)
    for layer_idx, e in pairs:
        block = blocks[layer_idx]
        expert_idx = start + e
        if expert_idx < start:
            raise AssertionError("refusing to prune a base expert")
        with torch.no_grad():
            block.gate.weight[expert_idx].zero_()
            block.expert_gates[e] = 0.0
        if reset_lora:
            from .canaries import zero_lora_b

            zero_lora_b(block.experts[expert_idx])
    return len(pairs)
