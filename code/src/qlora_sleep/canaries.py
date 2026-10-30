"""Canary fixtures: synthetic experts with known ground-truth importance.

Two shapes the pruning instrument must separate:

* RARE-BUT-CRITICAL: carries a fact nothing else knows, but the router almost
  never selects it. Utilisation says prune; importance says keep.
* FREQUENT-BUT-INERT: selected all the time, contributes nothing. Utilisation
  says keep; importance says prune.

These are also the shapes used on the GPU canary run at M7.
"""

from __future__ import annotations

import torch

from .model import moe_blocks, new_expert_start, routing_mode
from .router_warmup import lm_loss, token_batch

CANARY_FACTS = [
    "the access code for vault seven is four eight two one",
    "vault seven opens with the access code four eight two one",
    "the code four eight two one unlocks vault seven",
    "remember the access code for vault seven four eight two one",
]


def canary_texts(n: int = 4) -> list[str]:
    return [CANARY_FACTS[i % len(CANARY_FACTS)] for i in range(n)]


def zero_lora_b(module) -> None:
    """Zero the LoRA B matrices of an expert (peft keeps lora_B as an nn.Linear)."""
    for name in ("gate_proj", "up_proj", "down_proj"):
        mod = getattr(module, name)
        if hasattr(mod, "lora_B"):
            with torch.no_grad():
                mod.lora_B["default"].weight.zero_()


def _expert_params(model, expert_idx: int) -> list[torch.nn.Parameter]:
    tag = f".experts.{expert_idx}."
    return [p for name, p in model.named_parameters() if "lora_" in name and tag in name]


def install_canary_expert(
    model,
    tokenizer,
    texts: list[str],
    expert_offset: int = 0,
    steps: int = 60,
    lr: float = 1e-2,
    max_len: int = 64,
    seed: int = 0,
) -> dict:
    """Train ONE new expert (its LoRA, all layers) to be load-bearing on `texts`.

    Training uses forced inclusion so the expert actually receives tokens; that is
    also the convention `expert_importance` probes under, so the measurement is
    taken at the scale the expert was trained at.
    """
    start = new_expert_start(model)
    params = _expert_params(model, start + expert_offset)
    if not params:
        raise ValueError(f"no LoRA parameters on new expert offset {expert_offset}")
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    optim = torch.optim.AdamW(params, lr=lr)
    device = next(model.parameters()).device
    torch.manual_seed(seed)
    model.train()
    losses: list[float] = []
    for step in range(steps):
        text = texts[step % len(texts)]
        input_ids, attn, labels = token_batch(tokenizer, [text], max_len, device)
        with routing_mode(force_new_expert=expert_offset):
            loss = lm_loss(model, input_ids, attn, labels)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        losses.append(float(loss.detach()))
    model.eval()
    return {"losses": losses, "first": losses[0] if losses else None, "last": losses[-1] if losses else None}


def set_expert_mask(model, expert_offset: int, value: float) -> None:
    """Add `value` to one new expert's router logit (a per-expert router mask)."""
    for block in moe_blocks(model):
        with torch.no_grad():
            block.router_mask[block.new_expert_start + expert_offset] = value


def silence_expert(model, expert_offset: int, value: float = -50.0) -> None:
    """Make the router essentially never select this new expert.

    Driven by the per-expert router mask, not by a constant gate row: a constant
    row's logit is `const * sum(h)`, which is sign-dependent and therefore not a
    reliable way to silence or boost anything.
    """
    start = new_expert_start(model)
    for block in moe_blocks(model):
        with torch.no_grad():
            block.gate.weight[start + expert_offset].zero_()
    set_expert_mask(model, expert_offset, value)


def boost_expert(model, expert_offset: int, value: float = 20.0) -> None:
    """Make the router select this new expert almost always."""
    set_expert_mask(model, expert_offset, value)


def inert_expert(model, expert_offset: int) -> None:
    """Zero the expert's output path so it contributes exactly nothing."""
    start = new_expert_start(model)
    for block in moe_blocks(model):
        expert = block.experts[start + expert_offset]
        with torch.no_grad():
            gate = getattr(expert.gate_proj, "base_layer", expert.gate_proj)
            gate.weight.zero_()
            zero_lora_b(expert)


def dead_expert(model, expert_offset: int) -> None:
    """Silenced AND inert: the shape a broken sleep cycle would leave behind."""
    silence_expert(model, expert_offset)
    inert_expert(model, expert_offset)
