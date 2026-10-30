"""Risk (a), the router-warmup trap, as a fail-able asymmetry.

WITHOUT warmup a new expert is masked out of token-choice top-k, so it is never
selected and its LoRA receives exactly no gradient: it trains for zero steps
while the run looks healthy. WITH warmup (unmask, then reserve top-k slots) the
same expert gets tokens and a real gradient on the first step.

Both the LoRA gradient AND the router-column gradient are asserted: a warmup that
only moves the expert's LoRA while its router column stays at its initialisation
is the same silent failure one step later -- the expert would be trained and then
never selected once the reservation anneals away.
"""

from __future__ import annotations

import math

from qlora_sleep.model import (
    check_new_experts_only,
    lora_module_indices,
    moe_blocks,
    new_expert_start,
    routing_mode,
    set_new_expert_mask,
    set_trainable,
)
from qlora_sleep.router_warmup import aggregate_new_share, lm_loss, token_batch
from qlora_sleep.sft import SFTConfig, train_sft


def _new_expert_grad(
    model, tokenizer, text: str, forced_slots: int = 0, unlocked: bool = False
) -> tuple[float, float, float]:
    """(LoRA grad norm, new router-column grad norm, new-expert routing share)."""
    start = new_expert_start(model)
    if unlocked:
        set_new_expert_mask(model, 0.0)
    set_trainable(model, lora=True, routers=True, alphas=True)
    model.zero_grad(set_to_none=True)
    device = next(model.parameters()).device
    input_ids, attn, labels = token_batch(tokenizer, [text], 64, device)
    with routing_mode(forced_slots=forced_slots, record=True) as state:
        loss = lm_loss(model, input_ids, attn, labels)
    loss.backward()
    total = 0.0
    for name, param in model.named_parameters():
        if "lora_" in name and f".experts.{start}." in name and param.grad is not None:
            total += float(param.grad.detach().pow(2).sum())
    router = 0.0
    for block in moe_blocks(model):
        grad = block.gate.weight.grad
        if grad is not None:
            router += float(grad[start:].detach().pow(2).sum())
    share = aggregate_new_share(state, start)
    model.zero_grad(set_to_none=True)
    return math.sqrt(total), math.sqrt(router), share


def test_lora_lands_on_new_experts_only(expanded_model):
    indices = lora_module_indices(expanded_model)
    start = new_expert_start(expanded_model)
    assert indices == [start, start + 1]
    check_new_experts_only(expanded_model, n_base=start, k_new=2)
    # and the guard actually refuses a leak onto a base expert
    try:
        check_new_experts_only(expanded_model, n_base=start, k_new=3)
    except AssertionError:
        pass
    else:
        raise AssertionError("check_new_experts_only accepted an incomplete adapter set")


def test_no_gradient_without_warmup(expanded_model, tokenizer):
    text = "the access code for vault seven is four eight two one"
    lora, router, share = _new_expert_grad(expanded_model, tokenizer, text, forced_slots=0)
    assert share < 0.02, f"cold new-expert routing share should be ~0, got {share}"
    assert lora < 1e-7, f"cold new-expert LoRA grad should be ~0, got {lora}"
    assert router < 1e-7, f"cold new-expert router grad should be ~0, got {router}"


def test_warmup_restores_gradient(expanded_model, tokenizer):
    text = "the access code for vault seven is four eight two one"
    lora, router, share = _new_expert_grad(expanded_model, tokenizer, text, forced_slots=1, unlocked=True)
    assert share > 0.3, f"forced inclusion should lift the share, got {share}"
    assert lora > 1e-6, f"warmup step should produce a real LoRA gradient, got {lora}"
    assert router > 1e-9, f"warmup did not reach the router columns: {router}"


def test_gradient_asymmetry(expanded_model, tokenizer):
    text = "vault seven opens with the access code four eight two one"
    cold, cold_router, cold_share = _new_expert_grad(expanded_model, tokenizer, text, forced_slots=0)
    warm, warm_router, warm_share = _new_expert_grad(
        expanded_model, tokenizer, text, forced_slots=1, unlocked=True
    )
    assert warm > 1e3 * cold + 1e-8, f"no LoRA asymmetry: cold={cold} warm={warm}"
    assert warm_router > 1e3 * cold_router + 1e-12, (
        f"no router asymmetry: cold={cold_router} warm={warm_router}"
    )
    assert warm_share > 10 * cold_share, f"share did not move: cold={cold_share} warm={warm_share}"


def test_forced_inclusion_without_unmasking_trains_nothing(expanded_model, tokenizer):
    """The trap one step later: forced slots over a masked column are a no-op.

    A masked new expert has probability ~0, so even when its slot is forced its
    mixture weight is ~0, its output is ~0 and its gradient is ~0. Warmup must
    unlock the mask before it forces slots in; this test pins that order.
    """
    text = "the access code for vault seven is four eight two one"
    lora, router, share = _new_expert_grad(expanded_model, tokenizer, text, forced_slots=1, unlocked=False)
    assert share > 0.3, f"the slot was forced but the share did not lift: {share}"
    assert lora < 1e-7, f"masked forced expert should not train its LoRA, got {lora}"
    assert router < 1e-7, f"masked forced expert should not train its router, got {router}"


def test_alphas_are_frozen_by_the_optimiser(expanded_model):
    from qlora_sleep.model import param_groups

    groups = param_groups(expanded_model, lr_lora=1e-4, lr_router=1e-3)
    alpha_ids = {
        id(p)
        for name, p in expanded_model.named_parameters()
        if name.endswith("expert_gates")
    }
    assert alpha_ids, "no alpha gates found"
    for group in groups:
        for param in group["params"]:
            assert id(param) not in alpha_ids, "alpha gate leaked into the optimiser"


def test_sft_leaves_base_router_columns_untouched(expanded_model, tokenizer):
    """The SFT stage must move the new columns only.

    The brief's contract is that only new experts train. Training base router
    columns as well is how the cycle silently becomes a fine-tune that forgets the
    base distribution -- measured on the real model as no-context EM 20 -> 0 after a
    single cycle. The pre-fix failure mode this catches is `freeze_base_router=False`:
    the same call then moves base columns and the assertion fires.
    """
    start = new_expert_start(expanded_model)
    set_new_expert_mask(expanded_model, 0.0)
    before = {name: p.detach().clone() for name, p in expanded_model.named_parameters()}
    config = SFTConfig(steps=2, lr=1e-2, batch_size=1, max_len=32, seed=0)
    train_sft(expanded_model, tokenizer, ["the vault code is four eight two one"], config)
    base_moved, new_moved, lora_moved = [], [], []
    for name, param in expanded_model.named_parameters():
        delta = float((param.detach() - before[name]).abs().max())
        if "gate.weight" in name and param.shape[0] > start:
            if float((param.detach()[:start] - before[name][:start]).abs().max()) > 0:
                base_moved.append(name)
            if float((param.detach()[start:] - before[name][start:]).abs().max()) > 0:
                new_moved.append(name)
        elif "lora_" in name and delta != 0.0:
            lora_moved.append(name)
    assert not base_moved, f"SFT moved BASE router columns: {base_moved[:3]}"
    assert new_moved, "SFT did not move the new router columns"
    assert lora_moved, "SFT did not move the new-expert LoRA weights"
