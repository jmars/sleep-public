"""Risk (b): utilisation is the wrong pruning signal, importance is the right one.

The fixture plants the two shapes that matter:

* a RARE-BUT-CRITICAL canary: trained to carry a fact, then silenced in the router
  so it is almost never selected;
* a FREQUENT-BUT-INERT expert: selected constantly, contributes nothing.

Utilisation ranks the canary as the most prunable expert. Importance does not.
Both rankings are computed here and asserted against each other, so the test fails
if either metric degenerates into the other.
"""

from __future__ import annotations

import torch

from qlora_sleep.canaries import (
    boost_expert,
    canary_texts,
    inert_expert,
    install_canary_expert,
    silence_expert,
)
from qlora_sleep.model import moe_blocks, new_expert_start, routing_mode
from qlora_sleep.pruning import (
    expert_importance,
    select_prune,
    select_prune_utilisation,
)
from qlora_sleep.router_warmup import lm_loss, token_batch

CANARY = 0
INERT = 1


def _planted_model(expanded_model, tokenizer):
    texts = canary_texts(4)
    start = new_expert_start(expanded_model)
    # 1. make the canary load-bearing (trained under forced inclusion)
    train = install_canary_expert(
        expanded_model, tokenizer, texts, expert_offset=CANARY, steps=40, lr=1e-2, max_len=64
    )
    assert train["last"] < train["first"], f"canary did not learn: {train}"
    # 2. make the inert expert useless but frequently chosen
    inert_expert(expanded_model, INERT)
    boost_expert(expanded_model, INERT)
    # 3. starve the canary in the router
    silence_expert(expanded_model, CANARY)
    return texts, start


def test_utilisation_and_importance_disagree_on_the_canary(expanded_model, tokenizer):
    texts, start = _planted_model(expanded_model, tokenizer)
    result = expert_importance(expanded_model, tokenizer, texts, max_len=64)

    importance = result["importance"]
    utilisation = result["utilisation"]
    assert importance.shape == (2, 2)

    canary_imp = float(importance[:, CANARY].mean())
    inert_imp = float(importance[:, INERT].mean())
    canary_util = float(utilisation[:, CANARY].mean())
    inert_util = float(utilisation[:, INERT].mean())

    assert canary_util < 0.01, f"canary should be nearly unselected, got {canary_util}"
    assert inert_util > 0.2, f"inert expert should be frequent, got {inert_util}"
    assert canary_imp > 0.0, f"canary should reduce the loss when used, got {canary_imp}"
    assert canary_imp > inert_imp, (
        f"importance failed to rank the canary above the inert expert: "
        f"canary={canary_imp} inert={inert_imp}"
    )


def test_importance_pruning_keeps_the_canary(expanded_model, tokenizer):
    texts, start = _planted_model(expanded_model, tokenizer)
    result = expert_importance(expanded_model, tokenizer, texts, max_len=64)
    n_prune = 2  # one expert per layer out of two
    by_importance = select_prune(result["importance"], n_prune)
    by_utilisation = select_prune_utilisation(result["utilisation"], n_prune)

    canary_pairs = {(layer, CANARY) for layer in range(2)}
    assert canary_pairs.isdisjoint(set(by_importance)), (
        f"importance pruning dropped the canary: {by_importance}"
    )
    assert canary_pairs & set(by_utilisation), (
        f"utilisation pruning did not drop the canary, so the control is not a control: "
        f"{by_utilisation}"
    )


def test_importance_is_not_just_a_constant(expanded_model, tokenizer):
    texts, start = _planted_model(expanded_model, tokenizer)
    result = expert_importance(expanded_model, tokenizer, texts, max_len=64)
    spread = float(result["importance"].std())
    assert spread > 1e-3, f"importance is flat, so it cannot rank anything: {result['importance']}"
    assert result["base_loss"] > 0
    assert result["new_expert_start"] == start


def test_prune_zeroes_the_router_column_only_for_new_experts(expanded_model, tokenizer):
    from qlora_sleep.pruning import prune_experts

    texts, start = _planted_model(expanded_model, tokenizer)
    blocks = moe_blocks(expanded_model)
    before_base = blocks[0].gate.weight[:start].detach().clone()
    before_canary = blocks[0].gate.weight[start + CANARY].detach().clone()
    before_canary_mask = float(blocks[0].router_mask[start + CANARY])
    pruned = prune_experts(expanded_model, [(0, INERT)], reset_lora=True)
    assert pruned == 1
    assert torch.allclose(blocks[0].gate.weight[:start], before_base), "base router columns moved"
    assert torch.all(blocks[0].gate.weight[start + INERT] == 0), "pruned column not zeroed"
    assert float(blocks[0].expert_gates[INERT]) == 0.0
    assert torch.equal(blocks[0].gate.weight[start + CANARY], before_canary), "canary column was touched"
    assert float(blocks[0].router_mask[start + CANARY]) == before_canary_mask
