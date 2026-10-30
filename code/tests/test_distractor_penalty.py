"""W2: the contrastive distractor penalty, and the control that makes it attributable.

W1 (forced inclusion only) teaches the new columns to be selected on positive text and
never says NOT-HERE, which is what the measured warmup state looks like: the bank is
over-selected on held-out text as much as on trained text (contrast -0.100).

The asymmetry this file establishes: the difference between the two arms is the PENALTY,
not the extra forward pass it needs. An arm with the distractor batches loaded and
`penalty_weight = 0` must be bit-identical to W1 -- otherwise a W2 result could be an
artefact of the second forward (an extra eval-mode pass through the routers) rather than
of the penalty.
"""

from __future__ import annotations

import torch

from qlora_sleep.model import (
    apply_new_expert_lora,
    expand_experts,
    patch_routing,
    set_trainable,
    unlock_new_experts,
)
from qlora_sleep.router_warmup import (
    WarmupConfig,
    new_bank_prob_mass,
    run_router_warmup,
    token_batch,
)
from conftest import K_NEW, N_BASE, make_tiny_model

DISTRACTORS = [
    "the harbour crane was repaired in the spring of that year",
    "a regional committee reviews the tariffs every second winter",
]


def build(seed: int = 0):
    """Two models built this way are identical: every init is drawn from the seeded stream."""
    model = make_tiny_model(seed)
    model = expand_experts(model, k_new=K_NEW, seed=0)
    model = patch_routing(model)
    model = apply_new_expert_lora(model, n_base=N_BASE, k_new=K_NEW, r=4, alpha=8)
    set_trainable(model, lora=True, routers=True, alphas=True)
    return model


def mass_on_distractors(model, tokenizer) -> float:
    unlock_new_experts(model)
    ids, attn, _ = token_batch(tokenizer, DISTRACTORS, 64, next(model.parameters()).device)
    with torch.no_grad():
        return float(new_bank_prob_mass(model, ids, attn))


def warm(model, tokenizer, texts, distractors, weight):
    return run_router_warmup(
        model, tokenizer, texts,
        WarmupConfig(steps=4, forced_slots=1, lr=1e-2, max_len=64, seed=0,
                     distractor_texts=distractors, share_target=0.05,
                     penalty_weight=weight),
    )


def gate_weights(model) -> list[torch.Tensor]:
    return [p.detach().clone() for n, p in model.named_parameters() if n.endswith("gate.weight")]


def test_penalty_lowers_the_banks_mass_on_distractor_text(tokenizer, texts):
    w1 = build()
    warm(w1, tokenizer, texts, [], 0.0)
    w2 = build()
    warm(w2, tokenizer, texts, DISTRACTORS, 1.0)

    mass_w1 = mass_on_distractors(w1, tokenizer)
    mass_w2 = mass_on_distractors(w2, tokenizer)
    assert mass_w2 < mass_w1, (
        f"penalty did not lower the new bank's mass on held-out text: {mass_w2} vs {mass_w1}"
    )


def test_zero_weight_arm_is_identical_to_w1(tokenizer, texts):
    w1 = build()
    warm(w1, tokenizer, texts, [], 0.0)
    zero = build()
    warm(zero, tokenizer, texts, DISTRACTORS, 0.0)

    for a, b in zip(gate_weights(w1), gate_weights(zero)):
        assert torch.equal(a, b), "the distractor forward pass moved the router without the penalty"


def test_history_carries_both_shares(tokenizer, texts):
    model = build()
    result = warm(model, tokenizer, texts, DISTRACTORS, 1.0)
    rows = result["history"]
    assert len(rows) == 4
    for row in rows:
        assert row["share_pos"] is not None
        assert row["share_neg"] is not None
        assert 0.0 <= row["share_pos"] <= 1.0
        assert 0.0 <= row["share_neg"] <= 1.0
