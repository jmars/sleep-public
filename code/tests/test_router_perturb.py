"""Residual test: dreams written while the teacher's router is forced off its top-k.

The perturbation replaces standard top-k slots with experts drawn from the router
distribution restricted to the NON-top-k columns. Three properties have to hold before any
GPU minute is spent on it:

* it selects only experts the standard route did not select, without replacement, and
  carries each substitute's own probability (distribution unchanged, selection moved);
* strength zero is bit-identical to no state at all -- the eval path cannot be moved by a
  hook that is configured but inert (and the same instrument shows a non-zero strength
  DOES move the forward pass, so the check is falsifiable);
* the strength is logged (a1, drawn/displaced probability mass), so a dream pass cannot
  claim a perturbation it did not apply.
"""

from __future__ import annotations

import pytest
import torch

from qlora_sleep.model import (
    _perturb_selection,
    current_state,
    perturb_summary,
    routing_identity_check,
    routing_mode,
)
from qlora_sleep.router_warmup import token_batch


def _forward(model, tokenizer, text, max_len=64, **kwargs):
    device = next(model.parameters()).device
    input_ids, attn, _ = token_batch(tokenizer, [text], max_len, device)
    with torch.no_grad(), routing_mode(**kwargs) as state:
        logits = model(input_ids=input_ids, attention_mask=attn, use_cache=False).logits
    return logits, state


def test_perturb_replaces_non_topk_slots():
    torch.manual_seed(0)
    n_experts, top_k = 12, 3
    probs = torch.rand(5, n_experts)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    weights, idx = torch.topk(probs, top_k, dim=-1)
    generator = torch.Generator().manual_seed(0)
    new_idx, new_w, drawn, displaced = _perturb_selection(
        probs, idx.clone(), weights.clone(), 2, generator
    )
    # the drawn experts are outside the standard top-k, and are distinct from each other
    for row in range(probs.shape[0]):
        standard = set(idx[row].tolist())
        assert not (set(drawn[row].tolist()) & standard), "drew an expert the route already had"
        assert len(set(drawn[row].tolist())) == 2, "sampled with replacement"
        # exactly two slots changed, and the substitute carries its own probability
        changed = (new_idx[row] != idx[row]).sum().item()
        assert changed == 2
        for position in range(top_k):
            if new_idx[row, position] != idx[row, position]:
                expert = int(new_idx[row, position])
                assert new_w[row, position] == pytest.approx(float(probs[row, expert]))
    # the displaced slots are the lowest-probability standard slots
    lowest = weights.argsort(dim=-1, stable=True)[:, :2]
    assert torch.equal(displaced, weights.gather(1, lowest))


def test_perturb_zero_slots_is_a_no_op():
    torch.manual_seed(0)
    probs = torch.rand(4, 8)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    weights, idx = torch.topk(probs, 2, dim=-1)
    new_idx, new_w, drawn, _ = _perturb_selection(probs, idx.clone(), weights.clone(), 0, None)
    assert torch.equal(new_idx, idx)
    assert torch.equal(new_w, weights)
    assert drawn.numel() == 0


def test_perturb_forward_is_logged_and_off_standard(expanded_model, tokenizer):
    text = "the access code for vault seven is four eight two one"
    _, state = _forward(expanded_model, tokenizer, text, perturb_slots=1)
    stats = perturb_summary(state)
    assert stats["perturb_slots"] == 1
    assert stats["tokens"] > 0
    # every perturbed token differs from the standard route by construction
    assert stats["a1_off_standard_fraction"] == pytest.approx(1.0)
    assert stats["mean_drawn_prob"] > 0.0
    assert stats["mean_displaced_prob"] > 0.0
    # strength zero logs nothing at all
    _, inert = _forward(expanded_model, tokenizer, text)
    assert perturb_summary(inert)["tokens"] == 0
    assert perturb_summary(inert)["a1_off_standard_fraction"] == 0.0


def test_perturb_rejects_warmup_interventions(expanded_model, tokenizer):
    device = next(expanded_model.parameters()).device
    input_ids, attn, _ = token_batch(tokenizer, ["the old mill"], 32, device)
    with pytest.raises(ValueError):
        with routing_mode(forced_slots=1, perturb_slots=1):
            expanded_model(input_ids=input_ids, attention_mask=attn, use_cache=False)


def test_routing_identity_check_zero_strength_is_inert(expanded_model, tokenizer):
    report = routing_identity_check(
        expanded_model, tokenizer, ["the river passes the old mill before the bridge"],
        max_len=32, perturb_slots=1,
    )
    # the eval path (no state) and a configured-but-zero-strength state agree bit for bit
    assert report["identical"] is True
    assert report["max_abs_diff_zero_strength"] == 0.0
    # asymmetry control: the same instrument sees a non-zero strength move the forward pass
    assert report["perturbation_changes_logits"] is True
    assert report["max_abs_diff_perturbed"] > 0.0
    assert report["perturb"]["a1_off_standard_fraction"] == pytest.approx(1.0)
    # and the check leaves no state installed
    assert current_state() is None
