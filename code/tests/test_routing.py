"""Risk (a) continued: does the router actually move, and does the anneal land?

Forced inclusion must (i) lift the new-expert routing share while it is active,
(ii) anneal to zero, and (iii) not train anything except the routers -- otherwise
"warmup" is a second, uncontrolled fine-tune that can move the base model.
"""

from __future__ import annotations

from qlora_sleep.model import (
    moe_blocks,
    new_expert_mask,
    new_expert_start,
    routing_mode,
    set_new_expert_mask,
)
from qlora_sleep.router_warmup import (
    WarmupConfig,
    aggregate_new_share,
    expert_load,
    lm_loss,
    routing_share,
    run_router_warmup,
    token_batch,
    warmup_schedule,
)


def _record(model, tokenizer, text, unlocked=False, **kwargs):
    if unlocked:
        set_new_expert_mask(model, 0.0)
    device = next(model.parameters()).device
    input_ids, attn, labels = token_batch(tokenizer, [text], 64, device)
    with routing_mode(record=True, **kwargs) as state:
        loss = lm_loss(model, input_ids, attn, labels)
    return state, loss


def test_forced_slots_lift_the_share(expanded_model, tokenizer):
    start = new_expert_start(expanded_model)
    text = "the access code for vault seven is four eight two one"
    cold, _ = _record(expanded_model, tokenizer, text)
    warm, _ = _record(expanded_model, tokenizer, text, forced_slots=1, unlocked=True)
    cold_share = aggregate_new_share(cold, start)
    warm_share = aggregate_new_share(warm, start)
    assert cold_share < 0.02, f"cold share {cold_share}"
    assert warm_share > 0.3, f"warm share {warm_share}"
    # one of two top-k slots is reserved, so at least ~half the selections are new
    assert warm_share >= 0.45, f"forced slot did not take: {warm_share}"


def test_forced_slots_never_exceed_top_k(expanded_model, tokenizer):
    start = new_expert_start(expanded_model)
    text = "the river passes the old mill before the bridge"
    state, _ = _record(expanded_model, tokenizer, text, forced_slots=99, unlocked=True)
    blocks = moe_blocks(expanded_model)
    per_token = state.counts[0].sum().item() / max(state.calls[0], 1)
    assert per_token == blocks[0].top_k, f"forced slots changed the slot count: {per_token}"


def test_anneal_schedule_is_monotone_and_lands_on_zero():
    values = [warmup_schedule(step, 25, 2) for step in range(25)]
    assert values[0] == 2
    assert values[-1] == 1
    assert warmup_schedule(25, 25, 2) == 0
    assert warmup_schedule(99, 25, 2) == 0
    assert all(a >= b for a, b in zip(values, values[1:])), f"not monotone: {values}"
    assert warmup_schedule(0, 0, 2) == 0


def test_random_expert_adds_a_slot(expanded_model, tokenizer):
    text = "the river passes the old mill before the bridge"
    base, _ = _record(expanded_model, tokenizer, text)
    rnd, _ = _record(expanded_model, tokenizer, text, random_expert=True)
    blocks = moe_blocks(expanded_model)
    top_k = blocks[0].top_k
    base_slots = base.counts[0].sum().item() / max(base.calls[0], 1)
    rnd_slots = rnd.counts[0].sum().item() / max(rnd.calls[0], 1)
    assert base_slots == top_k
    assert rnd_slots == top_k + 1, f"random expert did not add a slot: {rnd_slots}"


def test_warmup_trains_routers_only(expanded_model, tokenizer, texts):
    start = new_expert_start(expanded_model)
    assert new_expert_mask(expanded_model) < 0, "fixture should start masked"
    before = {name: p.detach().clone() for name, p in expanded_model.named_parameters()}
    config = WarmupConfig(steps=3, forced_slots=1, lr=1e-2, max_len=64, seed=0)
    result = run_router_warmup(expanded_model, tokenizer, texts, config)
    assert len(result["history"]) == 3
    assert result["history"][-1]["forced_slots"] == 1
    router_moved = 0
    new_router_moved = 0
    base_moved: list[str] = []
    lora_moved = []
    alpha_moved = []
    for name, param in expanded_model.named_parameters():
        delta = float((param.detach() - before[name]).abs().max())
        if name.endswith("mlp.gate.weight"):
            if delta > 0:
                router_moved += 1
                # base columns must not move at all: they carry the frozen base's
                # routing behaviour. Counting "the new columns moved too" is not the
                # invariant -- a warmup that moved BOTH slices satisfies it, and that is
                # exactly the failure mode measured on the real model (EM 20 -> 0).
                base_delta = float((param.detach()[:start] - before[name][:start]).abs().max())
                if base_delta > 0:
                    base_moved.append(name)
                if float((param.detach()[start:] - before[name][start:]).abs().max()) > 0:
                    new_router_moved += 1
        elif "lora_" in name:
            if delta != 0.0:
                lora_moved.append(name)
        elif name.endswith("expert_gates"):
            if delta != 0.0:
                alpha_moved.append(name)
    assert router_moved > 0, "warmup did not move any router"
    assert not base_moved, f"warmup moved BASE router columns: {base_moved[:3]}"
    assert new_router_moved > 0, "warmup did not move the new columns"
    assert not lora_moved, f"warmup moved LoRA weights: {lora_moved[:3]}"
    assert not alpha_moved, f"warmup moved alpha gates: {alpha_moved[:3]}"
    for row in result["history"]:
        assert row["new_share"] > 0.3, f"warmup share collapsed at step {row['step']}: {row}"
        assert row["new_router_grad_norm"] > 0, f"no router gradient at step {row['step']}: {row}"
        assert row["base_router_grad_norm"] == 0.0, (
            f"base router columns received gradient at step {row['step']}: {row}"
        )
    assert new_expert_mask(expanded_model) == 0.0, "warmup must leave the new experts unmasked"


def test_routing_share_report_shape(expanded_model, tokenizer, texts):
    report = routing_share(expanded_model, tokenizer, texts, max_len=64)
    assert report["n_layers"] == 2
    assert len(report["layers"]) == 2
    assert "aggregate_new_share" in report
    layer0 = report["layers"]["0"]
    assert len(layer0["counts"]) == 8 + 2
    assert layer0["new_share"] is not None


def test_expert_load_shape(expanded_model, tokenizer, texts):
    load = expert_load(expanded_model, tokenizer, texts, max_len=64)
    assert load.shape == (2, 10)
    assert load.sum() > 0
