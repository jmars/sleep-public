"""Expert-bank surgery and routing interventions for OLMoE-style sparse MoE.

The model is treated as a bank of `num_experts` small MLPs per layer plus a
linear router. Sleep adds `k_new` cloned experts per layer, extends the router
with `k_new` small-initialised columns, and adds one scalar output gate
`alpha_e` per new expert.

`alpha_e` is a MEASUREMENT GATE, not a trained parameter: it stays at 1.0 (the
optimisers built by `param_groups` never receive it) and its gradient is the
behavioural importance signal used by `pruning.py`. Training it would destroy
the instrument.

Routing interventions (forced inclusion, random-expert dreaming, statistics
recording) are carried by a context variable so that they do not have to be
threaded through `model.forward`, `generate`, or gradient checkpointing.
"""

from __future__ import annotations

import contextvars
import math
import re
import types
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# routing state
# --------------------------------------------------------------------------- #


@dataclass
class RoutingState:
    """Interventions applied by the patched MoE forward for the current context."""

    forced_slots: int = 0
    """Reserve this many of the top-k slots for the highest-probability NEW experts."""

    force_new_expert: int | None = None
    """Append new expert `force_new_expert` (0-based within the new bank) as an EXTRA
    slot on top of top-k, without displacing anything. Used to probe one expert.

    The forced slot is weighted at the MEAN of the top-k weights rather than at the
    expert's own router probability: otherwise a silenced expert would be probed at
    weight ~0 and its importance would collapse to zero, which is exactly the
    utilisation signal we are trying to avoid."""

    random_expert: bool = False
    """Additionally select one uniformly random expert per token (paper, dreaming)."""

    record: bool = False
    """Accumulate per-expert selection counts, mean probabilities and activation norms."""

    generator: torch.Generator | None = None
    """Optional RNG for `random_expert` and `perturb_slots` (must live on the compute device)."""

    perturb_slots: int = 0
    """Replace this many standard top-k slots with experts drawn from OUTSIDE the top-k.

    Dreaming under router perturbation. The substitute is drawn from the router
    distribution ITSELF with the standard top-k columns zeroed (without replacement), so
    the distribution and temperature are untouched and only WHICH experts are consulted
    moves. Each substitute carries its own probability as its mixture weight, and the
    displaced positions are the LOWEST-probability standard slots, so the standard
    mixture is disturbed by the least amount the requested strength allows.

    Generation-time only: installed inside `generate_dreams`, never at eval. Eval reads
    run with no routing state at all, and `routing_identity_check` measures that the
    zero-strength state is bit-identical to it.
    """

    perturb: dict = field(default_factory=dict)
    """Perturbation telemetry (forward passes, tokens, off-standard tokens, drawn and
    displaced probability mass). Empty while the intervention is inert."""

    counts: dict[int, torch.Tensor] = field(default_factory=dict)
    prob_sums: dict[int, torch.Tensor] = field(default_factory=dict)
    act_sums: dict[int, torch.Tensor] = field(default_factory=dict)
    calls: dict[int, int] = field(default_factory=dict)

    def slot(self, layer_idx: int, num_experts: int, device: torch.device) -> None:
        if layer_idx not in self.counts:
            self.counts[layer_idx] = torch.zeros(num_experts, dtype=torch.float64, device=device)
            self.prob_sums[layer_idx] = torch.zeros(num_experts, dtype=torch.float64, device=device)
            self.act_sums[layer_idx] = torch.zeros(num_experts, dtype=torch.float64, device=device)
            self.calls[layer_idx] = 0


_ROUTING: contextvars.ContextVar[RoutingState | None] = contextvars.ContextVar("qlora_sleep_routing", default=None)


@contextmanager
def routing_mode(
    forced_slots: int = 0,
    force_new_expert: int | None = None,
    random_expert: bool = False,
    record: bool = False,
    generator: torch.Generator | None = None,
    perturb_slots: int = 0,
) -> Iterator[RoutingState]:
    """Install a routing state for the duration of the block."""
    state = RoutingState(
        forced_slots=forced_slots,
        force_new_expert=force_new_expert,
        random_expert=random_expert,
        record=record,
        generator=generator,
        perturb_slots=perturb_slots,
    )
    token = _ROUTING.set(state)
    try:
        yield state
    finally:
        _ROUTING.reset(token)


def current_state() -> RoutingState | None:
    return _ROUTING.get()


# --------------------------------------------------------------------------- #
# module discovery and patching
# --------------------------------------------------------------------------- #


def is_moe_block(module: torch.nn.Module) -> bool:
    return (
        hasattr(module, "experts")
        and hasattr(module, "gate")
        and hasattr(module, "top_k")
        and isinstance(getattr(module, "experts"), torch.nn.ModuleList)
    )


def moe_blocks(model: torch.nn.Module) -> list[torch.nn.Module]:
    """MoE blocks in registration order (which is layer order for OLMoE)."""
    return [m for _, m in model.named_modules() if is_moe_block(m)]


def _perturb_selection(
    probs: torch.Tensor,
    idx: torch.Tensor,
    weights: torch.Tensor,
    n_slots: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replace `n_slots` standard top-k slots with non-top-k experts.

    Sampling is from the router distribution with the standard top-k columns removed
    (without replacement), so temperature and distribution are untouched: what moves is
    which experts are consulted, not how sharply. The LOWEST-probability standard slots
    are displaced and each substitute carries its own probability as its mixture weight.

    Returns (idx, weights, drawn, displaced) where `displaced` is the probability mass
    that was removed -- the strength reading, as opposed to the requested slot count.
    """
    n_slots = min(int(n_slots), idx.shape[1])
    if n_slots <= 0:
        return idx, weights, idx[:, :0], idx[:, :0].to(weights.dtype)
    outside = probs.clone()
    outside.scatter_(1, idx, 0.0)
    drawn = []
    for _ in range(n_slots):
        draw = torch.multinomial(outside, 1, generator=generator)
        drawn.append(draw)
        # without replacement: a second draw may not repeat the first
        outside.scatter_(1, draw, 0.0)
    drawn = torch.cat(drawn, dim=-1)
    positions = weights.argsort(dim=-1, stable=True)[:, :n_slots]
    displaced = weights.gather(1, positions)
    idx = idx.scatter(1, positions, drawn)
    weights = weights.scatter(1, positions, probs.gather(1, drawn))
    return idx, weights, drawn, displaced


def _moe_forward(self, hidden_states: torch.Tensor):
    state = _ROUTING.get()
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    flat = hidden_states.view(-1, hidden_dim)

    router_logits = self.gate(flat)
    router_mask = getattr(self, "router_mask", None)
    if router_mask is not None:
        # Masked pre-allocation: new experts exist but are held out of the routing
        # until a sleep stage unlocks them. Added to the logits, so a large negative
        # value removes the expert from top-k selection without touching its weights.
        router_logits = router_logits + router_mask
    probs = F.softmax(router_logits, dim=1, dtype=torch.float)
    top_k = self.top_k
    weights, idx = torch.topk(probs, top_k, dim=-1)

    new_start = getattr(self, "new_expert_start", None)
    n_new = getattr(self, "num_new_experts", 0)

    if state is not None and state.perturb_slots > 0:
        if state.force_new_expert is not None or state.forced_slots > 0:
            raise ValueError(
                "router perturbation replaces standard top-k slots and cannot be combined "
                "with forced_slots/force_new_expert (those are warmup/probe interventions)"
            )
        standard_idx = idx
        n_perturb = min(int(state.perturb_slots), top_k)
        idx, weights, drawn, displaced = _perturb_selection(
            probs, idx, weights, n_perturb, state.generator
        )
        # a1: tokens whose expert SET differs from the standard top-k set. Logged rather
        # than assumed: a perturbation that silently selected nothing would otherwise be
        # indistinguishable from a dream pass that ran as standard.
        off_standard = (idx != standard_idx).any(dim=-1)
        log = state.perturb
        log["forwards"] = log.get("forwards", 0) + 1
        log["tokens"] = log.get("tokens", 0) + int(flat.shape[0])
        log["off_standard_tokens"] = log.get("off_standard_tokens", 0) + int(off_standard.sum())
        log["slots"] = log.get("slots", 0) + int(n_perturb) * int(flat.shape[0])
        log["drawn_prob_sum"] = log.get("drawn_prob_sum", 0.0) + float(
            probs.gather(1, drawn).detach().sum()
        )
        log["displaced_prob_sum"] = log.get("displaced_prob_sum", 0.0) + float(
            displaced.detach().sum()
        )

    if state is not None and new_start is not None and n_new > 0:
        if state.force_new_expert is not None:
            # Extra slot: nothing is displaced, so the loss delta attributable to
            # this expert is not contaminated by losing another expert's output.
            e = new_start + state.force_new_expert
            extra = torch.full((flat.shape[0], 1), e, device=flat.device, dtype=idx.dtype)
            idx = torch.cat([idx, extra], dim=-1)
            probe_weight = weights.mean(dim=-1, keepdim=True)
            weights = torch.cat([weights, probe_weight], dim=-1)
        elif state.forced_slots > 0:
            # The forced slot carries the new expert's OWN softmax probability, not a
            # substitute weight. That is what makes warmup train the router: the
            # column appears in the mixture weight, so d(loss)/d(column) is non-zero
            # and the router learns to keep selecting the expert on merit. (Weighting
            # the slot with a mean of base weights instead leaves the new column with
            # only an indirect gradient through the softmax normaliser -- a warmup
            # that moves nothing while its loss curve looks healthy.)
            #
            # Forced inclusion only does real work once the router mask is lifted;
            # masked columns have probability ~0 and their output vanishes. The
            # warmup procedure therefore unlocks first and forces second.
            nf = min(state.forced_slots, top_k)
            # Base slots are drawn from BASE columns only, so a forced new expert can
            # never occupy a base slot as well: a duplicate index would add the same
            # expert's output twice and inflate its recorded utilisation.
            base_w, base_i = torch.topk(probs[:, :new_start], top_k - nf, dim=-1)
            _, fi = torch.topk(probs[:, new_start:], nf, dim=-1)
            forced_w = probs.gather(1, fi + new_start)
            idx = torch.cat([base_i, fi + new_start], dim=-1)
            weights = torch.cat([base_w, forced_w], dim=-1)

    if state is not None and state.random_expert:
        rnd = torch.randint(
            0,
            self.num_experts,
            (flat.shape[0], 1),
            device=flat.device,
            generator=state.generator,
        )
        idx = torch.cat([idx, rnd], dim=-1)
        weights = torch.cat([weights, probs.gather(1, rnd)], dim=-1)

    if self.norm_topk_prob:
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    weights = weights.to(flat.dtype)

    recording = state is not None and state.record
    layer_idx = getattr(self, "_layer_idx", 0)
    if recording:
        state.slot(layer_idx, self.num_experts, flat.device)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim), dtype=flat.dtype, device=flat.device
    )
    expert_mask = F.one_hot(idx, num_classes=self.num_experts).permute(2, 1, 0)

    for expert_idx in range(self.num_experts):
        slot, top_x = torch.where(expert_mask[expert_idx])
        if top_x.numel() == 0:
            continue
        expert_layer = self.experts[expert_idx]
        current_state = flat[top_x]
        current_hidden_states = expert_layer(current_state)
        w = weights[top_x, slot, None]
        if new_start is not None and expert_idx >= new_start and hasattr(self, "expert_gates"):
            w = w * self.expert_gates[expert_idx - new_start]
        current_hidden_states = current_hidden_states * w
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(flat.dtype))

        if recording:
            n_sel = top_x.numel()
            state.counts[layer_idx][expert_idx] += n_sel
            if slot.numel():
                # activation magnitude actually written into the residual stream
                act = current_hidden_states.detach().norm(dim=-1).sum()
                state.act_sums[layer_idx][expert_idx] += act

    if recording:
        state.prob_sums[layer_idx] += probs.detach().sum(dim=0)
        state.calls[layer_idx] += flat.shape[0]

    return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim), router_logits


def patch_routing(model: torch.nn.Module) -> torch.nn.Module:
    """Replace every MoE block's forward with the intervention-aware version."""
    for i, block in enumerate(moe_blocks(model)):
        block._layer_idx = i
        block.forward = types.MethodType(_moe_forward, block)
    model._routing_patched = True
    return model


def perturb_summary(state: RoutingState) -> dict:
    """Normalise the perturbation telemetry into the strength readings a run logs.

    `a1_off_standard_fraction` is the brief's a1: the fraction of routed tokens whose
    expert set differs from the standard top-k set. `mean_drawn_prob` and
    `mean_displaced_prob` are the router probabilities substituted in and displaced out,
    i.e. how strongly the distribution was actually disturbed as opposed to how many
    slots were requested.
    """
    log = state.perturb
    tokens = int(log.get("tokens", 0))
    slots = int(log.get("slots", 0))
    return {
        "perturb_slots": int(state.perturb_slots),
        "forwards": int(log.get("forwards", 0)),
        "tokens": tokens,
        "off_standard_tokens": int(log.get("off_standard_tokens", 0)),
        "a1_off_standard_fraction": (log.get("off_standard_tokens", 0) / tokens) if tokens else 0.0,
        "mean_drawn_prob": (log.get("drawn_prob_sum", 0.0) / slots) if slots else 0.0,
        "mean_displaced_prob": (log.get("displaced_prob_sum", 0.0) / slots) if slots else 0.0,
    }


def routing_identity_check(
    model: torch.nn.Module,
    tokenizer,
    texts: list[str],
    max_len: int = 256,
    perturb_slots: int = 2,
    seed: int = 0,
) -> dict:
    """Prove the dreaming perturbation is INERT at strength zero (the eval path).

    Three forwards on the same input: no routing state (what every eval read does), a
    state with `perturb_slots=0`, and a state with the perturbation on. `identical` is the
    requirement -- the hook cannot move an eval read. `perturbation_changes_logits` is the
    ASYMMETRY control: a check that cannot fail proves nothing, so the same instrument has
    to show that a non-zero strength DOES move the forward pass.
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        enc = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len
        )
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        with torch.no_grad():
            plain = model(input_ids=input_ids, attention_mask=attn, use_cache=False).logits
            with routing_mode(perturb_slots=0):
                zero = model(input_ids=input_ids, attention_mask=attn, use_cache=False).logits
            generator = torch.Generator(device=device).manual_seed(seed)
            with routing_mode(perturb_slots=perturb_slots, generator=generator) as state:
                perturbed = model(input_ids=input_ids, attention_mask=attn, use_cache=False).logits
        stats = perturb_summary(state)
    finally:
        model.train(was_training)
    return {
        "n_tokens": int(input_ids.numel()),
        "identical": bool(torch.equal(plain, zero)),
        "max_abs_diff_zero_strength": float((plain - zero).abs().max()),
        "perturbation_changes_logits": bool(not torch.allclose(plain, perturbed)),
        "max_abs_diff_perturbed": float((plain - perturbed).abs().max()),
        "perturb": stats,
    }


# --------------------------------------------------------------------------- #
# expert surgery
# --------------------------------------------------------------------------- #


def _dequantised_weight(linear: torch.nn.Module, dtype: torch.dtype) -> torch.Tensor:
    """Recover a dense weight from a possibly 4-bit quantised Linear.

    bitsandbytes stores the packed data on the parameter (`Params4bit.dequantize`),
    not on the module, so probe the parameter first.
    """
    weight = getattr(linear, "weight", None)
    state = getattr(weight, "quant_state", None)
    if state is not None and getattr(weight, "bnb_quantized", False):
        # bitsandbytes 0.50: Params4bit.dequantize() returns the packed uint8 data,
        # so the dense weight has to be rebuilt from the quantisation state.
        import bitsandbytes.functional as bnb_functional

        return bnb_functional.dequantize_4bit(weight.data, state).to(dtype)
    if hasattr(weight, "dequantize"):
        dense = weight.dequantize()
        if dense.shape == weight.shape:
            return dense.to(dtype)
    if hasattr(linear, "dequantize"):
        return linear.dequantize().to(dtype)
    return weight.data.to(dtype)


def _clone_expert(src: torch.nn.Module, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    """Build a fresh bf16 copy of `src` (which may be a 4-bit quantised module)."""
    clone = type(src)(src.config)
    clone = clone.to(device=device, dtype=dtype)
    with torch.no_grad():
        for name in ("gate_proj", "up_proj", "down_proj"):
            getattr(clone, name).weight.copy_(_dequantised_weight(getattr(src, name), dtype))
    for p in clone.parameters():
        p.requires_grad_(False)
    return clone


def expand_experts(
    model: torch.nn.Module,
    k_new: int,
    load: torch.Tensor | None = None,
    init_noise_frac: float = 1.0,
    mask_value: float = 20.0,
    clone_from: str = "router_load",
    seed: int = 0,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.nn.Module:
    """Append `k_new` cloned experts per layer and extend each router.

    `load` is an optional (n_layers, n_base) tensor of routing counts; the
    `k_new` most-loaded base experts are cloned. Without `load`, the first
    `k_new` experts are cloned.

    New experts are MASKED at initialisation, following the paper's masked
    pre-allocation (sleep.txt:418-422): a `router_mask` buffer adds `-mask_value`
    to their router logits, so token-choice top-k never selects them until a sleep
    stage unlocks them. Without the mask the trap is only probabilistic -- whether
    a new expert is ever selected depends on where its initialised column happens
    to land relative to the base logits -- and the whole router-warmup risk becomes
    luck rather than a mechanism.
    """
    blocks = moe_blocks(model)
    if not blocks:
        raise ValueError("no MoE blocks found; is this an OLMoE-style model?")
    n_base = blocks[0].num_experts
    if load is not None and load.shape[0] != len(blocks):
        raise ValueError(f"load has {load.shape[0]} layers, model has {len(blocks)}")
    if k_new < 1:
        raise ValueError("k_new must be >= 1")
    device = torch.device(device) if device is not None else blocks[0].gate.weight.device
    if dtype is None:
        existing = blocks[0].gate.weight.dtype
        dtype = existing if existing.is_floating_point else torch.bfloat16

    gen = torch.Generator(device="cpu").manual_seed(seed)
    for li, block in enumerate(blocks):
        if getattr(block, "num_new_experts", 0):
            raise ValueError("experts already expanded; refusing to expand twice")
        if not isinstance(block.gate, torch.nn.Linear):
            raise TypeError(
                f"router is {type(block.gate).__name__}; skip it from quantisation "
                "(the child name 'gate' must be in llm_int8_skip_modules) so it stays a plain Linear"
            )
        if clone_from == "router_load" and load is not None:
            sources = torch.topk(load[li], k_new).indices.tolist()
        else:
            sources = list(range(k_new))
        for j, src_idx in enumerate(sources):
            clone = _clone_expert(block.experts[src_idx], device, dtype)
            clone.clone_source = int(src_idx)
            block.experts.append(clone)

        old = block.gate
        # the router must be a plain floating-point Linear; if it was quantised the
        # base columns are recovered from the packed weights rather than silently
        # producing a uint8 gate
        base_weight = _dequantised_weight(old, dtype)
        sigma_w = float(base_weight.detach().std())
        new_gate = torch.nn.Linear(
            old.in_features, n_base + k_new, bias=old.bias is not None, device=device, dtype=dtype
        )
        with torch.no_grad():
            new_gate.weight[:n_base] = base_weight
            noise = torch.randn(k_new, old.in_features, generator=gen).to(device=device, dtype=dtype)
            new_gate.weight[n_base:] = noise * (sigma_w * init_noise_frac)
            if old.bias is not None:
                new_gate.bias[:n_base] = old.bias.detach().to(dtype)
                new_gate.bias[n_base:] = 0.0
        block.gate = new_gate
        block.num_experts = n_base + k_new
        block.new_expert_start = n_base
        block.num_new_experts = k_new
        block.expert_gates = torch.nn.Parameter(torch.ones(k_new, device=device, dtype=torch.float32))
        mask = torch.zeros(n_base + k_new, device=device, dtype=dtype)
        mask[n_base:] = -mask_value
        block.register_buffer("router_mask", mask)

    model.config.num_experts = n_base + k_new
    if hasattr(model, "model") and hasattr(model.model, "num_experts"):
        model.model.num_experts = n_base + k_new
    return model


def new_expert_start(model: torch.nn.Module) -> int:
    blocks = moe_blocks(model)
    start = getattr(blocks[0], "new_expert_start", None)
    if start is None:
        raise ValueError("model has no new experts; call expand_experts first")
    return start


def zero_base_router_grad(model: torch.nn.Module) -> None:
    """Zero the gradient on BASE router columns, leaving only the new columns training.

    The base model is frozen; its routing behaviour is the thing the sleep cycle is
    supposed to preserve. Training all 64 base columns as well as the new ones lets a
    short, high-learning-rate warmup rotate the base router by a few percent per
    element -- measured on this model that is enough to take no-context SQuAD EM from
    20% to 0%. Constraining the gradient to the new columns removes that failure mode
    by construction instead of tuning around it.
    """
    for block in moe_blocks(model):
        grad = block.gate.weight.grad
        if grad is None:
            continue
        with torch.no_grad():
            grad[: block.new_expert_start] = 0.0
            if block.gate.bias is not None and block.gate.bias.grad is not None:
                block.gate.bias.grad[: block.new_expert_start] = 0.0


def base_router_grad_norm(model: torch.nn.Module) -> float:
    """L2 norm of the gradient currently sitting on BASE router columns."""
    total = 0.0
    for block in moe_blocks(model):
        grad = block.gate.weight.grad
        if grad is not None:
            total += float(grad[: block.new_expert_start].detach().pow(2).sum())
    return math.sqrt(total)


def set_new_expert_mask(model: torch.nn.Module, value: float) -> None:
    """Set the mask added to every NEW expert's router logit (0.0 = fully unlocked).

    The mask is an INITIAL condition and a switch, not a permanently-on filter:
    masked at surgery (so the frozen base behaves exactly as before, and so the
    router-warmup trap is deterministic), lifted for the sleep cycle and for
    evaluation (otherwise a new expert could never be used and could never help).
    """
    blocks = moe_blocks(model)
    for block in blocks:
        start = block.new_expert_start
        with torch.no_grad():
            block.router_mask[start:] = value


def new_expert_mask(model: torch.nn.Module) -> float:
    """Current mask value on the new experts (0.0 means unlocked)."""
    return float(moe_blocks(model)[0].router_mask[-1])


def unlock_new_experts(model: torch.nn.Module) -> None:
    set_new_expert_mask(model, 0.0)


# --------------------------------------------------------------------------- #
# LoRA on the new experts only
# --------------------------------------------------------------------------- #


def new_expert_lora_regex(n_base: int, k_new: int) -> str:
    alts = "|".join(str(i) for i in range(n_base, n_base + k_new))
    return rf".*\.experts\.(?:{alts})\.(?:gate_proj|up_proj|down_proj)"


_EXPERT_IDX_RE = re.compile(r"\.experts\.(\d+)\.")


def apply_new_expert_lora(
    model: torch.nn.Module,
    n_base: int,
    k_new: int,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.0,
) -> torch.nn.Module:
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=new_expert_lora_regex(n_base, k_new),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    check_new_experts_only(model, n_base, k_new)
    return model


def lora_module_indices(model: torch.nn.Module) -> list[int]:
    """Expert indices that carry at least one LoRA adapter."""
    found = set()
    for name, _ in model.named_parameters():
        if "lora_" in name:
            m = _EXPERT_IDX_RE.search(name)
            if m is not None:
                found.add(int(m.group(1)))
    return sorted(found)


def check_new_experts_only(model: torch.nn.Module, n_base: int, k_new: int) -> None:
    """Raise unless every LoRA adapter sits on a NEW expert, and all of them got one."""
    seen = set()
    for name, _ in model.named_parameters():
        if "lora_" not in name:
            continue
        m = _EXPERT_IDX_RE.search(name)
        if m is None:
            raise AssertionError(f"LoRA parameter outside any expert: {name}")
        idx = int(m.group(1))
        if idx < n_base:
            raise AssertionError(f"LoRA leaked onto base expert {idx}: {name}")
        seen.add(idx)
    expected = set(range(n_base, n_base + k_new))
    if not seen:
        raise AssertionError("no LoRA adapters were created on the new experts")
    if not seen <= expected:
        raise AssertionError(f"unexpected expert indices with LoRA: {sorted(seen - expected)}")
    # every new expert must be covered (per layer coverage is implied by the regex)
    if len(seen) != k_new:
        raise AssertionError(f"LoRA covered {len(seen)} of {k_new} new expert slots")


# --------------------------------------------------------------------------- #
# trainable sets
# --------------------------------------------------------------------------- #


def is_router_param(name: str) -> bool:
    return bool(re.search(r"\.mlp\.gate\.", name)) or name.endswith("mlp.gate.weight")


def is_alpha_param(name: str) -> bool:
    return name.endswith("expert_gates")


def set_trainable(
    model: torch.nn.Module,
    lora: bool = True,
    routers: bool = True,
    alphas: bool = True,
) -> dict[str, list[str]]:
    """Set requires_grad over the three param groups; everything else stays frozen."""
    groups: dict[str, list[str]] = {"lora": [], "router": [], "alpha": [], "frozen": []}
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad_(lora)
            groups["lora" if lora else "frozen"].append(name)
        elif is_alpha_param(name):
            # kept grad-enabled as the importance probe, but never handed to an optimiser
            param.requires_grad_(alphas)
            groups["alpha" if alphas else "frozen"].append(name)
        elif is_router_param(name):
            param.requires_grad_(routers)
            groups["router" if routers else "frozen"].append(name)
        else:
            param.requires_grad_(False)
            groups["frozen"].append(name)
    return groups


def param_groups(
    model: torch.nn.Module, lr_lora: float, lr_router: float, weight_decay: float = 0.0
) -> list[dict]:
    """Optimiser param groups: LoRA and routers only. Alpha gates are excluded."""
    lora_p, router_p = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" in name:
            lora_p.append(param)
        elif is_router_param(name):
            router_p.append(param)
        elif is_alpha_param(name):
            continue
    groups = []
    if lora_p:
        groups.append({"params": lora_p, "lr": lr_lora, "weight_decay": weight_decay})
    if router_p:
        groups.append({"params": router_p, "lr": lr_router, "weight_decay": 0.0})
    if not groups:
        raise ValueError("no trainable parameters selected")
    return groups


def count_trainable(model: torch.nn.Module) -> dict[str, int]:
    out = {"lora": 0, "router": 0, "alpha": 0, "total": 0}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        n = param.numel()
        out["total"] += n
        if "lora_" in name:
            out["lora"] += n
        elif is_alpha_param(name):
            out["alpha"] += n
        elif is_router_param(name):
            out["router"] += n
    return out


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


# transformers 5.x fuses OLMoE's experts into stacked tensors, which removes the
# per-expert ModuleList the whole design rests on (clone, append, LoRA on new slots).
# A version check that fires on the FIRST model touch, rather than a shape error
# somewhere inside the surgery.
TRANSFORMERS_MIN = (4, 55)
TRANSFORMERS_MAX_EXCLUSIVE = (5, 0)


def check_transformers_version() -> str:
    """Raise unless transformers is >= 4.55 and < 5. Returns the version string."""
    from transformers import __version__ as version

    parts = []
    for chunk in version.split(".")[:2]:
        digits = "".join(ch for ch in chunk if ch.isdigit())
        parts.append(int(digits or 0))
    while len(parts) < 2:
        parts.append(0)
    got = tuple(parts)
    if not (TRANSFORMERS_MIN <= got < TRANSFORMERS_MAX_EXCLUSIVE):
        raise RuntimeError(
            f"transformers {version} is outside the supported range "
            f">={'.'.join(map(str, TRANSFORMERS_MIN))},<{'.'.join(map(str, TRANSFORMERS_MAX_EXCLUSIVE))}: "
            "5.x fuses OLMoE experts into stacked tensors, which removes the per-expert "
            "ModuleList the expert surgery needs. Pin the version in the environment."
        )
    return version


def load_student(
    model_id: str,
    quantise: bool = True,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    skip_modules: tuple[str, ...] = ("lm_head", "gate"),
) -> torch.nn.Module:
    """Load the student: 4-bit NF4 base, eager attention, routers left unquantised."""
    check_transformers_version()
    from transformers import AutoModelForCausalLM

    kwargs: dict = {"torch_dtype": dtype, "attn_implementation": "eager"}
    if quantise:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=list(skip_modules),
        )
        kwargs["device_map"] = {"": device}
    else:
        kwargs["device_map"] = None
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if not quantise:
        model = model.to(device=device, dtype=dtype)
    return model


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def build_student(
    model_id: str,
    k_new: int,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    init_noise_frac: float = 1.0,
    mask_value: float = 20.0,
    load: torch.Tensor | None = None,
    quantise: bool = True,
    device: str = "cuda",
    dtype: torch.dtype | None = None,
) -> torch.nn.Module:
    """Full student assembly: quantise, clone experts, patch routing, add LoRA.

    `dtype` defaults to the dtype the base model's UNQUANTISED modules end up in.
    That matters: peft's k-bit preparation casts every non-4-bit parameter to fp32
    for stability, so a cloned expert built in bf16 would meet fp32 hidden states
    and fail the block's matmul. Following the base's dtype keeps the block
    homogeneous and costs ~1.6GB for 4 clones x 16 layers, which fits.
    """
    model = load_student(model_id, quantise=quantise, device=device, dtype=dtype or torch.bfloat16)
    if quantise:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    if dtype is None:
        base = moe_blocks(model)[0].gate.weight.dtype
        dtype = base if base.is_floating_point else torch.bfloat16
    model = expand_experts(
        model,
        k_new=k_new,
        load=load,
        init_noise_frac=init_noise_frac,
        mask_value=mask_value,
        device=device,
        dtype=dtype,
    )
    check_routers_are_trainable(model)
    model = patch_routing(model)
    model = apply_new_expert_lora(
        model, n_base=blocks_n_base(model), k_new=k_new, r=lora_r, alpha=lora_alpha, dropout=lora_dropout
    )
    set_trainable(model, lora=True, routers=True, alphas=True)
    return model


def check_routers_are_trainable(model: torch.nn.Module) -> None:
    """Invariant: every router is a plain floating-point Linear.

    transformers' 4-bit conversion skips a module only when its CHILD NAME is in
    `llm_int8_skip_modules` (e.g. "gate"), not its dotted path ("mlp.gate"). Getting
    that wrong silently yields a uint8 Linear4bit router, which cannot be trained.
    """
    for i, block in enumerate(moe_blocks(model)):
        weight = block.gate.weight
        if not weight.dtype.is_floating_point:
            raise TypeError(
                f"router {i} has dtype {weight.dtype}: it was quantised. Put the child name "
                f"('gate') in the quantisation skip list, not a dotted path."
            )
        if not isinstance(block.gate, torch.nn.Linear):
            raise TypeError(f"router {i} is {type(block.gate).__name__}, expected nn.Linear")


def blocks_n_base(model: torch.nn.Module) -> int:
    return new_expert_start(model)


# --------------------------------------------------------------------------- #
# runtime state: resume across a crash, and the rotating checkpoint
# --------------------------------------------------------------------------- #

RUNTIME_STATE_VERSION = 1


def save_runtime_state(model: torch.nn.Module, path, step: int | None = None) -> dict:
    """Persist what an interrupted arm needs to continue: adapters, routers, mask, step.

    The base weights are frozen and reload from the checkpoint, so only the runtime
    state is written (~9.4M LoRA + ~2.2M router parameters at r=16, i.e. tens of MB).
    The mask buffer is included because it is the SOLE routing control once the routers
    are frozen: a resume that silently reset it to 0.0 would evaluate an unbounded bank
    and the share band would no longer be the configuration the arm was trained under.
    """
    lora = {n: p.detach().to("cpu") for n, p in model.named_parameters() if "lora_" in n}
    routers = {n: p.detach().to("cpu") for n, p in model.named_parameters() if is_router_param(n)}
    masks = {
        str(i): block.router_mask.detach().to("cpu")
        for i, block in enumerate(moe_blocks(model))
    }
    state = {
        "version": RUNTIME_STATE_VERSION,
        "step": step,
        "new_expert_start": new_expert_start(model),
        "lora": lora,
        "routers": routers,
        "masks": masks,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    return {
        "path": str(path),
        "n_lora": len(lora),
        "n_routers": len(routers),
        "n_masks": len(masks),
        "step": step,
    }


def load_runtime_state(model: torch.nn.Module, path, strict: bool = True) -> dict:
    """Restore a state written by `save_runtime_state` into a freshly built student.

    Loud on every mismatch that would make the restored arm a different arm: a version
    bump, a different expert bank, or (in strict mode) a saved parameter the model does
    not have. Silently skipping those is how a resume becomes an untracked configuration.
    """
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    version = int(state.get("version", 0))
    if version != RUNTIME_STATE_VERSION:
        raise RuntimeError(
            f"runtime state {path} has version {version}, expected {RUNTIME_STATE_VERSION}"
        )
    start = new_expert_start(model)
    if int(state.get("new_expert_start", -1)) != start:
        raise RuntimeError(
            f"runtime state {path} was saved for a bank starting at "
            f"{state.get('new_expert_start')}, this model's starts at {start}"
        )
    params = dict(model.named_parameters())
    # the direction that matters: adapters the MODEL has and the state does not cover
    # would be left at their random initialisation, i.e. a silently different arm
    missing = [n for n in params if "lora_" in n and n not in state["lora"]]
    if strict and missing:
        raise KeyError(
            f"this model has {len(missing)} adapter tensors the runtime state lacks, "
            f"first: {missing[0]}"
        )
    loaded = 0
    with torch.no_grad():
        for group in ("lora", "routers"):
            for name, tensor in state[group].items():
                target = params.get(name)
                if target is None:
                    continue
                if tuple(target.shape) != tuple(tensor.shape):
                    raise ValueError(
                        f"shape mismatch restoring {name}: state {tuple(tensor.shape)} "
                        f"vs model {tuple(target.shape)}"
                    )
                target.copy_(tensor.to(device=target.device, dtype=target.dtype))
                loaded += 1
        for i, block in enumerate(moe_blocks(model)):
            saved = state["masks"].get(str(i))
            if saved is not None:
                block.router_mask.copy_(saved.to(block.router_mask.device))
    return {
        "loaded": loaded,
        "step": state.get("step"),
        "mask_value": new_expert_mask(model),
    }
