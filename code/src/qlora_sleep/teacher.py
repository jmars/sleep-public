"""Teacher: a bf16, unquantised snapshot of the same checkpoint, no gradients.

On a 16GB card the bf16 teacher (~14GB) and the NF4 student (~4GB) cannot be
resident together, so the teacher is loaded for dream generation only and
released before scoring/training. `release_teacher` is therefore part of the
cycle contract, not an optimisation.
"""

from __future__ import annotations

import gc

import torch

from .model import patch_routing


def load_teacher(
    model_id: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "eager",
):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        device_map={"": device},
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    patch_routing(model)
    return model


def release_teacher(teacher) -> None:
    """Drop a teacher and return its VRAM to the allocator.

    `del teacher` only unbinds THIS function's local name; the caller's binding still
    holds the model and its ~14GB. Callers must drop their own reference too --
    `dream_teacher_stage` below does it by construction, by holding the teacher
    inside a function scope that ends before the student is built. Getting this
    wrong is not a leak that shows up later: the very next allocation (the NF4
    quantisation of the student) OOMs.
    """
    del teacher
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def dream_teacher_stage(
    model_id: str,
    tokenizer,
    contexts: list[str],
    dream_configs: list,
    device: str = "cuda",
    attn_implementation: str = "eager",
    max_len: int = 256,
    stats: list | None = None,
) -> dict:
    """Load the teacher, measure expert load, generate dreams, release the teacher.

    Everything that touches the teacher lives in this scope, so when it returns the
    bf16 weights are unreferenced and the student can be built in the freed VRAM.

    `stats`, when given, receives one dict per passage with the routing perturbation
    strength actually applied during that passage's dream pass.
    """
    from .dream import generate_dreams
    from .router_warmup import expert_load

    teacher = load_teacher(model_id, device=device, attn_implementation=attn_implementation)
    try:
        load = expert_load(teacher, tokenizer, contexts, max_len=max_len, device=device)
        dreams = []
        for i, (context, cfg) in enumerate(zip(contexts, dream_configs)):
            passage_stats: dict = {}
            dreams.append(generate_dreams(teacher, tokenizer, context, cfg, stats=passage_stats))
            if stats is not None:
                stats.append(passage_stats)
    finally:
        # Order is the whole point: `release_teacher` empties the cache while the
        # caller's reference is still alive, so the ~14GB is not yet free and the
        # cache stays reserved-but-unallocated. Drop the reference FIRST, then
        # collect and empty, or the next allocation (the NF4 student) meets a
        # fragmented 12.5GiB cache and OOMs on a 2MB request.
        teacher = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {"dreams": dreams, "expert_load": load}


def teacher_footprint_gib(teacher) -> float:
    total = sum(p.numel() * p.element_size() for p in teacher.parameters())
    return total / (1024**3)
