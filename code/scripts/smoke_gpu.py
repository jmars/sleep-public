#!/usr/bin/env python3
"""B0 smoke gate: the five-minute check that runs BEFORE any experiment arm spends money.

Every assertion here is a thing that has already broken this project once, on another
card, in another session:

* CUDA present and the device named (the rental is not the local ROCm card);
* NF4 Linear4bit forward AND backward on the GPU (bitsandbytes against this CUDA);
* OLMoE loads 4-bit, the expert surgery appends clones and the router columns extend;
* LoRA lands on the NEW experts only (a base-expert adapter silently fine-tunes the base);
* one 16-token generate (the eval path) and one SFT micro-step with NON-ZERO LoRA grad
  (a step whose gradient is zero looks like a healthy loss curve);
* routing telemetry records on two texts, with the new bank unlocked.

Exit 0 = PASS. Any failure means fix the environment before any arm burns GPU time.

    python scripts/smoke_gpu.py --model allenai/OLMoE-1B-7B-0125
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from qlora_sleep.model import (  # noqa: E402
    build_student,
    check_transformers_version,
    lora_module_indices,
    load_tokenizer,
    new_expert_start,
    unlock_new_experts,
)
from qlora_sleep.router_warmup import (  # noqa: E402
    aggregate_new_share,
    routing_share,
    run_router_warmup,
    WarmupConfig,
)
from qlora_sleep.sft import QA_TEMPLATE, SFTConfig, prepare_qa_examples, train_sft  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(' :: ' + detail) if detail else ''}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0125")
    ap.add_argument("--k-new", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=3)
    args = ap.parse_args()

    t0 = time.time()
    version = check_transformers_version()
    check("transformers version in range", True, version)
    check("cuda available", torch.cuda.is_available(), torch.cuda.get_device_name(0))
    check("device is not the local ROCm card", "NVIDIA" in torch.cuda.get_device_name(0),
          torch.cuda.get_device_name(0))

    # ---- NF4 fwd + bwd on the GPU -------------------------------------------
    import bitsandbytes as bnb

    layer = bnb.nn.Linear4bit(64, 32, quant_type="nf4", compute_dtype=torch.bfloat16).to(args.device)
    x = torch.randn(8, 64, device=args.device, dtype=torch.bfloat16, requires_grad=True)
    out = layer(x)
    out.float().pow(2).mean().backward()
    check("NF4 Linear4bit forward", out.shape == (8, 32), str(tuple(out.shape)))
    check("NF4 Linear4bit backward", x.grad is not None and torch.isfinite(x.grad).all(),
          "grad finite" if x.grad is not None else "no grad")

    # ---- student build ------------------------------------------------------
    t = time.time()
    tok = load_tokenizer(args.model)
    model = build_student(
        args.model, k_new=args.k_new, lora_r=16, lora_alpha=32, lora_dropout=0.0,
        init_noise_frac=1.0, mask_value=20.0, load=None, quantise=True, device=args.device,
    )
    build_s = time.time() - t
    start = new_expert_start(model)
    check("OLMoE 4-bit load + expand_experts", start == 64, f"n_base={start}, build={build_s:.0f}s")
    indices = lora_module_indices(model)
    check("LoRA on new experts only", indices == list(range(start, start + args.k_new)),
          f"lora experts {indices}")
    check("VRAM after build", True,
          f"{torch.cuda.memory_allocated()/2**30:.1f} GiB allocated / "
          f"{torch.cuda.memory_reserved()/2**30:.1f} GiB reserved")

    # ---- generate (the eval path) ------------------------------------------
    prompt = "Answer the question with a short factual answer.\n\nQuestion: Who wrote Hamlet?\nAnswer:"
    enc = tok(prompt, return_tensors="pt").to(args.device)
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=16, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    text = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    check("16-token greedy generate", len(text.strip()) > 0, repr(text.strip()[:40]))

    # ---- routing telemetry --------------------------------------------------
    unlock_new_experts(model)
    share = routing_share(model, tok, ["the treaty was signed in the spring of 1962",
                                       "which river passes the old mill"], max_len=64)
    check("routing telemetry records", share["aggregate_new_share"] > 0.0,
          f"new share {share['aggregate_new_share']:.3f} (uniform {args.k_new}/{start+args.k_new})")

    # ---- one warmup step, then one SFT step with real LoRA gradient ---------
    pairs = [("Who wrote Hamlet?", "William Shakespeare"), ("Where was Shakespeare born?", "Stratford-upon-Avon")]
    prepared = prepare_qa_examples(pairs, "", max_containment=0.5, n=8)["pairs"]
    warm = [QA_TEMPLATE.format(question=q, answer=a) for q, a in pairs]
    run_router_warmup(model, tok, warm,
                      WarmupConfig(steps=args.steps, forced_slots=2, lr=1e-3, max_len=64, seed=0))
    with torch.no_grad():
        state_share = routing_share(model, tok, warm, max_len=64)["aggregate_new_share"]
    check("warmup moved the new columns", state_share > 0.0, f"share after warmup {state_share:.3f}")

    sft = train_sft(model, tok, prepared,
                    SFTConfig(steps=args.steps, lr=1e-4, batch_size=2, max_len=64, seed=0,
                              train_routers=False),
                    context=None)
    lora_grads = [p.grad for n, p in model.named_parameters() if "lora_" in n and p.grad is not None]
    nonzero = sum(1 for g in lora_grads if float(g.abs().sum()) > 0.0)
    check("SFT micro-step ran", len(sft["history"]) == args.steps,
          f"loss {sft['history'][0]['loss']:.3f} -> {sft['final']['loss']:.3f}")
    check("SFT produced non-zero LoRA gradient", nonzero > 0,
          f"{nonzero}/{len(lora_grads)} LoRA tensors with a non-zero final gradient")

    failed = [name for name, ok, _ in CHECKS if not ok]
    wall = time.time() - t0
    print(f"\n[smoke] {len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed, wall={wall:.0f}s")
    if failed:
        print(f"[smoke] FAILED: {failed}")
        return 1
    print("[smoke] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
