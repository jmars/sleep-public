#!/usr/bin/env python3
"""Teacher-only dream pass for the corpus protocol -> `runs/p6_dreams.pt`.

`run_corpus.py` can generate the cache itself (its phase 1), but the eval frames must be
FROZEN before any arm runs and `build_frames.py` mines the two-hop bridges out of the
dream cache, so the dream pass has to be a standalone step ahead of the arms. This is
that step. The cache format is exactly what `run_corpus.py` loads
(`{"dreams": [[str, ...], ...], "expert_load": ...}`), so a cache written here is
interchangeable with one written by the runner.

    python3 scripts/build_dreams.py --data data/squad_corpus200.jsonl --out runs/p6_dreams.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import torch  # noqa: E402

from squad_prep import read_jsonl  # noqa: E402

from qlora_sleep.config import load_config  # noqa: E402
from qlora_sleep.dream import DreamConfig  # noqa: E402
from qlora_sleep.model import load_tokenizer  # noqa: E402
from qlora_sleep.teacher import dream_teacher_stage  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="corpus_p6")
    ap.add_argument("--data", default="data/squad_corpus200.jsonl")
    ap.add_argument("--out", default="runs/p6_dreams.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    payload = read_jsonl(args.data)
    passages = payload["passages"]
    dream = cfg["dream"]
    max_len = int(cfg["sft"]["max_len"])

    tok = load_tokenizer(cfg["model"]["id"])
    t0 = time.time()
    stage = dream_teacher_stage(
        cfg["model"]["id"],
        tok,
        [p["context"] for p in passages],
        [
            DreamConfig(
                n_dreams=int(dream["n_dreams"]),
                temperature=float(dream["temperature"]),
                top_p=float(dream["top_p"]),
                max_new_tokens=int(dream["max_new_tokens"]),
                prompt_variant=str(dream["prompt_variant"]),
                random_expert=bool(dream["random_expert"]),
                seed=args.seed + i,
            )
            for i in range(len(passages))
        ],
        device=args.device,
        attn_implementation=cfg["model"]["attn_implementation"],
        max_len=max_len,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"dreams": stage["dreams"], "expert_load": stage["expert_load"]}, out)
    print(json.dumps({
        "data": args.data, "passages": len(passages),
        "dreams": sum(len(d) for d in stage["dreams"]),
        "variant": str(dream["prompt_variant"]), "temperature": float(dream["temperature"]),
        "seconds": round(time.time() - t0, 1), "out": str(out),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
