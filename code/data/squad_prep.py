"""SQuAD passage/question preparation for the SEAL knowledge-incorporation protocol.

We cannot reproduce SEAL's exact passage sample, so the protocol is reproduced in
KIND: group SQuAD v1.1 by context, keep passages with at least
`min_questions_per_passage` questions, and evaluate closed-book on those questions.
Every number is therefore comparable as a relative delta, not as an absolute
match to the paper (see the plan's re-basing rule).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def group_passages(rows: list[dict], min_questions: int = 1) -> list[dict]:
    by_context: dict[str, dict] = {}
    for row in rows:
        ctx = row["context"]
        entry = by_context.setdefault(
            ctx,
            {
                "passage_id": row.get("id", ""),
                "title": row.get("title", ""),
                "context": ctx,
                "questions": [],
            },
        )
        answers = row["answers"]["text"] if isinstance(row["answers"], dict) else row["answers"]
        entry["questions"].append({"question": row["question"], "answers": list(answers)})
    passages = [p for p in by_context.values() if len(p["questions"]) >= min_questions]
    return passages


def select_passages(passages: list[dict], n: int, seed: int = 0) -> list[dict]:
    if n >= len(passages):
        return list(passages)
    rng = random.Random(seed)
    return rng.sample(passages, n)


def _norm_context(text: str) -> str:
    return " ".join(text.split())


def exclude_contexts(passages: list[dict], contexts: set[str]) -> tuple[list[dict], int]:
    """Drop passages whose context is in `contexts` (whitespace-normalised match).

    This is the held-out guarantee for the corpus frame: the 200 training passages must
    be disjoint from base50 and gate24, otherwise the "held-out" non-destruction frame
    is trained on and the guard reads nothing.
    """
    keep = [p for p in passages if _norm_context(p["context"]) not in contexts]
    return keep, len(passages) - len(keep)


def load_squad(split: str = "train") -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("rajpurkar/squad", split=split)
    return [dict(row) for row in ds]


def build(
    n_passages: int,
    min_questions: int = 1,
    split: str = "train",
    seed: int = 0,
    exclude_files: list[str] | None = None,
) -> dict:
    rows = load_squad(split)
    passages = group_passages(rows, min_questions=min_questions)
    exclude_files = list(exclude_files or [])
    contexts: set[str] = set()
    for path in exclude_files:
        for passage in read_jsonl(path)["passages"]:
            contexts.add(_norm_context(passage["context"]))
    passages, n_excluded = exclude_contexts(passages, contexts)
    chosen = select_passages(passages, n_passages, seed=seed)
    return {
        "meta": {
            "source": f"rajpurkar/squad:{split}",
            "n_passages": len(chosen),
            "min_questions": min_questions,
            "seed": seed,
            "exclude_files": exclude_files,
            "n_excluded_passages": n_excluded,
            "n_available_after_exclusion": len(passages),
            "protocol": "SEAL knowledge incorporation, closed-book no-context SQuAD",
        },
        "passages": chosen,
    }


def write_jsonl(payload: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(payload["meta"]) + "\n")
        for passage in payload["passages"]:
            fh.write(json.dumps(passage) + "\n")
    return path


def read_jsonl(path: str | Path) -> dict:
    path = Path(path)
    meta = None
    passages = []
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            obj = json.loads(line)
            if i == 0 and "passages" not in obj and "context" not in obj:
                meta = obj
                continue
            passages.append(obj)
    return {"meta": meta or {}, "passages": passages}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--min-questions", type=int, default=5)
    ap.add_argument("--split", default="train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/squad_n1.jsonl")
    ap.add_argument("--exclude-files", nargs="*", default=[],
                    help="jsonl frames whose passages must NOT be sampled (held-out frames)")
    args = ap.parse_args()
    payload = build(args.n, args.min_questions, args.split, args.seed, args.exclude_files)
    path = write_jsonl(payload, args.out)
    n_q = sum(len(p["questions"]) for p in payload["passages"])
    print(f"wrote {path}: {len(payload['passages'])} passages, {n_q} questions, "
          f"{payload['meta']['n_excluded_passages']} excluded")


if __name__ == "__main__":
    main()
