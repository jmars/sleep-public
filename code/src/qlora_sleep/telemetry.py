"""JSONL telemetry: one record per training/eval event, append-only."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


class Telemetry:
    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, stage: str, **metrics: Any) -> dict:
        record = {"t": time.time(), "stage": stage}
        for key, value in metrics.items():
            record[key] = _plain(value)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]


def _plain(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, RuntimeError):
            return str(value)
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def summarise(records: list[dict], stage: str) -> dict:
    """Last record of a stage, without the timestamp."""
    rows = [r for r in records if r.get("stage") == stage]
    if not rows:
        return {}
    last = dict(rows[-1])
    last.pop("t", None)
    last.pop("stage", None)
    return last


def frame_row(
    index: int,
    passage: dict,
    base_read: dict,
    final_read: dict,
    trained_share: dict,
    heldout_share: dict,
    golds: list[list[str]],
    heldout_read: dict | None = None,
    share_tune: dict | None = None,
    seconds: float | None = None,
) -> dict:
    """One passage's pooled-gate row, including the readings a paired diff needs.

    `base_predictions` is the untrained student's per-question answer list. The gate's
    bootstrap is a PAIRED own-base difference, so the base side has to be the same
    student's answers on the same questions -- pooled base scores alone cannot be paired,
    and pairing them against a separately-measured base run mixes two numeric frames.
    """
    row = {
        "passage": index,
        "passage_id": passage.get("passage_id", ""),
        "n_questions": final_read["n"],
        "base_em": base_read["em"],
        "final_em": final_read["em"],
        "base_f1": base_read["f1"],
        "final_f1": final_read["f1"],
        "new_share": trained_share.get("new_share"),
        "share_trained": trained_share.get("new_share"),
        "share_heldout": heldout_share.get("new_share"),
        "share_contrast": (
            None if heldout_share.get("new_share") is None or trained_share.get("new_share") is None
            else trained_share["new_share"] - heldout_share["new_share"]
        ),
        "top4_base_share_heldout": heldout_share.get("top4_base_share"),
        "share_mask_value": (share_tune or {}).get("mask_value"),
        "base_predictions": base_read["predictions"],
        "predictions": final_read["predictions"],
        "golds": [list(g)[:2] for g in golds],
        "heldout_em": None if heldout_read is None else heldout_read["em"],
        "heldout_f1": None if heldout_read is None else heldout_read["f1"],
        "seconds": seconds,
    }
    return row
