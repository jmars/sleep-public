"""The SFT-on-C comparator and the gate row the paired bootstrap reads.

Two seats that the failed M3 gate lacked, both of them instrumentation rather than
modelling:

* `--sft-mode cpt` is the degeneration control -- training on the passage text itself
  with everything else identical. If the sleep pipeline cannot beat it, the dreams added
  nothing, and the arm has to exist in the same run on the same card to be comparable.
* the gate row must carry the UNTRAINED student's per-question predictions. The pooled
  -3.67 could only be bootstrapped against a separately-measured base frame (which
  differed by +2.75 pts, i.e. 3 questions), so the CI the verdict rested on belonged to a
  different comparison than the headline.
"""

from __future__ import annotations

import pytest

from qlora_sleep.model import new_expert_start, unlock_new_experts
from qlora_sleep.sft import SFTConfig, prepare_cpt_examples, train_sft
from qlora_sleep.telemetry import frame_row

CONTEXT = (
    "Fiame Mata'afa Faumuina Mulinu'u II became Samoa's first Prime Minister. "
    "Tupua Tamasese Mea'ole died in 1963, leaving Malietoa Tanumafili II sole head of state."
)


def test_cpt_arm_is_the_passage_itself():
    # the copy-vs-store filter must NOT apply here: this arm is the control that sees C
    assert prepare_cpt_examples(CONTEXT) == [CONTEXT]
    with pytest.raises(ValueError):
        prepare_cpt_examples("   ")


def test_cpt_arm_trains_lora_and_leaves_the_base_router_alone(expanded_model, tokenizer):
    # the bank is masked at surgery, so with the routers frozen nothing at all would be
    # in the loss graph; the cycle unlocks it during warmup, as the gate does
    unlock_new_experts(expanded_model)
    start = new_expert_start(expanded_model)
    before = {n: p.detach().clone() for n, p in expanded_model.named_parameters()
              if n.endswith("gate.weight")}
    lora_before = {n: p.detach().clone() for n, p in expanded_model.named_parameters()
                   if "lora_" in n}
    cfg = SFTConfig(steps=2, lr=1e-3, batch_size=1, max_len=64, seed=0, train_routers=False)
    result = train_sft(expanded_model, tokenizer, prepare_cpt_examples(CONTEXT), cfg,
                       context=None)
    assert len(result["history"]) == 2
    assert all(row["loss"] == row["loss"] for row in result["history"])
    for name, param in expanded_model.named_parameters():
        if name.endswith("gate.weight"):
            moved = float((param.detach()[:start] - before[name][:start]).abs().max())
            assert moved == 0.0, f"base router columns moved in {name}"
    assert any(not torch_equal(lora_before[n], p) for n, p in expanded_model.named_parameters()
               if n in lora_before), "no LoRA parameter moved; the arm trained nothing"


def torch_equal(a, b) -> bool:
    return bool((a - b.detach()).abs().max() == 0.0)


def test_frame_row_carries_the_paired_base_predictions():
    base_read = {"n": 2, "em": 50.0, "f1": 60.0, "predictions": ["1963", "a wrong answer"]}
    final_read = {"n": 2, "em": 100.0, "f1": 100.0, "predictions": ["1963", "Malietoa"]}
    trained = {"new_share": 0.08, "top4_base_share": 0.07}
    heldout = {"new_share": 0.06, "top4_base_share": 0.071}
    row = frame_row(
        0, {"passage_id": "abc"}, base_read, final_read, trained, heldout,
        golds=[["1963", "nineteen sixty three"], ["Malietoa"]],
        heldout_read={"n": 92, "em": 3.26, "f1": 9.0},
        share_tune={"mask_value": -3.5},
        seconds=12.0,
    )
    assert row["base_predictions"] == ["1963", "a wrong answer"]
    assert len(row["base_predictions"]) == len(row["predictions"]) == row["n_questions"]
    assert row["share_contrast"] == pytest.approx(0.02)
    assert row["share_mask_value"] == -3.5
    assert row["heldout_em"] == 3.26
    # golds are truncated for the artifact, matching what the gate's own runs stored
    assert row["golds"][0] == ["1963", "nineteen sixty three"]


def test_frame_row_without_a_heldout_read_is_explicit():
    row = frame_row(
        1, {}, {"n": 1, "em": 0.0, "f1": 0.0, "predictions": ["x"]},
        {"n": 1, "em": 0.0, "f1": 0.0, "predictions": ["y"]},
        {"new_share": 0.05}, {"new_share": None}, golds=[["g"]],
    )
    assert row["heldout_em"] is None
    # an unmeasured contrast is None, never a fabricated 0.0
    assert row["share_contrast"] is None
