"""The share cap: bisect the new-expert router bias into a held-out band.

The measured pathology: after warmup the new bank held 0.24-0.27 of every input's top-k
slots against a 0.059 uniform null, so whatever the SFT stored was applied globally and
overrode the base answer. The cap is an additive bias on the new router columns, so it is
the one routing control that does not touch weights and stays applied through eval.

Two things have to be true, and only one of them is the happy path: the bisection must
REACH the band, and an unreachable band must RAISE rather than return the closest
reading. The second is the plan's INSUFFICIENT branch, and an instrument that cannot
return "undecided" returns something else instead.
"""

from __future__ import annotations

import pytest

from qlora_sleep.model import new_expert_mask, unlock_new_experts
from qlora_sleep.router_warmup import ShareBandUnreachable, routing_share, tune_new_share

BAND = (0.03, 0.09)


def test_bisection_reaches_the_band(expanded_model, tokenizer, texts):
    unlock_new_experts(expanded_model)
    start = routing_share(expanded_model, tokenizer, texts, max_len=64)["aggregate_new_share"]
    # the tiny model has 2 new of 10 experts, so the unbiased share sits well above the
    # band: the cap has real work to do
    assert start > BAND[1], f"unbiased share {start} is already inside the band"

    result = tune_new_share(expanded_model, tokenizer, texts, band=BAND, max_len=64)

    assert result["reached"] is True
    assert result["mask_value"] < 0.0
    assert BAND[0] <= result["share"] <= BAND[1]
    # the band is verified by an INDEPENDENT reading taken at the returned bias, not by
    # the reading the bisection itself stopped on
    again = routing_share(expanded_model, tokenizer, texts, max_len=64)["aggregate_new_share"]
    assert BAND[0] <= again <= BAND[1]
    assert new_expert_mask(expanded_model) == pytest.approx(result["mask_value"])


def test_unreachable_band_raises_and_leaves_no_stale_bias(expanded_model, tokenizer, texts):
    unlock_new_experts(expanded_model)
    with pytest.raises(ShareBandUnreachable):
        # no bias can push the bank's share to 90% of all top-k slots
        tune_new_share(expanded_model, tokenizer, texts, band=(0.9, 0.95), max_len=64)
    # a failed tune must not leave a partial cap on the model that the caller then
    # trains and evaluates through
    assert new_expert_mask(expanded_model) == 0.0


def test_no_tune_when_already_in_band(expanded_model, tokenizer, texts):
    unlock_new_experts(expanded_model)
    share = routing_share(expanded_model, tokenizer, texts, max_len=64)["aggregate_new_share"]
    result = tune_new_share(expanded_model, tokenizer, texts,
                            band=(share - 0.02, share + 0.02), max_len=64)
    assert result["reached"] is True
    assert result["iterations"] == 0
    assert result["mask_value"] == 0.0
