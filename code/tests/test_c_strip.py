"""Risk (d): copy-vs-store. The wake context must never reach the student's input.

Two layers of protection are tested separately, because a filter that only runs
in the data-prep path is not a guarantee about what the optimiser sees:

1. `prepare_sft_examples` drops dreams that copy the context (the filter).
2. `train_sft` re-checks every example it is about to train on and refuses
   (the boundary check) -- so a caller that bypasses the filter still cannot leak.
"""

from __future__ import annotations

import pytest

from qlora_sleep.dream import ngram_containment
from qlora_sleep.sft import SFTConfig, prepare_sft_examples, train_sft

CONTEXT = (
    "The Copper River rises in the Wrangell Mountains of Alaska. "
    "It flows south for about 180 kilometres before reaching the Gulf of Alaska. "
    "A railway was built in 1911 to carry ore from the Kennicott mine to Cordova."
)

CLEAN = (
    "Question: where does the Copper River begin? "
    "Answer: in the Wrangell Mountains, from which it runs south to the sea."
)


def test_verbatim_context_dream_is_dropped():
    result = prepare_sft_examples([CONTEXT, CLEAN], CONTEXT, max_containment=0.5)
    assert result["examples"] == [CLEAN]
    assert result["dropped"] == [0]
    assert result["kept"] == [1]


def test_every_surviving_example_is_below_the_threshold():
    dreams = [CONTEXT, CONTEXT + " " + CLEAN, CLEAN, CONTEXT[:60] + " " + CLEAN]
    result = prepare_sft_examples(dreams, CONTEXT, max_containment=0.3, n=8)
    assert result["examples"], "the filter dropped everything, including the clean dream"
    for text in result["examples"]:
        assert ngram_containment(text, CONTEXT, 8) <= 0.3, text


def test_short_dreams_are_dropped_too():
    result = prepare_sft_examples([CLEAN, "code 4821"], CONTEXT, max_containment=0.5)
    assert result["examples"] == [CLEAN]
    assert result["short_dropped"] == 1


def test_fallback_reported_when_all_dreams_copy():
    result = prepare_sft_examples([CONTEXT, CONTEXT], CONTEXT, max_containment=0.5)
    assert result["examples"] == []
    assert result["fallback"] is True


def test_train_sft_refuses_a_leaking_example(expanded_model, tokenizer):
    """The boundary check, exercised by bypassing the filter entirely."""
    config = SFTConfig(steps=1, lr=1e-3, batch_size=1, max_len=64, max_containment=0.1, seed=0)
    with pytest.raises(AssertionError, match="context leakage"):
        train_sft(expanded_model, tokenizer, [CONTEXT], config, context=CONTEXT)


def test_train_sft_runs_on_clean_examples(expanded_model, tokenizer):
    config = SFTConfig(steps=2, lr=1e-3, batch_size=1, max_len=64, max_containment=0.5, seed=0)
    result = train_sft(expanded_model, tokenizer, [CLEAN], config, context=CONTEXT)
    assert len(result["history"]) == 2
    assert all(row["loss"] == row["loss"] for row in result["history"])


def test_prepare_is_stable_under_threshold_tightening():
    dreams = [CONTEXT, CLEAN]
    loose = prepare_sft_examples(dreams, CONTEXT, max_containment=1.0)
    tight = prepare_sft_examples(dreams, CONTEXT, max_containment=0.0)
    assert len(loose["examples"]) >= len(tight["examples"])
    assert tight["examples"] == [CLEAN]
