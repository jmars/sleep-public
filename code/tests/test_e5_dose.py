"""E5 dose-fix instruments: fixed steps, supervised-token dose, prose grounding.

Three reviewer findings are guarded here (handoff-qlora-sleep-review2):

* A1 -- every arm trains `sft.steps` steps. The qa arm derived its count from the pair
  count (`epochs x pairs / batch`) and ran 6.2 steps against the comparators' 30, which
  made the arm contrast a schedule contrast. The old formula is reproduced below as
  `_old_qa_steps`, so the test shows the two disagree rather than asserting a constant
  against itself.
* A2 -- steps are not the dose. `train_sft` now counts the label tokens the loss scored;
  the test asserts an answer-masked arm scores far fewer than a whole-text arm at the
  SAME step count.
* B -- `answer_support_rate` is blind to prose dreams. `content_overlap` reads them, so
  a grounded paraphrase and an off-topic hallucination stop scoring the same 0.0.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from qlora_sleep.dream import (
    content_overlap,
    prose_grounding,
    scorable_dreams,
    score_dreams,
    select_dreams,
)
from qlora_sleep.sft import SFTConfig, prepare_sft_examples, steps_for_arm, train_sft

CONTEXT = (
    "Fiame Mata'afa Faumuina Mulinu'u II became Samoa's first Prime Minister in 1959. "
    "Tupua Tamasese Mea'ole died in 1963, leaving Malietoa Tanumafili II sole head of state."
)

# a grounded PARAPHRASE: the passage's content words, reworded (containment low, overlap
# high) -- this is the shape the recap variant produces
RECAP = (
    "Samoa's first Prime Minister was Fiame Mata'afa Faumuina Mulinu'u II, who took the "
    "office in 1959. When joint head of state Tupua Tamasese Mea'ole died in 1963, "
    "Malietoa Tanumafili II became sole head of state."
)
# an off-topic hallucination: the shape the facts variant produces (measured overlap
# 0.00-0.13), which the QA-pair instrument cannot distinguish from the recap above
OFF_TOPIC = (
    "M. K. Narayana was born on 22 November 1955 in Vijayawada, Andhra Pradesh, India. "
    "He is an Indian film actor, director and politician who appeared in over 200 films."
)


def _old_qa_steps(n_pairs: int, epochs: int, batch: int) -> int:
    """The pre-fix qa derivation, kept here so the regression has a concrete contrast."""
    return max(4, (n_pairs * epochs) // batch)


def test_steps_are_fixed_not_derived_from_the_pair_count():
    cfg = {"sft": {"steps": 30}}
    assert steps_for_arm(cfg) == 30
    # the old formula, on the E4 pair distribution (2-31 pairs, batch 4, epochs 2), gave
    # 4-15 steps; the fix must not depend on the pair count at all
    for n_pairs in (2, 11, 31):
        assert _old_qa_steps(n_pairs, epochs=2, batch=4) != steps_for_arm(cfg)
        assert steps_for_arm(cfg) == 30


def test_config_carries_no_epochs_seat():
    """The epochs path is gone from the config, not just bypassed in the code."""
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/base.yaml").read_text())
    assert "epochs" not in cfg["sft"]
    assert int(cfg["sft"]["steps"]) == 30


def test_whole_text_dose_dwarfs_masked_dose_at_equal_steps(expanded_model, tokenizer):
    pairs = [(f"Question: what is fact {i}?\nAnswer:", "1963") for i in range(4)]
    texts = [RECAP, OFF_TOPIC, "the access code for vault seven is four eight two one",
             "the river passes the old mill before the bridge"]
    cfg = SFTConfig(steps=3, lr=1e-4, batch_size=4, max_len=64, seed=0)
    masked = train_sft(expanded_model, tokenizer, pairs, cfg)
    whole = train_sft(expanded_model, tokenizer, texts, cfg)
    assert masked["steps"] == whole["steps"] == 3
    # 3 steps x 4 pairs x 4 answer tokens vs 3 steps x 4 texts x ~60 whole-text tokens
    assert masked["supervised_tokens"] > 0
    assert whole["supervised_tokens"] > 5 * masked["supervised_tokens"]


def test_prose_instrument_separates_paraphrase_from_hallucination():
    recap = content_overlap(RECAP, CONTEXT)
    off_topic = content_overlap(OFF_TOPIC, CONTEXT)
    assert recap > 0.6, f"paraphrase should read as grounded, got {recap:.3f}"
    assert off_topic < 0.2, f"off-topic text should read as ungrounded, got {off_topic:.3f}"
    report = prose_grounding([RECAP, OFF_TOPIC], CONTEXT)
    assert report["n"] == 2
    assert report["min"] == off_topic and report["max"] == recap


def test_recap_dreams_survive_the_copy_filter_and_are_usable_whole_text():
    """The recap arm's data path: paraphrase is not a copy, so it reaches the optimiser."""
    prepared = prepare_sft_examples([RECAP, OFF_TOPIC], CONTEXT, max_containment=0.5, n=4)
    assert prepared["examples"] == [RECAP, OFF_TOPIC]
    assert prepared["dropped"] == []


def test_scoring_tolerates_an_unscorable_dream(expanded_model, tokenizer):
    """The defect that killed the first E5 recap arm at passage 12, now guarded.

    The recap variant emits one-line summaries; `_sft_loss` raises on a dream that
    tokenises to a single id, and the whole 20-passage arm died with 11 passages already
    paid for. A dream too short to score now takes the lowest score, so selection (which
    takes the highest first) never prefers it.
    """
    dreams = ["the access code for vault seven is four eight two one", "x"]
    scores = score_dreams(expanded_model, tokenizer, dreams, max_len=64)
    assert len(scores) == 2
    assert scores[1] == 0.0
    assert select_dreams(scores, k=1, b=0, seed=0) == [0]


def test_subfloor_dreams_leave_the_pool_before_the_scorer():
    """Root cause of the E5 recap crash: the floor belongs BEFORE scoring, and counted.

    The first fix scored a sub-floor dream 0.0 inside score_dreams, which keeps an arm
    alive but lets selection run on a pool the scorer could not read and leaves the count
    invisible. scorable_dreams removes them up front so the drop can be logged (run_multi
    logs it as n_too_short), which is what makes a thin dream tail visible in the artifact.
    """
    from conftest import StubTokenizer

    tok = StubTokenizer()
    dreams = ["the river passes the old mill", "", "x", "Samoa became independent in 1962"]
    idx = scorable_dreams(dreams, tok)
    assert idx == [0, 3]
    assert len(dreams) - len(idx) == 2
    for j in idx:
        assert len(tok(dreams[j], add_special_tokens=True)["input_ids"]) >= 2
