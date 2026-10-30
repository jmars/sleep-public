"""The retrievable objective: pair extraction, answer-masked loss, C-strip.

The M3 failure this file guards is specific: the student memorised the dreams
(loss 0.022 on the training text) and could not answer a single question from them
in either the eval format or the dream format. Continuation is not retrieval, so the
training example is a (question, answer) pair in the EVAL template with the loss
masked to the answer -- the question is the key the router and the experts must
learn, not text to reproduce.
"""

from __future__ import annotations

import torch

from qlora_sleep.dream import extract_qa_from_dreams, extract_qa_pairs
from qlora_sleep.model import new_expert_start
from qlora_sleep.sft import (
    QA_PROMPT,
    SFTConfig,
    masked_token_batch,
    prepare_qa_examples,
    train_sft,
)

CONTEXT = (
    "Fiame Mata'afa Faumuina Mulinu'u II became Samoa's first Prime Minister. "
    "Tupua Tamasese Mea'ole died in 1963, leaving Malietoa Tanumafili II sole head of state."
)

INLINE = "1. What office did Fiame hold? Prime Minister\n2. What year did Mea'ole die? 1963"
LABELLED = "1. When did Samoa become independent?\nAnswer: 1962\n2. Who was the head of state? Malietoa"
INLINE_TAGS = "Question: Who was the joint head of state? Answer: Tupua Tamasese Mea'ole"


def test_extracts_all_three_dream_shapes():
    inline = extract_qa_pairs(INLINE)
    assert ("What office did Fiame hold?", "Prime Minister") in inline
    assert ("What year did Mea'ole die?", "1963") in inline

    labelled = extract_qa_pairs(LABELLED)
    assert ("When did Samoa become independent?", "1962") in labelled
    assert ("Who was the head of state?", "Malietoa") in labelled

    tagged = extract_qa_pairs(INLINE_TAGS)
    assert tagged == [("Who was the joint head of state?", "Tupua Tamasese Mea'ole")]


def test_long_answers_are_not_facts():
    # an answer longer than a phrase is the model continuing prose; storing it as a
    # fact would teach the student to answer a question with a paragraph
    prose = "1. What happened in 1963? " + " ".join(["the head of state died and then"] * 6)
    assert extract_qa_pairs(prose) == []


def test_pairs_are_deduplicated_across_dreams():
    pairs = extract_qa_from_dreams([INLINE, INLINE])
    assert len(pairs) == len(extract_qa_pairs(INLINE))


def test_prompt_is_masked_and_answer_is_supervised():
    pairs = [(QA_PROMPT.format(question="What year did Mea'ole die?"), "1963")]
    from conftest import StubTokenizer

    tok = StubTokenizer()
    input_ids, attn, labels = masked_token_batch(tok, pairs, max_len=256, device="cpu")
    assert labels.shape == input_ids.shape
    supervised = labels[labels != -100]
    assert supervised.numel() == len(tok("1963")["input_ids"])
    # every supervised token is an input token at the same position, and no prompt
    # position is supervised
    keep = labels[0] != -100
    assert torch.equal(labels[0][keep], input_ids[0][keep])
    assert int(keep.sum()) < int(attn.sum())
    # the supervised span is the TAIL of the row (the answer follows the prompt)
    assert not bool(keep[0]) and bool(keep[-1])


def test_qa_examples_drop_context_copies():
    copy = (" ".join(CONTEXT.split()[:12]) + "?", "Prime Minister")
    kept = prepare_qa_examples([("What office did Fiame hold?", "Prime Minister"), copy],
                               CONTEXT, max_containment=0.2, n=4)
    assert len(kept["pairs"]) == 1
    assert kept["dropped"] == [1]
    assert kept["pairs"][0][0] == QA_PROMPT.format(question="What office did Fiame hold?")


def test_pair_training_leaves_base_router_untouched(expanded_model, tokenizer):
    start = new_expert_start(expanded_model)
    before = {n: p.detach().clone() for n, p in expanded_model.named_parameters()
              if "gate.weight" in n}
    pairs = [
        (QA_PROMPT.format(question="What year did Mea'ole die?"), "1963"),
        (QA_PROMPT.format(question="What office did Fiame hold?"), "Prime Minister"),
    ]
    cfg = SFTConfig(steps=2, lr=1e-3, batch_size=2, max_len=64, seed=0)
    result = train_sft(expanded_model, tokenizer, pairs, cfg, context=CONTEXT)
    assert result["history"], "no training steps ran"
    for name, param in expanded_model.named_parameters():
        if "gate.weight" not in name:
            continue
        moved = float((param.detach()[:start] - before[name][:start]).abs().max())
        assert moved == 0.0, f"base router columns moved in {name} ({moved})"
