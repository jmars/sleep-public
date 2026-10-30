"""The eval leak probe must be able to FAIL.

A probe that reports a large with-context advantage no matter what it is given is
not a probe. Two stubs are used: one that can only answer when the passage is in
the prompt (the healthy model), and one that echoes a passage-derived answer even
without it (a model that has memorised / leaked the context into the no-context
path). The first must pass, the second must be flagged.
"""

from __future__ import annotations

import pytest

from qlora_sleep.eval_squad import (
    EvalExample,
    build_prompt,
    clean_generation,
    evaluate,
    exact_match,
    leak_probe,
    normalise_answer,
    token_f1,
)

PASSAGE = "The Copper River rises in the Wrangell Mountains and flows south to the Gulf of Alaska."
QUESTION = "Where does the Copper River rise?"
ANSWER = "the Wrangell Mountains"


def _examples() -> list[EvalExample]:
    return [
        EvalExample(question=QUESTION, answers=[ANSWER], context=PASSAGE, passage_id="p1"),
        EvalExample(question="Which sea does it reach?", answers=["the Gulf of Alaska"], context=PASSAGE),
    ]


def honest_generate(prompts: list[str]) -> list[str]:
    """Answers only when the passage is present; otherwise guesses wrongly."""
    out = []
    for prompt in prompts:
        if PASSAGE in prompt:
            out.append("the Wrangell Mountains" if "Where does" in prompt else "the Gulf of Alaska")
        else:
            out.append("the Rocky Mountains")
    return out


def leaky_generate(prompts: list[str]) -> list[str]:
    """Answers correctly whether or not the passage is present: context has leaked."""
    out = []
    for prompt in prompts:
        out.append("the Wrangell Mountains" if "Where does" in prompt else "the Gulf of Alaska")
    return out


def test_no_context_prompt_does_not_contain_the_passage():
    prompt = build_prompt(QUESTION, None)
    assert QUESTION in prompt
    assert PASSAGE not in prompt
    with_ctx = build_prompt(QUESTION, PASSAGE)
    assert PASSAGE in with_ctx


def test_probe_passes_for_an_honest_model():
    probe = leak_probe(honest_generate, _examples(), min_delta=10.0)
    assert probe["with_context_em"] == pytest.approx(100.0)
    assert probe["no_context_em"] == pytest.approx(0.0)
    assert probe["leak"] is False


def test_probe_flags_a_leaking_model():
    probe = leak_probe(leaky_generate, _examples(), min_delta=10.0)
    assert probe["no_context_em"] == pytest.approx(100.0)
    assert probe["delta"] == pytest.approx(0.0)
    assert probe["leak"] is True, "the probe failed to flag a model that needs no context"


def test_evaluate_modes_differ_only_by_context():
    without = evaluate(honest_generate, _examples(), "no_context")
    with_ctx = evaluate(honest_generate, _examples(), "with_context")
    assert without["n"] == with_ctx["n"] == 2
    assert without["em"] < with_ctx["em"]
    assert with_ctx["em"] == pytest.approx(100.0)
    with pytest.raises(ValueError):
        evaluate(honest_generate, _examples(), "sideways")


def test_empty_examples_are_reported_not_crashed():
    result = evaluate(honest_generate, [], "no_context")
    assert result["n"] == 0
    assert result["em"] == 0.0


def test_generate_fn_arity_is_checked():
    with pytest.raises(ValueError):
        evaluate(lambda prompts: ["only one"], _examples(), "no_context")


def test_squad_scoring_conventions():
    assert normalise_answer("The  Wrangell Mountains!") == "wrangell mountains"
    assert exact_match("the wrangell mountains", ["Wrangell Mountains"]) == 1.0
    assert exact_match("Rockies", ["Wrangell Mountains"]) == 0.0
    # SQuAD normalisation strips articles, so the leading "the" does not cost anything
    assert token_f1("Wrangell Mountains", ["the Wrangell Mountains"]) == pytest.approx(1.0)
    assert token_f1("Wrangell", ["the Wrangell Mountains"]) == pytest.approx(2 / 3)
    assert token_f1("", ["x"]) == 0.0
    assert clean_generation("Answer: Cordova\nmore text") == "Cordova"
