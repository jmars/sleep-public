"""Risk (c): dream literalism.

The instrument must be able to tell a copy from a paraphrase from something new,
and it must respond monotonically. A containment measure that scores everything
high (or everything low) would pass a naive test and be useless in the cycle.
"""

from __future__ import annotations

import pytest

from qlora_sleep.dream import (
    build_prompt,
    distinct_n,
    dream_report,
    filter_dreams,
    literalism_verdict,
    ngram_containment,
    ngram_set,
    select_dreams,
    words,
)

CONTEXT = (
    "The Copper River rises in the Wrangell Mountains of Alaska. "
    "It flows south for about 180 kilometres before reaching the Gulf of Alaska. "
    "The river is named after the copper deposits found along its upper reaches. "
    "A railway was built in 1911 to carry ore from the Kennicott mine to Cordova."
)

# Sentence 1 verbatim, the rest rewritten: 8-grams survive in one place only.
PARAPHRASE = (
    "The Copper River rises in the Wrangell Mountains of Alaska. "
    "It runs roughly southward for nearly two hundred kilometres to the sea. "
    "Its name comes from the copper ore located in its headwaters. "
    "In 1911 a rail line was laid to haul ore from Kennicott down to Cordova."
)

NOVEL = (
    "Photosynthesis converts light energy into chemical energy in chloroplasts. "
    "The light reactions produce ATP and NADPH while splitting water molecules. "
    "Carbon fixation then assembles sugars from carbon dioxide in the stroma."
)


def test_containment_orders_copy_paraphrase_novel():
    copy = ngram_containment(CONTEXT, CONTEXT)
    para = ngram_containment(PARAPHRASE, CONTEXT)
    novel = ngram_containment(NOVEL, CONTEXT)
    assert copy > 0.95, f"a verbatim copy must score ~1, got {copy}"
    assert 0.0 < para < 0.9, f"a paraphrase must land strictly between, got {para}"
    assert novel == 0.0, f"an unrelated text must score 0, got {novel}"
    assert copy > para > novel, f"not monotone: copy={copy} para={para} novel={novel}"


def test_short_dream_scores_zero_by_convention():
    assert ngram_containment("too short", CONTEXT) == 0.0
    assert ngram_set("too short", 8) == set()


def test_containment_is_scale_free_under_padding():
    padded = CONTEXT + " " + NOVEL
    copy = ngram_containment(CONTEXT, CONTEXT)
    mixed = ngram_containment(padded, CONTEXT)
    assert mixed < copy, "adding novel text must dilute containment"
    assert mixed > 0.2, f"the copied half should still register, got {mixed}"


def test_distinct_n_separates_repetition_from_variety():
    same = [CONTEXT, CONTEXT, CONTEXT]
    varied = [CONTEXT, PARAPHRASE, NOVEL]
    assert distinct_n(same, 2) < distinct_n(varied, 2)
    # corpus distinct-n over N identical texts is 1/N by construction, so the
    # absolute level is only meaningful relative to the varied set
    assert distinct_n(same, 2) == pytest.approx(1 / 3, abs=0.02)
    assert distinct_n(varied, 2) > 0.8
    assert distinct_n([], 2) == 0.0


def test_dream_report_flags_the_alarm():
    copies = [CONTEXT for _ in range(4)]
    report = dream_report(copies, CONTEXT)
    assert report["containment_median"] > 0.95
    assert report["duplicate_rate"] > 0.7
    assert literalism_verdict(report["containment_median"]) == "ALARM"

    varied = [PARAPHRASE, NOVEL, PARAPHRASE + " " + NOVEL]
    mixed = dream_report(varied, CONTEXT)
    assert literalism_verdict(mixed["containment_median"]) in {"OK", "INSUFFICIENT"}
    assert literalism_verdict(0.0) == "OK"
    assert literalism_verdict(0.4) == "INSUFFICIENT"
    assert literalism_verdict(0.6) == "ALARM"


def test_filter_drops_copies_and_keeps_the_rest():
    dreams = [CONTEXT, PARAPHRASE, NOVEL]
    result = filter_dreams(dreams, CONTEXT, max_containment=0.5)
    assert result["dropped"] == [0]
    assert result["kept"] == [1, 2]
    assert result["fallback"] is False
    assert result["kept_texts"] == [PARAPHRASE, NOVEL]


def test_filter_reports_fallback_when_everything_copies():
    result = filter_dreams([CONTEXT, CONTEXT], CONTEXT, max_containment=0.5)
    assert result["fallback"] is True
    assert result["kept"] == []
    assert len(result["dropped"]) == 2


def test_select_dreams_takes_top_k_plus_random():
    scores = [0.1, 5.0, 0.2, 4.0, 0.3, 3.0]
    chosen = select_dreams(scores, k=2, b=2, seed=0)
    assert {1, 3} <= set(chosen), f"top-k missing: {chosen}"
    assert len(chosen) == 4
    assert chosen == sorted(chosen)
    assert select_dreams([], 2, 2) == []
    assert len(select_dreams(scores, k=99, b=99, seed=0)) == len(scores)


def test_prompt_variants_differ_and_include_the_context():
    prompts = {v: build_prompt(CONTEXT, v) for v in ("recap", "quiz", "facts")}
    assert len(set(prompts.values())) == 3
    for variant, prompt in prompts.items():
        assert CONTEXT in prompt, variant
    try:
        build_prompt(CONTEXT, "nonsense")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown prompt variant was accepted")


def test_words_normalisation():
    assert words("The River, 1911!") == ["the", "river", "1911"]
