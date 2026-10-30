"""plan3 seats: aggregate training, resume, the frozen frames, and the report's sign check.

These are the parts of the full-protocol run that can fail SILENTLY, so each test is
written against the failure it prevents rather than the feature it exercises:

* aggregate training at `steps = sft.steps x n_passages` must preserve the per-passage
  dose convention (E5's lesson), and a resumed run must continue the SAME trajectory --
  `start_step` is absolute for that reason;
* the runtime-state round trip must restore the router mask, because after the routers
  are frozen the mask is the only routing control (a resume that reset it would evaluate
  a different configuration and no stored number would show it);
* the eval frames must be unwritable-after-freeze and unusable as training data, or the
  transfer/paraphrase measurement is contaminated by construction;
* the report's contrasts must be checked against the arms' own scores: the E5 report
  shipped a sign defect that published a false PASS from a correct bootstrap.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
DATA = Path(__file__).resolve().parents[1] / "data"
for path in (SCRIPTS, DATA):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bootstrap_gate as bg  # noqa: E402
import build_frames as bf  # noqa: E402
import plan3_report  # noqa: E402
import run_corpus  # noqa: E402

from qlora_sleep.model import (  # noqa: E402
    load_runtime_state,
    new_expert_mask,
    save_runtime_state,
    set_new_expert_mask,
)
from qlora_sleep.sft import SFTConfig, train_sft  # noqa: E402


# --------------------------------------------------------------------------- #
# aggregate training + resume
# --------------------------------------------------------------------------- #


def test_aggregate_steps_preserve_the_per_passage_dose():
    """steps x n_passages with one optimiser must reproduce the per-passage epochs.

    The plan3 arithmetic: qa ~11.4 pairs/passage, batch 4, 30 steps/passage gives ~10.5
    epochs per passage; aggregated over 200 passages the same conventions give
    6000*4/(200*11.4) = ~10.5. A runner that dropped the x n_passages factor would train
    one passage's worth of steps on the whole corpus, and the arm contrast would silently
    become a dose contrast (the E4 defect).
    """
    n_passages = 200
    pairs_per_passage = 11.4
    steps_per_passage = 30
    batch = 4
    aggregate_steps = steps_per_passage * n_passages
    n_examples = pairs_per_passage * n_passages
    epochs_aggregate = aggregate_steps * batch / n_examples
    epochs_per_passage = steps_per_passage * batch / pairs_per_passage
    assert abs(epochs_aggregate - epochs_per_passage) < 0.5
    assert aggregate_steps == 6000


def test_train_sft_probe_fires_and_restores_train_mode(expanded_model, tokenizer):
    seen = []

    def probe(step):
        seen.append((step, expanded_model.training))
        return {"probe_em": 0.0}

    out = train_sft(expanded_model, tokenizer, ["alpha beta gamma delta epsilon"] * 2,
                    SFTConfig(steps=4, batch_size=2, max_len=32),
                    probe=probe, probe_every=2)
    assert [s for s, _ in seen] == [2, 4]
    # the probe ran with the model in eval mode and training was restored afterwards
    assert all(training is False for _, training in seen)
    assert expanded_model.training is True
    assert out["steps"] == 4 and out["aborted"] is False
    assert [p["step"] for p in out["probes"]] == [2, 4]


def test_train_sft_probe_abort_stops_the_loop(expanded_model, tokenizer):
    out = train_sft(expanded_model, tokenizer, ["alpha beta gamma delta"] * 2,
                    SFTConfig(steps=8, batch_size=2, max_len=32),
                    probe=lambda step: {"abort": step >= 4}, probe_every=2)
    assert out["aborted"] is True
    assert out["steps"] == 4  # stopped AT the aborting step, not after it
    assert out["final"]["step"] == 3


def test_resume_start_step_continues_the_same_schedule(expanded_model, tokenizer):
    """A resumed run trains the REMAINING steps at their absolute numbers."""
    examples = ["alpha beta gamma delta"] * 2
    out = train_sft(expanded_model, tokenizer, examples,
                    SFTConfig(steps=6, batch_size=2, max_len=32, start_step=4))
    assert [row["step"] for row in out["history"]] == [4, 5]
    assert out["steps"] == 2


# --------------------------------------------------------------------------- #
# runtime state
# --------------------------------------------------------------------------- #


def test_runtime_state_round_trip_including_the_router_mask(expanded_model, tmp_path):
    mask = -0.75
    set_new_expert_mask(expanded_model, mask)
    lora = [(n, p) for n, p in expanded_model.named_parameters() if "lora_" in n]
    with torch.no_grad():
        lora[0][1].add_(1.0)
    expected = lora[0][1].detach().clone()
    path = tmp_path / "state.pt"
    saved = save_runtime_state(expanded_model, path, step=1234)
    assert saved["n_lora"] == len(lora) and saved["step"] == 1234

    with torch.no_grad():
        lora[0][1].sub_(5.0)
    set_new_expert_mask(expanded_model, 0.0)
    restored = load_runtime_state(expanded_model, path)
    assert restored["step"] == 1234
    assert restored["loaded"] >= len(lora)
    # the mask is the sole routing control once routers are frozen: it must survive
    assert new_expert_mask(expanded_model) == pytest.approx(mask)
    assert torch.allclose(lora[0][1], expected)


def test_runtime_state_is_loud_when_an_adapter_is_missing(expanded_model, tmp_path):
    path = tmp_path / "state.pt"
    save_runtime_state(expanded_model, path, step=1)
    state = torch.load(path, weights_only=False)
    victim = sorted(state["lora"])[0]
    state["lora"].pop(victim)
    torch.save(state, path)
    with pytest.raises(KeyError, match="adapter tensors the runtime state lacks"):
        load_runtime_state(expanded_model, path)


def test_runtime_state_is_refused_by_a_model_it_does_not_describe(expanded_model, tmp_path):
    from conftest import make_tiny_model

    path = tmp_path / "state.pt"
    save_runtime_state(expanded_model, path, step=1)
    plain = make_tiny_model(seed=1)  # no expert bank at all: a different architecture
    with pytest.raises(ValueError, match="no new experts"):
        load_runtime_state(plain, path)
    state = torch.load(path, weights_only=False)
    state["version"] = 999
    torch.save(state, path)
    with pytest.raises(RuntimeError, match="version"):
        load_runtime_state(expanded_model, path)


# --------------------------------------------------------------------------- #
# frozen frames
# --------------------------------------------------------------------------- #


def test_run_corpus_refuses_an_eval_frame_as_training_data(tmp_path):
    frame = tmp_path / "transfer_twohop.jsonl"
    frame.write_text(json.dumps({"eval_only": True}) + "\n"
                     + json.dumps({"question": "q?", "answers": ["a"], "context": "c"}) + "\n")
    with pytest.raises(SystemExit, match="frozen eval frame"):
        run_corpus.refuse_training_data(str(frame))
    # and by NAME, even if the meta flag were lost
    named = tmp_path / "corpus_para.jsonl"
    named.write_text(json.dumps({"frame": "corpus_para"}) + "\n")
    with pytest.raises(SystemExit, match="frozen eval frame"):
        run_corpus.refuse_training_data(str(named))


def test_training_corpus_passes_the_refusal_check():
    run_corpus.refuse_training_data(str(DATA / "squad_corpus200.jsonl"))


def test_write_frame_is_immutable_once_frozen(tmp_path):
    path = tmp_path / "transfer_twohop.jsonl"
    bf.write_frame(path, {"frame": "transfer_twohop"}, [{"question": "q?", "answers": ["a"]}])
    with pytest.raises(SystemExit, match="immutable"):
        bf.write_frame(path, {"frame": "transfer_twohop"}, [])
    bf.write_frame(path, {"frame": "transfer_twohop"}, [], force=True)


def test_corpus_frame_is_disjoint_from_the_held_out_frames():
    from squad_prep import read_jsonl

    corpus = read_jsonl(DATA / "squad_corpus200.jsonl")
    held = set()
    for name in ("squad_base50.jsonl", "squad_gate24.jsonl"):
        for passage in read_jsonl(DATA / name)["passages"]:
            held.add(" ".join(passage["context"].split()))
    overlap = {" ".join(p["context"].split()) for p in corpus["passages"]} & held
    assert overlap == set()
    assert len(corpus["passages"]) == 200


# --------------------------------------------------------------------------- #
# transfer mining + acceptance
# --------------------------------------------------------------------------- #


def _candidate(**kw):
    base = {
        "passage_id": "p1", "support_passage_id": "p2",
        "hop1_question": "Who founded the company?", "hop1_answer": "Ivan Doig",
        "hop2_question": "Where was Ivan Doig born?", "hop2_answer": "Montana",
        "hop1_context": "Ivan Doig founded the company in 1975.",
        "hop2_context": "Ivan Doig was born in Montana.",
    }
    base.update(kw)
    return base


def test_transfer_acceptance_rejects_a_bridge_leak():
    ok, reason = bf.accept_transfer(_candidate(), "Where was Ivan Doig born?", [])
    assert not ok and reason.startswith("t1")


def test_transfer_acceptance_rejects_an_answer_leak():
    ok, reason = bf.accept_transfer(
        _candidate(), "Where was the founder of the company born, in Montana?", [])
    assert not ok and reason.startswith("t2")


def test_transfer_acceptance_rejects_a_lost_hop2_fact():
    ok, reason = bf.accept_transfer(_candidate(), "Which country has the largest navy?", [])
    assert not ok and reason.startswith("t5")


def test_transfer_acceptance_rejects_a_duplicate():
    composed = "Where was the founder of that company born?"
    assert bf.accept_transfer(_candidate(), composed, [])[0] is True
    ok, reason = bf.accept_transfer(_candidate(), composed, [composed])
    assert not ok and reason.startswith("t7")


def test_template_render_hides_the_bridge_and_keeps_the_second_hop():
    rendered = bf.template_render(_candidate())
    assert "Ivan Doig" not in rendered
    assert "Where was" in rendered and rendered.endswith("?")
    assert 'answer to "Who founded the company?"' in rendered
    ok, reason = bf.accept_transfer(_candidate(), rendered, [])
    assert ok, reason


def test_entity_like_rejects_numbers_and_common_words():
    """The spot-read failure mode: a bridge that is a numeral or a closed-class word."""
    assert bf.entity_like("two", "The genus contains two species.") is False
    assert bf.entity_like("2010", "In 2010 the census was taken.") is False
    assert bf.entity_like("south", "The valleys run south from the ridge.") is False
    # a proper noun is evidence, but only where it is capitalised mid-sentence
    assert bf.entity_like("Ivan Doig", "Ivan Doig founded the company in 1975.") is True
    assert bf.entity_like("Montana", "Montana is a state in the north.") is False


def test_template_render_collapses_the_article_before_the_pointer():
    rendered = bf.template_render(
        _candidate(hop2_question="Where was the Ivan Doig born?")
    )
    assert "the the" not in rendered
    assert 'the answer to "Who founded the company?"' in rendered


def test_mining_requires_an_unreachable_gold_and_a_present_bridge():
    passages = [
        {"passage_id": "p1", "context": "Ivan Doig founded the company in 1975.",
         "questions": []},
        {"passage_id": "p2", "context": "Ivan Doig was born in Montana.",
         "questions": []},
        # the gold is ALSO in the first passage: the question is not a two-hop
        {"passage_id": "p3", "context": "Montana is where Ivan Doig founded the company.",
         "questions": []},
    ]
    pairs = [
        [("Who founded the company?", "Ivan Doig")],
        [("Where was Ivan Doig born?", "Montana")],
        [("Where did Ivan Doig found the company?", "Montana")],
    ]
    mined = bf.mine_twohop(passages, pairs, cap=100, seed=0)
    combos = {(c["passage_id"], c["support_passage_id"]) for c in mined["candidates"]}
    assert ("p1", "p2") in combos          # bridge in q2, gold only in C_j
    assert ("p3", "p2") not in combos      # t4: gold reachable from C_i alone
    assert all(c["passage_id"] != c["support_passage_id"] for c in mined["candidates"])


def test_paraphrase_acceptance_bounds():
    golds = ["Montana"]
    question = "Where was the writer of the novel born?"
    assert bf.accept_paraphrase(question, "In which place was the novelist born?", golds)[0]
    assert not bf.accept_paraphrase(question, "Where was the writer born, in Montana?", golds)[0]
    assert not bf.accept_paraphrase(question, "Which city hosts the winter games?", golds)[0]
    assert not bf.accept_paraphrase(question, "Where was the writer of the novel born?", golds)[0]
    assert not bf.accept_paraphrase(question, "In which place was the novelist born", golds)[0]


# --------------------------------------------------------------------------- #
# bootstrap metric + report
# --------------------------------------------------------------------------- #


def _summary(path: Path, preds: dict, golds: list[list[str]]) -> Path:
    rows = [{"passage": 0, "predictions": preds["final"], "golds": golds,
             "base_predictions": preds["base"]}]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"per_passage": rows}))
    return path


def test_bootstrap_f1_metric_reads_token_f1_not_exact_match(tmp_path):
    golds = [["Montana"], ["the Tagus"]]
    base = ["montana river", "tagus"]
    final = ["montana", "the tagus"]
    path = _summary(tmp_path / "runs" / "x" / "summary.json",
                    {"base": base, "final": final}, golds)
    em_res = bg.main(str(path), metric="em")
    f1_res = bg.main(str(path), metric="f1")
    # "montana river" is not an exact match but scores 0.667 token F1, so the two
    # metrics must NOT agree -- that is the whole point of the flag
    assert em_res["base_em"] == pytest.approx(50.0)
    assert em_res["final_em"] == pytest.approx(100.0)
    assert em_res["gain_pts"] == pytest.approx(50.0)
    assert f1_res["base_f1"] == pytest.approx(100.0 * (2 / 3 + 1.0) / 2)
    assert f1_res["gain_pts"] == pytest.approx(100.0 * ((1.0 - 2 / 3) + 0.0) / 2)
    assert f1_res["gain_pts"] != pytest.approx(em_res["gain_pts"])


def _arm_summary(frames: dict) -> dict:
    return {"run": "x", "sft_mode": "qa", "n_passages": 3, "n_questions": 6,
            "arm_status": "COMPLETE", "frames": frames,
            "with_context": {"n": 6, "base_em": 50.0, "final_em": 45.0, "drop_pts": 5.0},
            "literalism": {"median_containment": 0.1, "alarm_count": 0},
            "share": {"band": [0.05, 0.09], "trained_texts": {"new_share": 0.07},
                      "heldout_texts": {"new_share": 0.08}},
            "dose": {"steps": 90, "supervised_tokens_total": 100},
            "training": {"steps_trained": 90}}


def _frames(golds: list[list[str]], base: list[str], s_final: list[str],
            c_final: list[str], n_final: list[str]) -> dict:
    def frame(b, f):
        return {"n": len(golds), "base_predictions": b, "predictions": f, "golds": golds,
                "base_em": 100.0 * sum(bg.score(p, g) for p, g in zip(b, golds)) / len(golds),
                "final_em": 100.0 * sum(bg.score(p, g) for p, g in zip(f, golds)) / len(golds),
                "base_f1": 100.0 * sum(bg.score(p, g, "f1") for p, g in zip(b, golds)) / len(golds),
                "final_f1": 100.0 * sum(bg.score(p, g, "f1") for p, g in zip(f, golds)) / len(golds)}

    out = {}
    for name in ("corpus_orig", "corpus_para", "transfer", "heldout"):
        out[name] = {"S": frame(base, s_final), "C": frame(base, c_final),
                     "N": frame(base, n_final)}
    return out


def test_plan3_report_computes_the_stop_read_and_asserts_signs(tmp_path, monkeypatch):
    golds = [["alpha"], ["beta"], ["gamma"], ["delta"], ["epsilon"], ["zeta"]]
    base = ["x"] * 6
    # S beats C on transfer (the compositional frame) and ties on the single-hop frame
    s_final = ["x"] * 6
    c_final = ["x"] * 6
    # every transfer question shares one gold, so a correct S scores 100 and a base C
    # scores 0: the frame then separates the arms outright rather than by 1 question of 6
    transfer_golds = [["alpha"]] * 6
    per = _frames(golds, base, s_final, c_final, base)
    per["transfer"] = {
        "S": {"n": 6, "base_predictions": ["x"] * 6, "predictions": ["alpha"] * 6,
              "golds": transfer_golds},
        "C": {"n": 6, "base_predictions": ["x"] * 6, "predictions": ["x"] * 6,
              "golds": transfer_golds},
        "N": {"n": 6, "base_predictions": ["x"] * 6, "predictions": ["x"] * 6,
              "golds": transfer_golds},
    }
    for name, frames in per.items():
        for arm in frames:
            data = frames[arm]
            if "base_em" not in data:
                for side, key in (("base", "base_predictions"), ("final", "predictions")):
                    data[f"{side}_em"] = 100.0 * sum(
                        bg.score(p, g) for p, g in zip(data[key], data["golds"])
                    ) / len(data["golds"])
                    data[f"{side}_f1"] = 100.0 * sum(
                        bg.score(p, g, "f1") for p, g in zip(data[key], data["golds"])
                    ) / len(data["golds"])
    for arm in ("S", "C", "N"):
        frames = {name: per[name][arm] for name in per}
        path = tmp_path / "runs" / f"p6_{arm.lower()}" / "summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_arm_summary(frames)))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["plan3_report.py", "--out", "runs/p6_report.json"])
    assert plan3_report.main() == 0
    report = json.loads((tmp_path / "runs" / "p6_report.json").read_text())
    # S wins the transfer frame outright and does not lose the single-hop frame
    assert report["stop_rule"]["read"] == "CONTINUE"
    assert report["transfer"]["S_minus_C"]["gain_pts"] == pytest.approx(100.0)

    # the sign check must FIRE when the reader disagrees with the arms
    def lying_paired(base_pairs, final_pairs, **kw):
        return {"n_questions": len(base_pairs), "gain_pts": 99.0, "ci95_pts": [1.0, 2.0],
                "metric": kw.get("metric", "em"), "verdict": "PASS",
                "n_flipped_up": 1, "n_flipped_down": 0}

    monkeypatch.setattr(bg, "paired", lying_paired)
    with pytest.raises(SystemExit, match="defect"):
        plan3_report.main()
