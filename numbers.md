# numbers.md — every quantitative claim in paper.md, with its artefact

Scope: M0-E5 (small frame), the registered full protocol (200 passages / 923 questions, three
arms) and the residual routing test (two paired arms). Rows in sections 1-12 were re-read from the
artefact named in the last column on 2026-10-29; sections 13-14 were re-read on 2026-10-30 after
the corpus arms and the residual arms were copied back from their rented cards; section 15 lists
claims deliberately kept out of paper.md. Paths are relative to this deposit's root, which
carries `runs/` and `code/` as the lab tree carried them (three rows cite artefacts that are not
in the deposit — the replicated paper's text, the HF model blob, and the `*.pt` dream caches; see
the README's "Not included, by design").

Status codes:

- **MEASURED** — the value is read directly from the named artefact.
- **DERIVED** — arithmetic on MEASURED values, with the inputs named in the row.
- **RECORD** — a value that exists only in the project's run log, not in an
  on-disk artefact of this deposit. Marked so that no reader mistakes it for a measurement, and used in
  paper.md only where the row says so explicitly (cost accounting).

---

## 1. Reference frame — the paper being replicated

Source: `~/research/sleep.txt`, Table 3 (lines 588-596), Llama-3.2-1B backbone,
mean no-context SQuAD accuracy (%), single-passage (n = 1) / continued pretraining (n = 200).

| id | quantity | value | artefact | status |
|---|---|---|---|---|
| P1 | base model | 31.9 / 31.9 | sleep.txt:589 | MEASURED |
| P2 | fine-tuned model with no dreaming | 33.4 / 32.0 | sleep.txt:590 | MEASURED |
| P3 | SEAL | 46.7 / 43.2 | sleep.txt:591 | MEASURED |
| P4 | Sleep (Transformer) | 48.1 / 44.3 | sleep.txt:592 | MEASURED |
| P5 | removing Dreaming (ablation) | 35.7 / 36.2 | sleep.txt:596 | MEASURED |
| P6 | removing gradient-based selection | 47.1 / 45.2 | sleep.txt:594 | MEASURED |
| P7 | removing random expert | 48.0 / 44.7 | sleep.txt:595 | MEASURED |
| P8 | sleep minus no-dreaming, single passage | 12.4 pts | DERIVED, 48.1 - 35.7 (P4, P5) | DERIVED |
| P9 | CPT regime size | 200 passages, 974 questions | sleep.txt:619-620 | MEASURED |
| P10 | dreams sampled per passage in the CPT regime | 5 | sleep.txt:620 | MEASURED |
| P11 | Sleep minus fine-tuned-with-no-dreaming, corpus column | 12.3 pts | DERIVED, 44.3 - 32.0 (P4, P2) | DERIVED |
| P12 | random-expert ablation minus Sleep, single passage | -0.1 pts (48.0 - 48.1) | DERIVED, P7, P4 | DERIVED |
| P13 | gradient-selection ablation minus Sleep, single passage | -1.0 pts (47.1 - 48.1) | DERIVED, P6, P4 | DERIVED |
| P14 | random-expert mechanism, as the paper describes it | "each router in MoE blocks additionally chooses a random expert and so incorporates random irrelevant knowledge to the dreaming" (sleep.txt:452-453); the expert chosen is a **low-rank** expert of the model's own, grown and trained by the sleep process — "we present a random low-rank expert selection in Mixture of Experts (MoE) ... that aims to mix the knowledge by combining the knowledge of irrelevant experts for output computation" (sleep.txt:169-171) | sleep.txt:452-453 and sleep.txt:169-171 | MEASURED |
| P15 | Sleep minus removing-Dreaming, corpus column | 8.1 pts (44.3 - 36.2) | DERIVED, P4, P5 | DERIVED |

## 2. Substrate and surgery

| id | quantity | value | artefact | status |
|---|---|---|---|---|
| S1 | OLMoE-1B-7B-0125 shape | 16 layers, 64 routed experts, top-8 | `.cache/hf/hub/models--allenai--OLMoE-1B-7B-0125/blobs/ba817ac021c5aedf0b16b8a3982ad7d6d27534b3` (config.json: num_hidden_layers 16, num_experts 64, num_experts_per_tok 8) | MEASURED |
| S2 | new experts per layer, appended after NF4 quantisation | k_new = 4, new_expert_start = 64 | `runs/m1_warmup/metrics.jsonl` (stage=student) | MEASURED |
| S3 | trainable parameters | 11,665,472 | `runs/m1_warmup/metrics.jsonl` (stage=student, trainable) | MEASURED |
| S4 | LoRA on the new experts | r = 16, alpha = 32, dropout 0.0 | `configs/base.yaml` (lora) | MEASURED |
| S5 | uniform null for the new bank's routing share | 4/68 = 0.0588 | DERIVED from S1, S2 | DERIVED |
| S6 | dream sampling config | 8 dreams/passage, top_p 0.95, max 256 new tokens, keep top-4 by gradient + 2 random, 8-gram containment filter 0.5 | `configs/base.yaml` (dream) | MEASURED |
| S7 | router warmup | 25 steps, 2 forced slots annealed to 0, lr 1e-3 | `configs/base.yaml` (warmup) | MEASURED |
| S8 | SFT schedule, every arm | 30 steps, lr 1e-4, batch 4, routers frozen | `configs/base.yaml` (sft) | MEASURED |
| S9 | the dream generator's expert bank | the teacher is the unexpanded checkpoint (64 base experts, no new experts), so the reserve bank does not exist in the generator's forward pass; `expand_experts` is applied to the student only | `src/qlora_sleep/teacher.py` (`load_teacher`); `src/qlora_sleep/model.py` (`expand_experts` called from the student build only) | MEASURED |

## 3. Base readings and the re-basing rule (M0)

| id | quantity | value | artefact | status |
|---|---|---|---|---|
| B1 | base, gate frame (20 passages / 109 questions) | EM 11.93, F1 23.72 | `runs/b0_base_gate20/summary.json` | MEASURED |
| B2 | base, held-out frame (20 passages / 92 questions) | EM 2.17, F1 11.20 | `runs/b0_base50_20/summary.json` | MEASURED |
| B3 | base, wider frame (229 questions) | EM 6.99, F1 16.42 | `runs/m0_base/summary.json` | MEASURED |
| B4 | leak probe, gate frame | no-context 12.5 vs with-context 55.0, delta 42.5 | `runs/b0_base_gate20/summary.json` (leak) | MEASURED |
| B5 | leak probe, held-out frame | no-context 2.5 vs with-context 65.0, delta 62.5 | `runs/b0_base50_20/summary.json` (leak) | MEASURED |
| B6 | re-basing gap against the paper's base | -19.97 pts | `runs/b0_base_gate20/summary.json` (rebase_gap) | MEASURED |
| B7 | gate frame composition | 24 passages / 132 questions in file, first 20 passages = 109 questions | `data/squad_gate24.jsonl` | MEASURED |
| B8 | held-out frame composition | 50 passages / 229 questions in file, first 20 passages = 92 questions | `data/squad_base50.jsonl` | MEASURED |

## 3b. Decision rules and thresholds (pre-registered)

| id | quantity | value | artefact | status |
|---|---|---|---|---|
| R1 | bootstrap PASS rule | gain >= +2.0 pts and CI95 lower bound > 0 | `scripts/bootstrap_gate.py` (docstring and main) | MEASURED |
| R2 | bootstrap FAIL rule | gain <= 0 | `scripts/bootstrap_gate.py` | MEASURED |
| R3 | bootstrap settings | 20000 resamples, seed 0 | `scripts/bootstrap_gate.py` (n_boot=20000, seed=0) | MEASURED |
| R4 | routing-share band for the new bank | [0.05, 0.09] | `configs/base.yaml` (warmup.share_target 0.07), `runs/e2_arms/summary.json` (share_band) | MEASURED |
| R5 | E2 condition b bar (trained minus held-out contrast) | 0.02 | pre-registered before the residual run (project record, not in the deposit) | RECORD |
| R6 | non-destruction bound on the held-out frame | 2.0 pts | `configs/base.yaml` (gates.heldout_max_drop) | MEASURED |
| R7 | literalism alarm / insufficient thresholds | 0.5 / 0.3 | `configs/base.yaml` (literalism_alarm, literalism_insufficient) | MEASURED |
| R8 | leak-probe minimum delta | 10.0 pts | `configs/base.yaml` (gates.leak_min_delta) | MEASURED |
| R9 | re-basing tolerance | 5.0 pts | `configs/base.yaml` (gates.rebase_tolerance) | MEASURED |
| R10 | degeneration margin | 2.0 pts | `configs/base.yaml` (gates.degeneration_margin) | MEASURED |

## 4. Expert surgery and router warmup (M1)

| id | quantity | value | artefact | status |
|---|---|---|---|---|
| W1 | warmup-only reading vs base, Samoa frame (5 questions) | EM 20.0 vs 20.0, gain 0.0 | `runs/m1_warmup/summary.json` | MEASURED |
| W2 | gradient on the new router columns | 16.57 at step 0, 1.97 at step 24 | `runs/m1_warmup/metrics.jsonl` (warmup_step, new_router_grad_norm) | MEASURED |
| W3 | new-bank routing share | 0.3966 at step 0 (2 forced slots), 0.1882 at step 24 (1 forced), 0.0751 after annealing | `runs/m1_warmup/metrics.jsonl` (new_share), `runs/m1_warmup/summary.json` (new_expert_share) | MEASURED |
| W4 | warmup + eval wall clock | 90.3 s | `runs/m1_warmup/summary.json` (timings.total_s 90.28) | MEASURED |

## 5. First pooled gate and its diagnosis (M3)

| id | quantity | value | artefact | status |
|---|---|---|---|---|
| T1 | first pooled gate, own-student base to final | EM 14.68 to 11.01, gain -3.67 pts | `runs/m3_gate20_qa2/summary.json` | MEASURED |
| T2 | same run against the external 4-bit base | gain -0.92 pts | `runs/m3_gate20_qa2/summary.json` (gain_over_external_base) | MEASURED |
| T3 | own-student base vs external 4-bit base, same 109 questions | 14.68 vs 11.93, difference 2.75 pts | DERIVED, T1 and B1 | DERIVED |
| T4 | dream groundedness at M3 | mean answer support 0.533 (median 0.563), 8.35 pairs/passage, range 0.00-1.00, min 1 pair | `runs/m3_gate20_qa2/metrics.jsonl` (cycle rows) | MEASURED |
| T5 | early injection probe (CONFOUNDED: four arms shared one model instance, no weight reset; see review note) | base EM 20.0, post-warmup 40.0, all four SFT arms 0.0 | `runs/probe_injection.json` | MEASURED |
| T6 | clean injection probe: grounded pairs only, fresh student, 30 steps, routers frozen | 13 pairs: EM 20.0 to 20.0 (delta 0.0), F1 26.7 to 20.0; matched ungrounded control (20 pairs): EM 20.0 to 0.0 | `runs/probe_grounded_fresh/summary.json` | MEASURED |
| T7 | freshness check, first SFT step loss | 4.595 (grounded) vs 2.993 (ungrounded) | `runs/probe_grounded_fresh/summary.json` (loss_first) | MEASURED |
| T8 | new-bank share, post-SFT, trained vs never-trained text | context 0.1615 vs 0.2585 (contrast -0.0970); prompts 0.2385 vs 0.2739 (contrast -0.0354) | `runs/probe_heldout/summary.json` (post_sft) | MEASURED |
| T9 | same contrasts after warmup only | context -0.0996, prompts -0.0317 | `runs/probe_heldout/summary.json` (post_warmup) | MEASURED |
| T10 | base-expert control, post-SFT | top-4 base share 0.0896 (trained context) / 0.0731 (trained prompts); hottest single base expert 0.0234 / 0.0201 | `runs/probe_heldout/summary.json` (checkpoints.post_sft.shares: keys `trained_context` 0.089617/0.023387, `trained_prompts` 0.073057/0.020101) | MEASURED |
| T11 | held-out EM across checkpoints | 3.26 (post-surgery) to 6.52 (post-warmup) to 2.17 (post-SFT) | `runs/probe_heldout/summary.json` | MEASURED |
| T12 | three-cycle run, per-cycle readings | post-surgery 20.0, post-warmup 20.0, cycle1 0.0, cycle2 0.0, cycle3 20.0, post-cycle 20.0 | `runs/m3_cycle3b/summary.json` (stage_evals) | MEASURED |
| T13 | memorisation of the training dreams, three-cycle run | mean loss 0.0217 over 6 dreams | `runs/m3_cycle3b/summary.json` (probes.memorisation) | MEASURED |
| T14 | new-bank share on three text types, three-cycle run | 0.2062 context, 0.2835 eval prompts, 0.2672 dreams | `runs/m3_cycle3b/summary.json` (probes.routing) | MEASURED |
| T15 | format probe, same 5 questions in three templates | eval template EM 20.0, QA template 0.0, quiz template 0.0 | `runs/m3_cycle3b/summary.json` (probes.format) | MEASURED |
| T16 | routers trained during fine-tuning in the diagnostic runs (the substrate is not router-untrainable) | the ablation arms `control`, `low-lr` and `qa` carry `train_routers: true` (and `no-router` / `qa-no-router` carry false); the three-cycle run's `SFTConfig` leaves `train_routers` at its default, true, because `scripts/run_cycle.py` does not override it, and so does the held-out-share probe (`.probe_heldout.py`, the source of T8); every arm from §4.3 on runs through `scripts/run_multi.py` (E3-E5, R) or `scripts/run_corpus.py` (p6), both of which read `sft.train_routers` from the config, and the configs set it false — recorded directly in the three `runs/p6_*/summary.json`, inherited from `configs/base.yaml` by the other arms, whose summaries do not carry the field | `runs/ablate/results.json` (`train_routers` per arm); `src/qlora_sleep/sft.py` (`SFTConfig.train_routers: bool = True`); `scripts/run_cycle.py` (SFTConfig construction); `.probe_heldout.py` (SFTConfig construction, no `train_routers`); `configs/base.yaml` (sft.train_routers false, with the routers-on share drift 0.230 -> 0.263 recorded in the comment) and `configs/corpus_p6.yaml`; `scripts/run_multi.py`, `scripts/run_corpus.py` (both read the key); the three p6 summaries | MEASURED (artefact + code) |
| T17 | a frozen teacher regenerates identical dreams unless the per-cycle seed advances | with the teacher frozen and one seed, every cycle samples the same dreams, so N cycles train one dream set N times; the runner now derives cycle `c`'s dream seed as `seed + 1000 * c` (warmup on cycle 0 only) | `scripts/run_cycle.py` (the per-cycle `DreamConfig(seed=args.seed + 1000 * c)` and its comment); the observation itself is in the run record, not on disk | RECORD (code) |

## 6. Dream factuality sweep (E1)

Source unless stated: `runs/e1_factuality/summary.json` (12 passages, 8 dreams per cell)
and `runs/e1_quiz07/summary.json` (one cell, same frame).

| id | quantity | value | artefact | status |
|---|---|---|---|---|
| G1 | quiz prompt, temperature 0.9 | mean answer support 0.6556, 15.08 pairs/passage, verdict INSUFFICIENT | e1_factuality grid | MEASURED |
| G2 | quiz prompt, temperature 0.7 | mean answer support 0.7495, 12.33 pairs/passage, verdict PASS | e1_quiz07 grid | MEASURED |
| G3 | facts prompt at 0.9 / 0.7 | 0.083 and 0.000 pairs/passage, both verdict NO_PAIRS | e1_factuality grid | MEASURED |
| G4 | recap prompt at 0.9 | 0.000 pairs/passage, verdict NO_PAIRS | e1_factuality grid | MEASURED |
| G5 | 8-gram containment median, every cell | 0.0 | e1_factuality and e1_quiz07 grid (containment_median) | MEASURED |
| G6 | prose overlap median (content words of the dream present in the passage), facts@0.9 / recap@0.9 / facts@0.7 / quiz@0.9 | 0.025 / 0.905 / 0.017 / 0.468 | e1_factuality grid (prose_overlap.median), 16 dreams per cell | MEASURED |
| G7 | prose-overlap recompute provenance | 3 cells relabelled FAIL to NO_PAIRS, 0 mismatches against stored support/pairs | `runs/e1_factuality/summary.json` (verdict_regenerated) | MEASURED |

## 7. Warmup-arm selection (E2)

Source: `runs/e2_arms/summary.json`. Four text sets per checkpoint: trained context (1),
trained prompts (5), held-out contexts (20), held-out prompts (92). Band [0.05, 0.09].

| id | quantity | value | artefact | status |
|---|---|---|---|---|
| A1 | W1 post-warmup share, trained prompts vs held-out prompts | 0.204 vs 0.234, contrast -0.031 | e2_arms arms.w1.checkpoints.post_warmup | MEASURED |
| A2 | W1 post-tune, same contrast | 0.084 vs 0.094, contrast -0.011; tune-set share 0.0577; mask -0.75 | e2_arms arms.w1 | MEASURED |
| A3 | W2 post-warmup contrast | -0.026 | e2_arms arms.w2.checkpoints.post_warmup | MEASURED |
| A4 | W2 post-tune contrast | 0.080 vs 0.088, contrast -0.008; tune-set share 0.0668; mask -0.5625 | e2_arms arms.w2 | MEASURED |
| A5 | trained-frame and held-out EM across the three checkpoints (both arms) | trained 20.0, 40.0, 40.0; held-out 3.26, 6.52, 4.35 | e2_arms checkpoints | MEASURED |
| A6 | decision | W1 (band reached, condition c passed, contrast condition b failed) | `runs/e2_arms/summary.json` (decision) | MEASURED |
| A7 | share on the tuning set before any bias, both arms | 0.2205 (W1), 0.2105 (W2) | `runs/e2_arms/summary.json` (share_tune_history, mask 0.0) | MEASURED |

## 8. Integration cycle (E3)

Source: `runs/e3_samoa/summary.json` (1 passage, 5 questions; held-out frame 92 questions).

| id | quantity | value | artefact | status |
|---|---|---|---|---|
| I1 | frame reading | EM 20.0 to 0.0 | e3_samoa pooled | MEASURED |
| I2 | held-out reading | 3.26 to 2.17, drop 1.09 pts | e3_samoa | MEASURED |
| I3 | with-context reading | 80.0 to 80.0, drop 0.0 | e3_samoa per_passage | MEASURED |
| I4 | post-SFT share, trained vs held-out prompts | 0.0924 vs 0.0775, contrast +0.0149; mask -0.75 | e3_samoa per_passage | MEASURED |

## 9. Pooled gate, first session (E4) — H100 SXM5 80GB, 20 passages / 109 questions

Sources: `runs/e4_sleep/summary.json`, `runs/e4_cpt/summary.json`,
`runs/e4_nodream/summary.json`; bootstraps recomputed with
`scripts/bootstrap_gate.py` (20000 resamples, seed 0). Contrasts are reported as
left minus right; the script prints right minus left, so the sign is flipped from its
raw output.

| id | quantity | value | artefact | status |
|---|---|---|---|---|
| F1 | sleep arm (qa objective) | EM 13.76 to 14.68, +0.92 pts; CI95 [-3.67, +5.50], 4 up / 3 down; INSUFFICIENT | e4_sleep + bootstrap own-base | MEASURED |
| F2 | SFT-on-C arm (continued pretraining on the passage) | EM 13.76 to 14.68, +0.92 pts; CI95 [-3.67, +5.50], 4 up / 3 down; INSUFFICIENT | e4_cpt + bootstrap own-base | MEASURED |
| F3 | no-dream ablation (all 8 dreams, no selection, whole-text loss) | EM 13.76 to 19.27, +5.50 pts; CI95 [0.00, +11.01], 8 up / 2 down; INSUFFICIENT | e4_nodream + bootstrap own-base | MEASURED |
| F4 | sleep minus SFT-on-C | 0.00 pts; CI95 [-3.67, +3.67], 2 up / 2 down | bootstrap two-arm | MEASURED |
| F5 | no-dream minus sleep | +4.59 pts; CI95 [+0.92, +9.17], 5 up / 0 down | bootstrap two-arm | MEASURED |
| F6 | no-dream minus SFT-on-C | +4.59 pts; CI95 [0.00, +9.17], 6 up / 1 down | bootstrap two-arm | MEASURED |
| F7 | held-out frame, all three arms | 2.17 to 4.13, drop -1.96 pts over 460 question-reads (5 of 20 students) | the three summaries | MEASURED |
| F8 | SFT steps per passage | sleep 6.20 (range 4-15); SFT-on-C 30.0; no-dream 30.0 | the three `metrics.jsonl` (stage=sft, steps) | MEASURED |
| F9 | held-out new-bank share range, by arm | sleep 0.0783-0.0905; SFT-on-C 0.0776-0.0904; no-dream 0.0785-0.0907 — all three arms reach above the 0.09 ceiling at the top of their ranges | the three summaries (per_passage share_heldout) | MEASURED |
| F10 | in-run dream groundedness | mean answer support 0.744, 11.45 pairs/passage | `runs/e4_sleep/metrics.jsonl` (cycle rows) | MEASURED |
| F11 | router bias after bisection, every student | -0.75 (20 of 20 in each arm) | the three summaries (per_passage share_mask_value) | MEASURED |
| F12 | arm wall clock / teacher phase | sleep 2133.9 s, SFT-on-C 1856.2 s, no-dream 1992.7 s; teacher phase 641.9 s | the three summaries (timings) | MEASURED |
| F13 | comparators' step count as a multiple of the dream arm's | 30.0 / 6.20 = 4.84, about five times | DERIVED from F8 | DERIVED |

## 10. Pooled gate, step-matched session (E5) — A100 SXM4 40GB, same frame

Sources: `runs/e5_sleep_qa/summary.json`, `runs/e5_sleep_recap/summary.json`,
`runs/e5_cpt/summary.json`, `runs/e4_nodream/summary.json` (reused, not rerun), and
`runs/e5_report.json` (all nine contrasts with their CIs).

| id | quantity | value | artefact | status |
|---|---|---|---|---|
| H1 | sleep arm, QA objective, 30 steps | EM 13.76 to 17.43, +3.67 pts; CI95 [-2.75, +10.09], 9 up / 5 down; INSUFFICIENT | e5_sleep_qa + e5_report | MEASURED |
| H2 | sleep arm, recap-prose dreams, whole-text loss, 30 steps | EM 13.76 to 9.17, -4.59 pts; CI95 [-10.09, +0.92], 2 up / 7 down; FAIL | e5_sleep_recap + e5_report | MEASURED |
| H3 | SFT-on-C, 30 steps, rerun for the dose counter | EM 13.76 to 17.43, +3.67 pts; CI95 [-1.83, +9.17], 7 up / 3 down | e5_cpt + e5_report | MEASURED |
| H4 | no-dream ablation | +5.50 pts; CI95 [0.00, +11.01], 8 up / 2 down (the E4 run, reused) | e4_nodream + e5_report | MEASURED |
| H5 | sleep minus SFT-on-C | 0.00 pts; CI95 [-6.42, +6.42], 6 up / 6 down | e5_report bootstraps | MEASURED |
| H6 | sleep minus no-dream | -1.83 pts; CI95 [-7.34, +3.67], 4 up / 6 down | e5_report bootstraps | MEASURED |
| H7 | recap minus SFT-on-C | -8.26 pts; CI95 [-14.68, -2.75], 1 up / 10 down | e5_report bootstraps | MEASURED |
| H8 | recap minus no-dream | -10.09 pts; CI95 [-16.51, -4.59], 1 up / 12 down | e5_report bootstraps | MEASURED |
| H9 | sleep minus recap | +8.26 pts; CI95 [+1.83, +15.60], 12 up / 3 down | e5_report bootstraps | MEASURED |
| H10 | held-out frame | QA 3.26 to 2.61 (drop 0.65); recap 3.26 to 2.39 (drop 0.87); SFT-on-C 3.26 to 4.13 (drop -0.87) | the three summaries | MEASURED |
| H11 | supervised tokens per passage (label tokens the loss scored, summed over the 30 steps) | QA 475.4 (range 210-880); recap 5872.3 (2070-16440); SFT-on-C 4665.0 (2100-7680); no-dream not recorded (pre-instrument) | the three summaries (dose) | MEASURED |
| H12 | SFT steps per passage, all three arms | 30.0, range 30-30 in every arm | the three summaries (dose.steps) | MEASURED |
| H13 | QA dose as a fraction of the SFT-on-C dose | 475.4 / 4665.0 = 0.102, about one tenth | DERIVED from H11 | DERIVED |
| H14 | recap dose as a multiple of the SFT-on-C dose | 5872.3 / 4665.0 = 1.26 | DERIVED from H11 | DERIVED |
| H15 | held-out new-bank share range | QA 0.0734-0.0882 (inside the band); SFT-on-C 0.0772-0.0930; recap 0.0737-0.0963 (ceiling breached by up to 0.0063) | the three summaries | MEASURED |
| H16 | literalism verdicts, recap arm | ALARM on 2 of 20 passages: p13 containment 0.5674, p14 containment 1.0000; other 18 OK | `runs/e5_sleep_recap/metrics.jsonl` (cycle rows) | MEASURED |
| H17 | dreams trained vs dreams selected, recap arm | 13 of 20 passages trained fewer dreams than selected; p13 3 of 6, p14 2 of 6; copy filter dropped 3 and 4 | `runs/e5_sleep_recap/metrics.jsonl` (n_kept, n_examples, n_dropped) | MEASURED |
| H18 | dream length | recap 33.9 words mean per passage (per-passage means 13.0-71.7); quiz dreams 106.0; passage text 120.7 | the three summaries (mean_dream_words) | MEASURED |
| H19 | prose overlap median, recap vs quiz arms | 0.852 vs 0.570 | e5_report arms | MEASURED |
| H20 | in-run groundedness, QA arm vs recap arm | QA mean support 0.744 (median 0.771), 11.45 pairs/passage; recap 0.000 support, 0.000 pairs | the two `metrics.jsonl` (cycle rows) | MEASURED |
| H21 | warmup text kind | recap arm fell back to prose (dream_prose); QA and SFT-on-C arms used QA pairs | e5_report arms (warmup_text_kind) | MEASURED |
| H22 | arm wall clock | sleep_qa 3084.7 s, cpt 3298.0 s, sleep_recap 3464.7 s | the three summaries (timings.total_s) | MEASURED |
| H23 | literalism verdicts, QA and SFT-on-C arms | OK on 20 of 20 passages each | e5_report arms (literalism_verdicts) | MEASURED |
| H24 | dose records for the three E4 arms | absent — those arms predate the supervised-token counter, so their dose is not recorded | `runs/e4_sleep/summary.json`, `runs/e4_cpt/summary.json`, `runs/e4_nodream/summary.json` (no `dose` key) | MEASURED |

## 11. Instrument findings that bound every threshold

| id | quantity | value | artefact | status |
|---|---|---|---|---|
| X1 | same arm, same settings, two sessions (E4 SFT-on-C vs E5 SFT-on-C) | +0.92 vs +3.67 pts, a 2.75-pt swing | `runs/e4_cpt/summary.json`, `runs/e5_cpt/summary.json` | MEASURED |
| X2 | per-question agreement between those two runs | golds identical 109/109; untrained base predictions differ on 18/109; trained predictions differ on 53/109 | the two summaries (per_passage predictions, base_predictions) | MEASURED |
| X3 | passages carrying the whole EM difference | passage 7: 0.0 to 16.7 (6 questions); passage 14: 20.0 to 60.0 (5 questions), i.e. 3 questions | the two summaries | MEASURED |
| X4 | the borderline dream-vs-passage contrast at matched steps, E4 vs E5 | no-dream minus SFT-on-C: +4.59 pts, CI95 [0.00, +9.17], 6 up / 1 down in E4; -1.83 pts, CI95 [-7.34, +2.75], 3 up / 5 down in E5 | bootstrap two-arm on the E4 pair; e5_report (cpt_minus_nodream) | MEASURED |
| X5 | question resolution of the pooled frame | 109 questions, 0.92 pts per question | DERIVED, 100/109 | DERIVED |
| X6 | eval batch size, every arm and frame | 32 | the six summaries (eval_batch) | MEASURED |
| X7 | E1 reading replicated out of sample in the E4 run | support 0.744 (E4 in-run) vs 0.7495 (E1 quiz@0.7) | F10 and G2 | DERIVED |
| X8 | pre-counter estimate of the dose gap, against the measured value | estimated 40-290 times fewer supervised tokens for the answer-masked arm, from per-example word counts (quiz-pair answers ~2.0 words against passages ~115, 30 steps x 4 examples); measured on the means: 9.8 times (H13) | an internal review pass (its derivation inputs are H11 and H13, both on disk above) | RECORD (DERIVED) |

## 12. Cost accounting

Instance windows are recorded in the project's run log, not in an artefact on
disk; the arithmetic below is DERIVED from those windows and the published hourly rates.
There is no invoice in the repository.

| id | quantity | value | artefact | status |
|---|---|---|---|---|
| C1 | session 1: H100 SXM5 80GB, us-south-2, 15:01-17:44 local | 2.72 h at $4.29/h = $11.67 | rental-provider session log (not in the deposit); rate from the provider's listed price | RECORD (DERIVED) |
| C2 | session 2: A100 SXM4 40GB, us-west-2, 18:23-22:35 local | 4.20 h at $1.99/h = $8.36 | rental-provider session log (not in the deposit) | RECORD (DERIVED) |
| C3 | rental spend, sessions 1-2 | about $20.03 over 6.92 h (11.67 + 8.36) | C1 + C2 | RECORD (DERIVED) |
| C4 | local RX 9070 spend for M0-M3 (rootful podman ROCm container) | $0 (owned hardware) | project log (not in the deposit) | RECORD |
| C5 | pre-run projections | about $5 (A100-80GB) to $8 (H100) for 3 GPU-h (plan-stage estimate); cap about $6.4 for the small-frame stage (project record, not in the deposit) | RECORD |
| C6 | budget for the full protocol | $60-90 (pre-run allowance set before the corpus arms ran) | project record (not in the deposit) | RECORD |
| C7 | mismatch against the pre-run estimate | the pre-run allowance was about $35 of rental; the measured total for the first two sessions is about $20.03. Reported as a mismatch, not reconciled: the allowance was a pre-run figure, not an invoice | project record (not in the deposit); arithmetic on C1 + C2 | RECORD (DERIVED) |
| C8 | session 3: A100 SXM4 40GB, us-west-2, full protocol (launched 23:52 UTC (session continued after an agent restart), terminated 15:47 UTC) | 15.92 h at $1.99/h = $31.68; plan3 projection 13.9 h / about $28 | run record; arm walls in Q18 | RECORD (DERIVED) |
| C9 | session 4: A100 SXM4 40GB, us-east-1, residual test (16:00-18:13 UTC) | 2.20 h at $1.99/h = $4.38; 2.01 GPU-h of measured run time (V9) | run record; V9 | RECORD (DERIVED) |
| C10 | total rental, all four sessions | 25.04 h, about $56.09 (C1 + C2 + C8 + C9) | C1, C2, C8, C9 | RECORD (DERIVED) |

## 13. Corpus-scale protocol (full protocol) — A100 SXM4 40GB, 200 passages / 923 questions

Sources: `runs/p6_s/summary.json`, `runs/p6_c/summary.json`, `runs/p6_n/summary.json`,
`runs/p6_true_base/summary.json`, `runs/p6_report.json`, `runs/p6_dreams.pt`,
`data/squad_corpus200.jsonl`, `data/corpus_para.jsonl`, `data/transfer_twohop.jsonl`,
`runs/p6_frames/{build,mining}.json`, `boxlogs/p6_*.log`. Arms: S = dream (answer-masked QA pairs
from gradient-selected dreams), C = SFT-on-C (whole passage text), N = no-selection (all 8 dreams,
whole-text loss). All three run 6000 steps serially on one card, seed 0.

| id | quantity | value | artefact | status |
|---|---|---|---|---|
| Q1 | corpus frame composition | 200 passages / 923 questions; 74 held-out contexts excluded; 18,817 available after exclusion | `runs/p6_s/summary.json` (data_meta) and `data/squad_corpus200.jsonl` (header + 200 rows) | MEASURED |
| Q2 | true no-surgery base, corpus frame | EM 11.48, F1 22.11 over 923 questions; leak probe no-context 5.00 vs with-context 55.00, delta 50.0; re-basing gap -20.42 against the paper's 31.9 | `runs/p6_true_base/summary.json` (no_context_em, no_context_f1, leak, rebase_gap) | MEASURED |
| Q3 | surgery + warmup cost | 0.54 pts (11.48 true base minus 10.94 own post-surgery base) | DERIVED, Q2 and Q4 | DERIVED |
| Q4 | pre-training reads, identical in all three arms | corpus_orig EM 10.9426 / F1 21.8755; corpus_para 8.8795 / 19.5845; transfer 0.8547 / 8.3543; heldout 4.3478 / 13.8552 | the three summaries (`frames`, `n_questions`); per-arm `base_em` equal on every frame | MEASURED |
| Q5 | arm S final frames | corpus_orig EM 0.33 (F1 10.94); corpus_para 0.42 (9.74); transfer 0.00 (5.99); heldout 0.00 (6.47) | `runs/p6_s/summary.json` (frames) | MEASURED |
| Q6 | arm C final frames | corpus_orig EM 4.66 (F1 11.45); corpus_para 2.54 (8.74); transfer 0.00 (2.82); heldout 1.09 (5.28) | `runs/p6_c/summary.json` (frames) | MEASURED |
| Q7 | arm N final frames | corpus_orig EM 11.48 (F1 21.74); corpus_para 11.84 (19.67); transfer 1.71 (6.94); heldout 2.17 (8.57) | `runs/p6_n/summary.json` (frames) | MEASURED |
| Q8 | S minus C, primary single-hop frame (corpus_orig, EM) | -4.33 pts; CI95 [-5.74, -2.93]; 3 questions up, 43 down; F1 -0.51 [-2.10, +1.06] | `runs/p6_report.json` (single_hop.S_minus_C, .S_minus_C_f1) | MEASURED |
| Q9 | own-base gains, corpus_orig (EM) | S -10.62 [-12.68, -8.67] (0 up / 98 down); C -6.28 [-8.23, -4.44] (12 up / 70 down); N +0.54 [-1.41, +2.49] (46 up / 41 down) | `runs/p6_report.json` (single_hop) | MEASURED |
| Q10 | S minus N, corpus_orig (EM) | -11.16 pts; CI95 [-13.22, -9.21]; 1 up, 104 down | `runs/p6_report.json` (single_hop.S_minus_N) | MEASURED |
| Q11 | held-out and with-context drops, by arm | held-out (92 q, base 4.35): S to 0.00 (drop 4.35), C to 1.09 (3.26), N to 2.17 (2.17); with-context (150 q, base 56.67): S 24.67 (drop 32.0), C 36.67 (20.0), N 50.00 (6.67). Guards: 3.0 held-out, 10.0 with-context | `runs/p6_report.json` (guards) and the three summaries (`with_context`, frames.heldout) | MEASURED |
| Q12 | supervised-token dose, by arm | S 94,187 total (470.9 mean/passage) over 1924 examples, 12.47 visits/example; C 3,474,120 (17,370.6) over 200 examples, 120.0 visits; N 4,482,846 (22,414.2) over 1586 examples, 15.13 visits | the three summaries (`dose`) | MEASURED |
| Q13 | S dose as a fraction of C dose | 94,187 / 3,474,120 = 0.0271, i.e. C supervises 36.9 times more | DERIVED from Q12 | DERIVED |
| Q14 | training loss, by arm (first / last / minimum over the 6000 steps) | S 4.1401 / 0.0259 / 0.0004; C 0.9312 / 0.0246 / 0.0077; N 1.1513 / 0.2428 / 0.1526 | the three `metrics.jsonl` (stage=sft_step, `loss`); last values also in the summaries (`training.last_loss`) | MEASURED |
| Q15 | mid-training probes (200-question no-context subsample, base 11.00) | S: 0.00 at steps 1000-5000 (delta -11.00, streak reaching 5), 1.50 at 6000 (delta -9.50); C: 7.00 / 4.00 / 3.50 / 5.50 / 3.50 / 5.00; N: 11.00 / 13.50 / 12.50 / 12.50 / 13.50 / 14.00 | the three `metrics.jsonl` (stage=sft_probe) | MEASURED |
| Q16 | literalism at corpus scale | median containment 0.000, 0 alarms, 200 OK in every arm | the three summaries (`literalism`) | MEASURED |
| Q17 | routing share at evaluation, trained / held-out | S 0.1098 / 0.0953; C 0.0666 / 0.0647; N 0.1142 / 0.1096; mask -0.75 in all three; tune-set share 0.0651 | the three summaries (`share`) | MEASURED |
| Q18 | arm wall clocks | S 14478.2 s (4.02 h); C 12751.0 s (3.54 h); N 12994.4 s (3.61 h) | the three summaries (`timings.total_s`) | MEASURED |
| Q19 | dream pass | 200 passages x 8 = 1600 dreams, quiz prompt at temperature 0.7, 10260.0 s (2.85 h) | `boxlogs/p6_dreams.log` (the JSON summary line); `runs/p6_dreams.pt` | MEASURED |
| Q20 | paraphrase frame yield | 473 of 923 accepted (51.2 per cent), below the pre-registered 800, so the primary single-hop frame stays corpus_orig | `runs/p6_report.json` (n_para_accepted, primary_single_hop_frame); `data/corpus_para.jsonl` (474 lines = header + 473) | MEASURED |
| Q21 | transfer frame | 117 rows; 84 compositional; 33 hop2-answerable. Pooled EM 0.00 for S and C, and 1.71 for N (2 of 117 rows, both in the 33 non-compositional rows); EM 0.00 for all three arms on the 84-row compositional stratum. S minus C on token F1: +3.17 [+0.97, +5.47] pooled (27 up / 14 down), +3.39 [+0.73, +6.17] compositional (21 up / 11 down) | `runs/p6_report.json` (transfer, transfer_strata); `runs/p6_{s,c,n}/summary.json` (frames.transfer.final_em 0.0 / 0.0 / 1.7094); `data/transfer_twohop.jsonl` (118 lines = header + 117) | MEASURED |
| Q22 | pre-registered stop-rule read | STOP-NEGATIVE, reason: CI95 excludes 0 against S on corpus_orig (-4.33 [-5.74, -2.93]); the transfer frame's nominal interval for S is recorded in the same block and is not used | `runs/p6_report.json` (stop_rule) | MEASURED |
| Q23 | dose-efficiency read on the paraphrase frame | S gain minus C gain = -2.11 pts; CI95 [-3.81, -0.63]; verdict UNPROVEN by the registered rule (the interval excludes 0 against S, so the claim cannot be made in S's favour) | `runs/p6_report.json` (dose_efficiency) | MEASURED |
| Q24 | paper-kind secondary read (S minus N against the 8.1-pt armed bar) | -11.16 pts [-13.22, -9.21]; gap_reached false; direction reversed | `runs/p6_report.json` (secondary) | MEASURED |
| Q25 | spot-read verdicts on the 20 accepted transfer rows | first build 16 of 20 degenerate; after the bridge predicate was tightened to proper nouns, about 9 of 20 degenerate; the frame was frozen and reported pooled and stratified, with no admission criterion tuned after a reading | `runs/p6_frames/spot_read.txt`, `runs/p6_frames_v1/spot_read.txt`; verdicts in the run record | RECORD |
| Q26 | transfer-frame mining yield | 360 candidates from 1580 supported dream pairs; 1065 candidates rejected as non-entity bridges | `runs/p6_frames/mining.json` (n_supported_pairs, n_candidates, n_bridge_not_entity) | MEASURED |
| Q27 | abort rule and its floor anchoring | rule: streak >= 3 probes more than 10 pts below own base AND latest more than 15 below. With a probe base of 11.00, a delta of -15 is an exact match of -4.0, so the tail clause is unreachable and the rule cannot fire | `scripts/run_corpus.py` (the abort expression); probe base from Q15 | MEASURED (code) / DERIVED (arithmetic) |

## 14. Residual routing test — A100 SXM4 40GB, 20 passages / 109 questions

Sources: `runs/r_std/summary.json`, `runs/r_pert/summary.json`, `runs/residual_report.json`,
`runs/e4_dreams.pt`, `runs/r_pert_dreams.pt`, `boxlogs/{r_std,r_pert}.log`. Both arms: quiz-at-0.7
dreams, 30 steps per passage, W1 warmup, share band [0.05, 0.09], routers frozen, seed 0, eval
batch 32, held-out read every 4th student.

| id | quantity | value | artefact | status |
|---|---|---|---|---|
| V1 | R-std arm (as-validated cached dream set) | EM 13.7615 to 17.4312, gain +3.6697 pts; CI95 [-2.75, +10.09]; 9 up / 5 down; token F1 +3.60 [-2.83, +10.35]; held-out 3.261 to 2.826 (drop 0.43); 9508 supervised tokens | `runs/residual_report.json` (arms.R_std), `runs/r_std/summary.json` | MEASURED |
| V2 | R-pert arm (2 of 8 top-k slots replaced during dream generation) | EM 13.7615 to 13.7615, gain +0.00 pts; CI95 [-6.42, +6.42]; 6 up / 6 down; token F1 +1.59 [-4.59, +7.85]; held-out 3.261 to 3.478 (drop -0.22); 9094 supervised tokens (mean 454.7/passage) | `runs/residual_report.json` (arms.R_pert), `runs/r_pert/summary.json` | MEASURED |
| V3 | paired contrast, R_pert minus R_std on the difference of own-base gains | EM -3.67 pts, CI95 [-10.09, +2.75], INSUFFICIENT (5 favouring R_pert / 9 favouring R_std); F1 -2.01 [-8.79, +4.44], INSUFFICIENT (22 / 19) | `runs/residual_report.json` (contrast) | MEASURED |
| V4 | initial condition across the two arms | per-passage base EM identical on 20 of 20 passages, 0 differing | `runs/residual_report.json` (initial_condition) | MEASURED |
| V5 | perturbation telemetry, R-pert dream pass | a1 (off-standard token fraction) 1.0 over 1,148,416 routed tokens; 2 of 8 top-k slots replaced; mean drawn probability 0.0137 against mean displaced probability 0.0292; 20 of 20 passages carry stats | `runs/residual_report.json` (arms.R_pert.dream_perturb); `boxlogs/r_pert.log` | MEASURED |
| V6 | evaluation routing identity | hook off; zero-strength state bit-identical to the no-state evaluation path (max abs diff 0.0); a non-zero strength moves the logits (max abs diff 9.06 on R-std) | `runs/residual_report.json` (arms.R_std.eval_routing_identity); `boxlogs/r_pert.log` (the r_pert summary field is null, see V10) | MEASURED |
| V7 | dream-level comparison of the two dream sets | answer support 0.7436 to 0.6731; extracted pairs per passage 11.45 to 13.50; kept 6.00 in both arms; mean dream length 106.02 to 114.85 words; containment median 0.000 and 0 alarms in both; 0 of 160 dreams verbatim identical at matched index | `runs/r_std/metrics.jsonl` and `runs/r_pert/metrics.jsonl` (stage=cycle; r_std carries the same cached dream set as `runs/e4_sleep`, whose cycle rows predate the `mean_words` field); verbatim count recomputed from `runs/e4_dreams.pt` against `runs/r_pert_dreams.pt` | MEASURED |
| V8 | R-std reproduces the E5 dream arm | base 13.761468, final 17.431193, gain 3.669725 in both `runs/r_std/summary.json` and `runs/e5_sleep_qa/summary.json` (identical to 4 decimal places), on a different card in a different region | the two summaries | MEASURED |
| V9 | arm wall clocks | R-std 2945.1 s; R-pert 4278.8 s including a 1084 s perturbed dream pass | `runs/residual_report.json` (arms.*.wall_s); `boxlogs/r_pert.log` | MEASURED |
| V10 | named artefact defects that survived the measurement | `runs/r_pert/summary.json` carries `dream_perturb.source = "cache"` although that session generated the dreams (truth: n_passages_with_stats 20, tokens 1,148,416, log line 1084 s); its `eval_routing_identity` is null because the check sits inside the passage loop, which a resumed run skips (measurement in `boxlogs/r_pert.log` and in r_std's summary on the same build) | the artefact itself, against V5/V6 | RECORD |

## 15. Claims deliberately NOT in paper.md

Recorded so that a later reader does not add them back without re-measuring:

- The two one-passage recap probes at temperature 0.7 and 0.9 (containment median 0.92,
  mean dream length 17 words, 4 of 8 dreams near-identical at 0.7). Their directories
  `runs/probe_e5_recap`, `runs/probe_e5_recap09`, `runs/probe_e5_qa` came back from the
  rented card EMPTY, and `runs/e5_sleep_recap_failed_p12` is empty as well, so these
  values exist only in the run record. paper.md makes the same points from the recap
  arm's own telemetry (H16, H17, H18, H19), which is on disk, and refers to the early probe
  only qualitatively ("near-copies"), with no value from it.
- The eval-batch sensitivity reading (one answer of 109 flipping between batch settings).
  No artefact carries a base reading other than 11.93 on the gate frame (B1), so the
  batch comparison is not reproducible here; paper.md states only that one batch size was
  used throughout (X6).
- The `facts` prompt cells' characterisation as off-topic regurgitation: the overlap
  medians are on disk (G6), the qualitative reading rests on 16 retained samples.
- Any number from the first M3 pooled run `runs/m3_gate20_qa`, which was killed before
  producing a reading (the directory holds telemetry only).
- The dream-level **content-word Jaccard** between the two residual dream sets. The run record
  carries 0.199 (mean, matched index); recomputing it from `runs/e4_dreams.pt` against
  `runs/r_pert_dreams.pt` with the paper's word tokeniser over all 8 dreams per passage gives
  0.2617. The two conventions are not the same measurement and neither is on disk as a stored
  field, so paper.md makes the "materially different" point from V5 (every routed token
  off-standard) and V7 (0 of 160 verbatim identical), both of which reproduce exactly.
- The dream-level groundedness and length readings computed over **all eight** dreams per passage,
  as distinct from the in-run cycle telemetry. Recomputed offline: answer support 0.7234 to
  0.6785, pairs 12.30 to 15.25, mean length 119.76 to 125.19 words. The runner's cycle rows score
  the **kept** dreams (6 per passage) and read 0.7436 to 0.6731, 11.45 to 13.50 and 106.02 to
  114.85 — those are V7 and those are what paper.md cites. The run record's "120.7 to 125.9"
  length pair is the all-eight convention, not V7's.
- The per-passage dream-length range for the recap arm (13.0-71.7 words) as a dream-level
  distribution: H18's means are on disk, the range is in the run record.
