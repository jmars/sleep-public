# Does the sleep paradigm survive a different substrate? — the deposit

Artifact for the paper `paper.md`:

> *Does the sleep paradigm survive a different substrate? A mechanism assay of dream-based
> consolidation on a frozen 4-bit mixture of experts*

**Published: concept doi [10.5281/zenodo.23820950](https://doi.org/10.5281/zenodo.23820950)** —
it always resolves to the latest archived version. Cite the paper by the concept DOI, which always resolves to
the latest archived version.

This repository is the paper's **evidence deposit**: the paper, its claim-to-artefact table, the
code the runs executed, and the run data from which every headline number is re-derivable. The
paper is a **mechanism assay**, not a capability claim — it re-hosts the offline sleep stage of
Behrouz et al. (arXiv:2606.03979v2, "Language Models Need Sleep") on a substrate the paper never
uses: a frozen 4-bit NF4 mixture of experts (OLMoE-1B-7B-0125) with a reserve of cloned experts
per layer and QLoRA adapters as the only trainable capacity, rather than the paper's Hope/CMS
architecture with low-rank expert expansion. The assay's answer is a **substrate-expressiveness
thesis**: the paradigm's plumbing — a reserve held inert until unlocked, seeded so the router can
find it, fed synthetic text — is expressible here and was expressed; its coupled dynamic — dreams
written from a route perturbed onto that reserve while the reserve is being trained on the dreams
that perturbation produced — is not, and no arm here supplied it. Four measurements carry the
thesis: a reserve bank that can be capped but never taught to prefer storage-worthy text; a dream
arm statistically indistinguishable from continued pretraining on the same passages on the
20-passage frame; a corpus-scale run (200 passages, 923 questions) in which the dream arm is the
worst of three, 0.33 exact match against 4.66 for fine-tuning on the passages and 11.48 for a
no-selection dream-prose ablation from a common own base of 10.94, which the pre-registered stop
rule reads **STOP-NEGATIVE**; and a router-perturbation arm that changed the dreams materially and
the student not at all (contrast -3.67 points, interval [-10.09, +2.75], INSUFFICIENT). What
transfers is the instrument set, not the verdict.

## What is here

| path | what it is |
|---|---|
| `paper.md` | the paper (markdown — the source of truth) |
| `paper.pdf` | the paper rendered to PDF, generated from `paper.md`: `pandoc paper.md --pdf-engine=typst -o paper.pdf` |
| `numbers.md` | every quantitative claim in `paper.md`, one row each, with the file and key it was read from and a MEASURED / DERIVED / RECORD status — the claim-to-artefact table |
| `code/` | the code the runs executed, verbatim: the package, the runners, the configs, the tests, the prepared frames, and the packaging metadata |
| `code/src/qlora_sleep/` | the package: `model.py` (expert-bank surgery, router extension, share-cap mask, routing telemetry), `teacher.py` (bf16 dream generator and the generation-time router perturbation), `dream.py` (sampling, gradient scoring, selection, 8-gram containment and answer-support instruments), `router_warmup.py`, `sft.py` (QLoRA SFT on context-stripped dreams; the QA-pair and whole-text objectives), `pruning.py`, `eval_squad.py` (closed-book no-context reading, exact match and token F1), `telemetry.py` (JSONL), `canaries.py`, `config.py` |
| `code/scripts/` | the runners and the CPU-only report statistics: `run_cycle.py`, `run_multi.py`, `run_corpus.py` (the corpus protocol), `eval_baselines.py`, `build_frames.py`, `build_dreams.py`, `warmup_arms.py`, `sweep_factuality.py`, `sweep_literalism.py`, `bootstrap_gate.py`, `e5_report.py`, `plan3_report.py`, `residual_report.py`, `dose_report.py`, `regen_factuality_verdicts.py`, `smoke_gpu.py`, `box_bringup.sh` (the rented-box bring-up) |
| `code/configs/` | `base.yaml` (substrate, surgery, dream, warmup and SFT defaults, with the gate thresholds), `single_n1.yaml`, `cpt_n200.yaml`, `corpus_p6.yaml` |
| `code/tests/` | the 13 pytest modules — gradient flow through the new experts, routing share, context stripping, literalism, importance-versus-utilisation pruning, no-context evaluation, QA-pair SFT, the share cap, the router perturbation, and the corpus-frame builders |
| `code/data/` | `squad_prep.py` and the prepared frames: `squad_base50.jsonl`, `squad_gate24.jsonl` (20 passages / 109 questions), `squad_corpus200.jsonl` (200 passages / 923 questions), `squad_n1.jsonl`, `corpus_para.jsonl` (the paraphrase frame), `transfer_twohop.jsonl` (the two-hop transfer frame) |
| `code/pyproject.toml` | the package metadata the box installed from (the dependency pins matter: `transformers<5` keeps the per-expert `ModuleList` the surgery needs) |
| `runs/` | the cited run evidence — for each cited run set: `metrics.jsonl` (per-step and per-stage telemetry), `summary.json` (pooled readings, per-frame readings, dose, routing share, literalism, timings), `rows.jsonl` where the runner wrote one, the small `.log` files, and the derived report JSONs (`e5_report.json`, `p6_report.json`, `residual_report.json`) |
| `boxlogs/` | the execution box's stdout logs for the corpus and residual sessions — the record for the two artefact defects the paper names rather than repairs (§4.9, Appendix A) |
| `README.md`, `LICENSE`, `LICENSE-paper` | this file and the licences |
| `SNAPSHOT` | the revision of the working tree this deposit was taken from — the pin |

Paths inside `paper.md` resolve as follows: `runs/` and `boxlogs/` are at this deposit's root,
unchanged; `src/`, `scripts/`, `configs/`, `tests/` and `data/` — which the paper's Appendix A
cites from the tree root — live under `code/`. `numbers.md` states the same mapping.

## Where the headline numbers live

Every number in the paper carries an id that resolves to a row in `numbers.md`, which names the
file and key. The load-bearing ones:

| quantity | where in `runs/` |
|---|---|
| corpus-scale arms, 0.33 / 4.66 / 11.48 EM from own base 10.94 | `p6_s/summary.json`, `p6_c/summary.json`, `p6_n/summary.json` → `frames.corpus_orig.base_em` / `final_em` |
| the true no-surgery base, 11.48 EM / 22.11 F1 over 923 questions, and the -20.42 re-basing gap against the paper's 31.9 | `p6_true_base/summary.json` → `no_context_em`, `no_context_f1`, `rebase_gap` |
| the pre-registered **STOP-NEGATIVE** read | `p6_report.json` → `stop_rule`; the arm contrasts in `single_hop` |
| residual arms +3.67 and +0.00, and the paired contrast -3.67 [-10.09, +2.75] | `residual_report.json` → `arms.*.corpus_em`, `contrast.em` |
| the perturbation was material: 1,148,416 routed tokens, a1 = 1.0, 2 of 8 top-k slots replaced | `residual_report.json` → `arms.R_pert.dream_perturb`; `boxlogs/r_pert.log` |
| the two dream sets moved in groundedness: answer support 0.7436 to 0.6731, 0 of 160 dreams verbatim identical | `r_std/metrics.jsonl`, `r_pert/metrics.jsonl` → `stage=cycle` |
| the small-frame tie with continued pretraining, 0.00 [-6.42, +6.42] | `e5_report.json` → `bootstraps.sleep_qa_minus_cpt`; the arms in `e5_sleep_qa/`, `e5_cpt/`, `e5_sleep_recap/` |
| the first-session arms (+0.92 dream, +0.92 CPT, +5.50 no-dream) and the cross-session variance | `e4_sleep/`, `e4_cpt/`, `e4_nodream/` → `pooled_gain` |
| the gate-frame base, 11.93 EM / 23.72 F1 over 109 questions | `b0_base_gate20/summary.json` |
| trainable parameter count, 11,665,472 | `m1_warmup/metrics.jsonl` → `stage=student`, `trainable` |
| the literalism instrument and its sweep | `m2_literalism/metrics.jsonl`; `e1_factuality/summary.json`, `e1_quiz07/summary.json` |
| supervised-token dose per arm, and the routing share at evaluation | the three `p6_*/summary.json` → `dose`, `share` |
| the dream pass, 200 x 8 dreams at quiz/0.7 in 10,260 s | `boxlogs/p6_dreams.log` (the summary line) |
| the paper's own reference numbers (31.9, 33.4, 46.7, 48.1, 35.7, 44.3, 32.0) | not in this deposit — they are read from the replicated paper's text; `numbers.md` §1 names the line of `sleep.txt` each one comes from |

## How to run it

The statistics are CPU-only and need nothing but Python: each report script recomputes its numbers
from the per-question predictions and asserts that every contrast's sign agrees with the two arms'
own pooled values.

```sh
cd code
python3 scripts/bootstrap_gate.py ../runs/e5_sleep_qa/summary.json                       # own-base paired
python3 scripts/bootstrap_gate.py ../runs/e5_sleep_qa/summary.json ../runs/e5_cpt/summary.json
python3 scripts/e5_report.py --out /tmp/e5_report.json                                  # nine contrasts
python3 scripts/plan3_report.py --out /tmp/p6_report.json                               # corpus arms, stop rule
python3 scripts/residual_report.py --out /tmp/residual_report.json                      # paired perturbation arms
python3 scripts/regen_factuality_verdicts.py                                            # E1 verdict relabel
```

Re-running an arm needs a GPU: `code/pyproject.toml` pins the environment (torch, transformers < 5,
peft, bitsandbytes, datasets), `code/configs/` holds the four configurations the runs used, and
`code/scripts/box_bringup.sh` is the bring-up script the sessions ran (its output is in
`boxlogs/bringup.log`). The local
development path for the early milestones was a rootful podman ROCm container on an RX 9070; the
paper's runs from §4.3 on are CUDA, on rented A100/H100 instances. `code/tests/` runs on CPU
against tiny random mixtures of experts and is the fastest way to see the instruments behave.

## Not included, by design

- **The dream caches (`*.pt`)** — `e4_dreams.pt`, `p6_dreams.pt`, `p6_smoke_dreams.pt`,
  `e5_recap_dreams.pt`, `r_pert_dreams.pt`, `probe_recap.pt`, `probe_recap09.pt`. They are
  regenerate-able from the shipped generator and the shipped configs
  (`scripts/build_dreams.py` under `configs/corpus_p6.yaml`), and the paper's dream-level claims
  are carried by the in-run telemetry that is here (`metrics.jsonl`, `stage=cycle`), not by the
  tensors. Two `numbers.md` rows cite them as provenance for the dream pass; those rows are
  marked, and no value in `paper.md` is read from a `.pt` file.
- **No file here exceeds 5 MB.** The largest is `runs/p6_s/metrics.jsonl` at 3.3 MB; the rule that
  produced the cut is size, and the dream caches above are the only excluded payload.
- **The private lab tree** — the working tree the deposit was taken from
  (`~/research/qlora-sleep`), including the root-level probe and diagnostic scripts that predate
  the package (`.probe_*.py`, `.diag_*.py`, `.ablate_sft.py`, `.dev_olmoe_src.py`), the container
  and Hugging Face caches (`.cache/`), `__pycache__`, `.pytest_cache`, and the box-side scratch
  files. The package, the runners, the configs, the tests and the prepared frames are all here;
  the scratch probes are not, and `SNAPSHOT` records the omission.
- **API tokens and host addresses** — no credential of any kind is in this deposit. The rented
  boxes were driven with a token read at run time from outside the tree; the logs that are here
  carry no token and no host address, and the deposit was grepped for both before release. The one
  IP-shaped string in the run data is a model prediction in `runs/p6_s/summary.json`
  (`frames.corpus_orig.predictions[798]`, the memorised SQuAD passage title
  `1.84.1.67.2: The Protector of Small Animals`), not an address and not in any host field.
- **Four empty run directories** — `runs/probe_e5_qa`, `runs/probe_e5_recap`,
  `runs/probe_e5_recap09` and `runs/e5_sleep_recap_failed_p12` came back from their rented card
  empty on disk, so they carry no files here. They are named in `numbers.md` §15 as the reason
  the recap-probe values exist only in the run record.

## This deposit is DERIVED — do not hand-edit it

Every file here is materialised from a private working tree. Edit a file here and the next sync
overwrites it: the working tree is the source. But this repository is a **snapshot taken at a
stated revision** of that tree, not a live mirror of it — and this tree has no version control at
all, so the pin is a time and a source path rather than a sha. `SNAPSHOT` records both, and names
the one file whose bytes differ from the source (this deposit's `numbers.md` scope line).

## Licence

MIT — see `LICENSE`, covering the code under `code/` and the run data under `runs/` and
`boxlogs/`. The paper (`paper.md`, `paper.pdf`) and the claim-to-artefact table (`numbers.md`) are
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) — see `LICENSE-paper`, the same split the
earlier deposits use. The copyright holder is Jaye Timothy Marshall; the earlier papers' author
line carries the same name and ORCID iD
([0009-0002-5209-8161](https://orcid.org/0009-0002-5209-8161)).
