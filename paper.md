# Does the sleep paradigm survive a different substrate?

### A mechanism assay of dream-based consolidation on a frozen 4-bit mixture of experts

**Draft — complete for the small-frame results (M0-E5), the registered full protocol and the
residual routing test. The central interpretive claim is a substrate-expressiveness thesis
(§1, §5): the paradigm's plumbing transfers to this substrate, its coupled dreaming dynamic does
not.**

---

## Abstract

Behrouz et al. (arXiv:2606.03979v2, "Language Models Need Sleep") report that an offline
sleep stage — dreaming over a wake context, scoring the dreams by gradient, then
supervised fine-tuning on the selected dreams with the context stripped — incorporates
SQuAD passages into a Llama-3.2-1B backbone far better than its own removing-dreaming
ablation (48.1 vs 35.7 mean no-context accuracy, single passage) and, at its corpus scale,
better than fine-tuning on the passages themselves (44.3 vs 32.0) [P4, P5, P2].

Our question is narrower, and our answer is a thesis about substrates. The paradigm has two
parts that are not equally portable. Its **plumbing** — a reserve of capacity held inert
until it is unlocked, seeded so the router can find it, and fed synthetic text — is
expressible on a frozen-router QLoRA mixture of experts, and we express all of it (§2). Its
**dynamic** — a route perturbed onto that reserve capacity while the dreams are being
written, with the routing and the capacity trained on the dreams that perturbation produced,
so that perturbation and storage are coupled — is not expressible here. Routing itself is
trainable on this substrate and we trained it; what no arm here supplied is the coupling,
because the perturbation runs at dream-generation time in a generator that carries no reserve
capacity, and every arm from §4.3 on trains the capacity on standard-routed dreams with the
routers frozen (§2.2).
Each outcome below is consistent with what that substrate would predict, and we report the
four in the order we measured them.

**Consolidation is bounded.** One additive bias holds the reserve bank near its null routing
share, but it never comes to prefer storage-worthy text: the trained-versus-held-out share
contrast sits at or below zero in three of the four contrasts we measured, and the fourth is
+0.0149 on a single-passage probe (§4.2, §4.3) — not the signature a coupled substrate would
show.

**Injection is absent.** On the 20-passage frame the dream arm gains 3.67 points over its own
untrained reading and is statistically indistinguishable from continued pretraining on the
same passages (0.00 points, interval [-6.42, +6.42]) [H5].

**Corpus-scale destruction.** At the paper's corpus regime — 200 passages and 923 questions, at a
6000-step dose we set (30 steps times 200 passages) on one adapter set — the dream arm is the
worst of three, not the best: 0.33 exact match against 4.66 for fine-tuning on the passages
themselves and 11.48 for a no-selection dream-prose ablation, from a common own base of 10.94
[Q5-Q8]. The paper's corpus-column ordering — its sleep method leading its own no-dreaming
fine-tune by 12.3 points [P11] — did not reproduce on our substrate; it inverted, and both
objectives that touch the facts directly destroy the closed-book reading, with the passage in
context as well (§4.8).

**Perturbation alone did not inject.** Dreams written with two of eight top-k routing slots
replaced by non-top-k draws — a material intervention, 1,148,416 routed tokens with every token's
expert set off-standard [V5] — did not inject where standard-routing dreams did: gain +0.00
against +3.67, contrast -3.67 points with an interval of [-10.09, +2.75], which the pre-registered
rule reads INSUFFICIENT rather than positive [V3]. The null is a joint one: the perturbed dreams
were also less grounded (answer support 0.744 to 0.673) [V7], so routing and factuality moved
together and neither can be credited with the null alone (§4.9). The standard-routing arm
reproduced the earlier session's arm to four decimal places on a different card in a different
region [V8] — this project's reproducibility anchor.

Two nuances keep the thesis honest. Our own ablation cannot attribute the corpus-scale
failure to the missing dynamic alone: the paper's own removing-random-expert row (48.0) sits
0.1 point below its full method (48.1) [P7, P4], so random routing specifically is not what
carries the paper's gain, and our null on perturbation is consistent with the paper's numbers
rather than evidence against them. And the paper's outer loop (ReSTEM, dream-policy training)
was deferred here, so our generator was never trained: a mechanism we did not test, not one
we tested and refuted.

This is a mechanism assay, not a capability claim. The read-out is closed-book recall of facts
from passages read once, scored by exact match; nothing here licenses a statement about what
the model knows or how well it answers questions. Our negatives are about our substrate. The
paper's mechanism on the paper's substrate is not what we measured.

---

## 1. Introduction

The sleep paradigm's mechanism is easy to state and hard to attribute. A wake phase holds a
context `C` in view. A sleep phase samples dreams conditioned on `C`, scores each dream by
the gradient of a supervised loss with respect to the trainable parameters, keeps the top
`k` plus `b` random samples for diversity, and fine-tunes on the kept dreams **with `C`
removed from the input**. Consolidation then moves what was learned into a slower tier and
resets the fast tier's low-rank parameters, freeing capacity for the next cycle.

Each of those steps is a claim about a substrate, and the claims divide into two kinds.

The first kind is **plumbing**: reserve capacity, seeding, synthetic text. Masked pre-allocation
presumes a fixed-capacity bank in which new capacity can sit inert until unlocked; gradient-based
selection presumes that the gradient through the trainable parameters is a usable proxy for
usefulness; and the sleep stage as a whole presumes only that a model can be trained on its own
generated text with the wake context stripped. All of that can be re-hosted on a very different
backbone, and we re-host it.

The second kind is a **dynamic**: the coupling between where the forward pass goes and what the
dreams contain. In the paper the dreams are written while each router additionally selects a random
expert, so that "irrelevant knowledge" is mixed into the dream [P14] — and the experts being mixed
in are the model's own low-rank experts, grown and trained by the sleep process itself. The routing that produces
that perturbation is trained rather than frozen, on the same dream data, together with the capacity
the dreams are supposed to fill. Perturbation and storage are therefore one mechanism: the student
is asked to write down, from an off-standard route through its own trained capacity, what the
standard route saw. That dynamic presumes routers that can be perturbed during generation **and**
trained afterwards, on a bank that is itself part of the trained system.

Our substrate supplies the plumbing and not the coupled dynamic. We re-host the mechanism on
OLMoE-1B-7B-0125 [S1]: 16 layers, 64 routed experts per layer, top-8 token-choice routing,
quantised to NF4 with the base frozen [S4]. Our adapters are four cloned experts per layer, each
wrapped in LoRA rank 16, appended after quantisation so they stay in bf16; the trainable set is
11,665,472 parameters [S3]. This is our reading of the paper's "low-rank expert expansion with
masked pre-allocation", transposed onto an open-weights substrate, and it is a genuinely
different one: no memory hierarchy, no consolidation, no outer loop, and — the difference this
paper is about — routers that are frozen during fine-tuning in every arm from §4.3 on, and were
trained on standard-routed dream text only, never on the dreams a perturbed route writes.

That asymmetry is our central interpretive claim, and it is falsifiable in one direction: if the
plumbing were sufficient, a dream stage built out of it would inject facts, and would inject more
of them than training on the wake passage does. It does neither. The four outcomes we report are
each consistent with what this substrate predicts, and we state them in the order we measured
them.

1. **Consolidation is bounded** (§4.2, §4.3). A frozen top-8 router gives a new expert no
   gradient until it can enter the top 8, so "masked until activation" is a precondition here
   rather than a convenience; once unlocked, the bank can be held near its null share but not
   taught to prefer storage-worthy text.
2. **Injection is absent** (§4.4). On the 20-passage frame the dream stage adds nothing
   detectable over training on the passage itself.
3. **At corpus scale the pipeline destroys the reading** (§4.8). Both objectives that touch the
   facts directly collapse the closed-book read-out.
4. **Perturbation alone did not inject** (§4.9). Supplying the dynamic's most visible
   ingredient — off-standard routing while the dreams are written — without the coupling did
   not inject.

What we do **not** claim: that dreaming cannot work, that a 1B-active mixture of experts lacks
the capacity for fact storage, that random routing is the load-bearing part of the paper's
mechanism (the paper's own ablation says it is worth 0.1 point [P4, P7]), or that our negative
transfers to the paper's substrate. Our negatives are about our substrate. The claim is bounded
by the instrument: closed-book exact-match recall of facts from passages read once, one seed per
frame, and a substrate that omits the memory hierarchy, the consolidation step, the routing
trained on its own perturbed dreams and the outer loop (§2).

---

## 2. The paradigm and our substrate mapping

| Paper mechanism | Our substitute | Status here |
|---|---|---|
| Multi-level memory hierarchy (Hope/CMS; slow and fast frequency blocks) | none — one frozen 4-bit bank of 64 experts per layer, no slow tier | **not implemented** |
| Low-rank expert expansion (`A`, `B` per MLP) | 4 cloned experts per layer, LoRA rank 16 each, appended after NF4 quantisation so they remain bf16 | implemented |
| Masked pre-allocation (parameters present, masked before activation) | new experts held out of routing by an additive mask on their router logits until a 25-step warmup | implemented, and load-bearing (§2.1) |
| Random-expert routing during dreaming | one extra uniformly random expert slot appended to top-k during dream generation (all earlier runs), plus, in the residual test, two of eight top-k slots replaced by non-top-k draws | implemented as a generation-time sampling perturbation, **not** as trained routing coupled to the reserve bank (§2.2) |
| Gradient-based dream selection (top-`k` plus `b` random) | 8 dreams per passage, keep the top 4 by gradient norm of the SFT loss plus 2 random, 8-gram copy filter | implemented |
| SFT on selected dreams with `C` stripped | the student never sees `C`; dreams are the only training text | implemented |
| Consolidation and synaptic reset of the fast tier | none | **not implemented** |
| Outer self-improvement loop (ReSTEM) | none; one selection pass, one cycle per passage | **not implemented** |

The omitted mechanisms are not incidental. The memory hierarchy is where the paper's capacity
recycling happens, and the outer loop is where dream quality is trained rather than merely
selected. A negative result on this substrate is therefore a statement about **selection plus
SFT on frozen-4-bit adapters**, not about the full architecture.

### 2.1 Masked pre-allocation is load-bearing here in a way the paper does not need to discuss

The paper introduces masking as an implementation convenience: keep the parameters present and
mask them in the forward and backward pass until their first activation, so growing the module
never requires reshaping tensors. On a frozen top-8 token-choice router the mask does more
than that. A new column has zero gradient while it is masked, because it never appears in the
top 8; and it can only appear once its logit is competitive, which is a property of the
router, not of the expert's weights. The expert is trained by nothing until the router selects
it, and the router is trained by nothing unless the expert is selected.

We hit this trap twice. In the first version of the warmup, the forced slot was given the mean
of the base top-k weights — a quantity that is not a function of the new expert's router
column at all. The loss curve looked healthy while only base columns moved. In the second
version, forced slots were placed over a masked column, whose weight is approximately zero, so
nothing trained. The fix is that warmup lifts the mask first (so the new columns have real
probabilities) and the forced slot carries the new expert's own softmax probability. After the
fix, the gradient on the new columns falls from 16.57 at the first warmup step to 1.97 at the
last, and the new bank's routing share falls from 0.3966 (two forced slots) to 0.1882 (one
forced slot) to 0.0751 after annealing [W2, W3] — the router selects the new experts on merit
once it is allowed to.

Two consequences for anyone re-hosting this paradigm: the warmup is not a nicety, and the
diagnostic that distinguishes a working warmup from a broken one is the gradient on the new
router columns, not the training loss.

### 2.2 What this substrate can express, and what it cannot

Three properties of the substrate decide which parts of the paradigm are available, and they are
worth stating separately because our results are read against them.

**Reserve capacity is expressible.** Four cloned experts per layer, appended after quantisation
and held inert by a router mask, give the paper's masked pre-allocation a faithful substitute: the
parameters exist, are trainable, and are invisible to routing until unlocked [S2, S3].

**A generation-time routing perturbation is expressible.** The forward hook can replace top-k
slots with non-top-k draws while the teacher writes a dream, and we implemented exactly that
(§3.5): two of eight slots replaced at the substitutes' own probabilities, every routed token
off-standard [V5]. What the hook cannot do is change what those experts contain. In the paper the
expert mixed in is one of the model's own low-rank experts, part of the capacity the sleep process
grows and trains [P14]. Here the generator is the unexpanded checkpoint [S9]: it has 64 base
experts and no reserve bank at all, so the extra or replaced slot can only draw from frozen base
experts that nothing in this project trained for the job the dream needs. The reserve bank the
student is meant to fill does not exist in the generator's forward pass. That is perturbation
**without coupling**, and §4.9 measures what it is worth.

**The coupled dynamic is not expressible here.** Trained routing *is* available on this
substrate, and the paper must not be read as claiming otherwise. The base router columns are
frozen by quantisation, but the new columns are trainable: our warmup trains them (§2.1), and
the diagnostic runs of §4.2 trained them during fine-tuning as well [T16]. What no run here
supplied is the coupling — a route perturbed while the dreams are written, through the same
capacity, with the routing and the adapters trained on the dreams that perturbation produced.
Two measured facts foreclose that compound as configured. First, the generator is the
unexpanded checkpoint [S9]: its forward pass has 64 base experts and no reserve bank, so a
perturbed or replaced slot can only draw on frozen base experts and the perturbed route never
passes through the capacity the student is about to fill. Second, the bank over-attracted in the
diagnostic runs — a fifth to more than a quarter of all routing slots on every text type we
probed, with the trained-minus-held-out contrast wrong-signed (§4.2) [T8, T14] — and co-training
the routing with the capacity drifted it further, so the routers were frozen during fine-tuning
by pre-registered design (§4.3, fix 3). The co-trained half was therefore only ever measured
against standard-routed dreams, and read no better than its own base at the resolution it was
measured at; it was never given a run with perturbation on the generation side. So the
inexpressible thing is narrower than "trained routing", and it is the thing that matters:
perturbation during generation coupled to routing and adapters trained on that perturbed
distribution. No amount of generation-time
perturbation substitutes for it, and the part we could supply in isolation (§4.9) is the part the
paper's own ablation prices at 0.1 point [P4, P7].

One consequence should be stated before the results, because it bounds how far they can be pushed:
the paper's own ablations show that random routing specifically is not where its gain lives.
Removing random-expert routing costs 0.1 point (48.0 against 48.1) and removing gradient-based
selection costs 1.0 point (47.1 against 48.1), while removing dreaming altogether costs 12.4
(35.7) [P4, P5, P6, P7]. The load-bearing part is dreaming as a whole — of which the coupled
dynamic is the part this substrate never supplied — not the random-expert sampling we could
reproduce. Our residual test is therefore a test of the
ingredient we *could* supply, and its null (§4.9) is consistent with the paper's own ablation
rather than evidence against the paper.

---

## 3. Method

### 3.1 Data and frames

SQuAD, following the SEAL knowledge-incorporation protocol the paper uses. Six frames, all
closed-book:

- **Gate frame**: the first 20 passages of a 24-passage / 132-question file, giving 20 passages
  and 109 questions [B7]. Every result in §4.1-§4.7 and §4.9 is read on this frame.
- **Held-out frame**: the first 20 passages of a 50-passage / 229-question file, giving 92
  questions over text the student never trains on [B8]. Used for the non-destruction condition.
- **Wider frame**: all 229 questions, used only for the initial base reading [B3].
- **Corpus frame**: 200 passages and 923 questions drawn from the same source with the two
  small frames excluded, so no evaluation context or question is trained on [Q1]. This is our
  reconstruction of the paper's continued-pretraining regime at the paper's 974-question scale
  [P9]; the realised count is 923 and is recorded at build, since that figure is not exactly
  reproducible from the released data. The paper samples five dreams per passage in that regime
  [P10]; we sample eight and keep six (top-4 by gradient plus 2 random) [S6], and the
  answer-masked arms train on the question-answer pairs extracted from the kept dreams.
- **Paraphrase frame**: 473 of the 923 corpus questions rewritten by the teacher into a second
  surface form [Q20]. The registered protocol needed 800 accepted rows to make this the primary
  frame; 473 is 51.2 per cent, so the primary single-hop frame stays the corpus frame and the
  paraphrase read is reported as secondary [Q23].
- **Transfer frame**: 117 two-hop questions built by mining bridge entities out of the corpus
  dream cache, with 84 rows in the compositional stratum the base cannot answer from the second
  passage alone [Q21]. The frame is eval-only and frozen. Our own spot-read of the 20 accepted
  rows judged 16 of 20 degenerate in the first build and about 9 of 20 after the bridge predicate
  was tightened to proper nouns; the verdict is RECORD, the frame is reported pooled and
  stratified, and no admission criterion was tuned after a reading was seen.

### 3.2 Instruments

- **Read-out**: greedy, no-context generation, normalised exact match and token F1. The paper
  scores with an LLM judge; we use exact match, which is stricter, so absolute values are not
  comparable to the paper's [P1-P5] and we do not compare them (§4.1).
- **Leak probe**: the same questions with and without the passage in context. A delta below 10
  points means the no-context reading is leaking. Both frames used for decisions pass, with
  deltas of 42.5 and 62.5 points [B4, B5].
- **Routing share**: the fraction of actual top-k slot selections held by the new bank, pooled
  over tokens and layers, with the full 68-expert count vector logged so that any control is
  recomputable. The uniform null is 4/68 = 0.0588 [S5]. We also log the top-4 base experts'
  share and the hottest single base expert, because "the new bank holds a quarter of all routing
  slots" is only informative against those [T10].
- **Literalism**: median 8-gram containment of a dream in its source passage. Above 0.5 is an
  alarm: near-copies of `C` are rebuilt continued pretraining.
- **Groundedness**: `answer_support_rate`, the fraction of extracted question-answer pairs whose
  answer string occurs in the passage. This is telemetry, never a filter: filtering pairs by
  containment in `C` would give the sleep stage access to the wake context and void the
  copy-versus-store distinction the instrument exists to protect.
- **Prose overlap**: the fraction of a dream's content words that occur in the source passage.
  Added because the pair extractor is question-shaped by construction and cannot see prose
  dreams at all [G3, G4, G6].
- **Dose**: label tokens the loss actually scored, summed over a passage's fine-tuning steps.
  Recorded per arm because step counts and doses diverge by more than an order of magnitude
  across objectives [H11].
- **Perturbation telemetry**: for a dream pass written under perturbed routing, the fraction of
  routed tokens whose expert set differs from the standard top-k (`a1`), the mean probability of
  the experts drawn and of the experts they displaced, and the token count actually routed. This
  is the instrument that separates a real manipulation from a silent no-op [V5].
- **Routing identity at evaluation**: with the perturbation hook off, the evaluation forward pass
  must be bit-identical to the same pass with no routing state at all, and the same check must
  see a non-zero perturbation strength move the logits. A check that cannot fail is not a check
  [V6].
- **With-context guard**: the same questions with the passage in view, so that "the reading
  dropped" can be separated from "the model no longer reads at all". At corpus scale a drop of
  more than 10 points counts as damage [Q11].
- **Mid-training probes**: a fixed 200-question no-context subsample, read every 1000 steps, with
  a pre-registered abort rule (§3.4) so a collapsing arm can be stopped rather than paid for.
- **Statistics**: paired per-question bootstrap, 20000 resamples, seed 0, with the decision rule
  fixed before any number was read — PASS requires a gain of at least +2.0 points **and** a 95
  per cent interval whose lower bound is above zero; FAIL at a gain of zero or below;
  INSUFFICIENT otherwise. The pairing is always within a session against the same run's own
  untrained student [X6]. Two-arm contrasts are bootstrapped on the difference of the two arms'
  own-base gains, per question, never as two point estimates compared [V3].

### 3.3 Gates, fixed before measurement

- **Re-basing rule**: if our base reading differs from the paper's 31.9 by more than 5 points,
  the absolute gates are demoted and the primary gate becomes relative: the dream arm must lead
  the no-dream ablation by the paper's own 12.4-point margin [P8].
- **Degeneration guard**: a dream arm within 2 points of continued pretraining is a pipeline
  that has collapsed into rebuilt continued pretraining, not a sleep effect.
- **Four pooled conditions**: (i) gain over the run's own base of at least +2.0 points with the
  interval's lower bound above zero; (ii) dream arm at least 2 points above continued
  pretraining; (iii) dream arm at least 2 points above the no-selection ablation; (iv)
  non-destruction — the held-out reading may not drop by more than 2 points.
- **Arms**: a dream arm (gradient selection, context stripped), a continued-pretraining arm on
  the passage itself (same surgery, same warmup, same routing bias; only the data differs), and
  a no-selection ablation (all 8 dreams, whole-text loss). Every arm trains exactly 30 steps
  [S8] after the step mismatch of §4.3 was measured.
- **Row mapping to the paper's Table 3** [P1-P8]: our dream arm corresponds to Sleep (48.1,
  single passage); our continued-pretraining arm corresponds to the paper's Fine-tuned Model
  with No Dreaming (33.4), which trains on the passage without any dream stage; and our
  no-selection ablation — all eight dreams, whole-text loss, no gradient selection — has no
  row in the paper at all, whose removing-gradient-based-selection row (47.1) still selects
  [P6]. The 12.4-point bar [P8] is Sleep minus the paper's removing-Dreaming row (48.1 -
  35.7) [P4, P5], so it is a bar for a dream arm over an ablation that does not dream; we
  apply it to our dream arm over our own no-selection ablation, which still dreams, and we
  state that asymmetry rather than glossing it. The corpus column's counterpart is Sleep (44.3)
  against removing-Dreaming (36.2), a gap of 8.1 points [P15], and that is the bar our corpus-scale
  secondary read uses.
- **Corpus-scale guards** [Q11]: non-destruction is 3.0 points on the held-out frame and 10.0
  points on the with-context read. Both are wider than the 20-passage bound because the arm now
  spends 6000 steps on one adapter set instead of 30 per passage.
- **Pre-registered stop rule**: on the two read-outs (the single-hop frame and the transfer
  frame), STOP-NEGATIVE if the paired interval excludes zero against the dream arm on either
  read-out, or if the dream arm fails non-destruction while the comparator passes it; CONTINUE
  only if the interval excludes zero in the dream arm's favour on at least one read-out and not
  against it on the other; STOP-NULL if the interval contains zero on both read-outs and the
  point estimate is below 2.0 points in magnitude [Q22].
- **Residual-test verdict rule**: MECHANISM_ISOLATED if the paired interval of the difference
  between the two arms' own-base gains excludes zero in the perturbed arm's favour, DESTRUCTIVE
  if it excludes zero against it, INSUFFICIENT if it contains zero [V3].

### 3.4 The registered full protocol

The corpus-scale run was pre-registered before it was paid for, and the design is deliberately
narrow: three arms — dream (answer-masked question-answer pairs from gradient-selected dreams),
SFT-on-C (whole passage text, the paper's fine-tuned-with-no-dreaming row), and no-selection (all
eight dreams, whole-text loss, no gradient selection) — trained **strictly serially on one card**,
so that all three share one initial condition: same build, same 25-step warmup, same bisected
bias, same seed, and identical pre-training reads on all four frames [Q4]. Each arm trains 6000
steps, which is 30 steps per passage times 200 passages, and that is the dose every earlier gate
in this paper multiplied by ten. The dream cache is 1600 dreams written by the teacher at
temperature 0.7 with the quiz prompt, 2.85 GPU-hours [Q19]. One seed; a second seed exists only on
the INSUFFICIENT branch of the stop rule and was not run.

The abort rule is part of the registration: three consecutive probes more than 10 points below the
arm's own base **and** the latest more than 15 points below -> save, mark the arm destroyed, stop.
It did not fire, and §8.9 explains why it could not have.

### 3.5 The residual routing test

The residual test isolates the one ingredient of the dynamic this substrate can supply: routing
perturbation during dream generation. It is a paired experiment on the 20-passage gate frame, both
arms serial on one card, same seed, same warmup arm, same share band, routers frozen, 30 steps per
passage [V1, V2].

- **R-std** is the as-validated instrument: the cached quiz-at-0.7 dream set the earlier session
  used, so this arm is a same-instrument control and its reading is comparable to §4.4.
- **R-pert** generates its own dreams with two of the eight top-k slots replaced, during
  dream-generation forward passes only, by experts drawn from the router distribution with the
  top-k columns zeroed — sampled without replacement, substituted into the lowest-probability
  slots at the substitutes' own probabilities, distribution and temperature unchanged [V5].
  The base random-expert slot is kept in both arms, so the contrast is the replacement alone.

Both arms are then evaluated with the hook off, and the identity check of §3.2 is run on each
[V6]. Two arms of 30 steps cannot resolve a few points at 109 questions; the rule in §3.3 was
fixed in advance, and the resolution is reported beside the verdict.

---

## 4. Results

### 4.1 The reference frame does not transfer, so we re-based

Our substrate's no-context reading is far below the paper's. On the gate frame the frozen
4-bit base reads 11.93 exact match [B1]; on the wider 229-question frame, 6.99 [B3]; on the
held-out frame, 2.17 [B2]. Against the paper's 31.9 [P1] the gate-frame gap is -19.97 points
[B6], which fires the pre-registered re-basing rule. The absolute comparison to 48.1 is
therefore demoted, and every decision below is relative: within a session, against the same
run's own untrained student, with the paper's relative margin (12.4 points, single passage)
[P8] as the bar and its ablation ordering as the reference.

This is not a footnote. The paper's removing-dreaming ablation reads 35.7, which sits 12.4
points below its Sleep arm and *above* its own fine-tuned baseline (33.4) [P2, P4, P5]; that
ordering is the paper's statement about what the dream stage adds over training on the passage,
and it is the ordering our substrate tests. It is also the weakest of the paper's sleep-family
variants, below removing gradient-based selection (47.1) and removing random-expert routing
(48.0) [P6, P7] — the family is ordered by how much of the dreaming is left in it, not by which
sub-component was ablated. On our substrate the dream-versus-passage ordering did not reproduce:
the arm without a dream stage is statistically indistinguishable from the dream arm — 0.00
points, interval [-6.42, +6.42] (§4.4) — and at corpus dose it inverts (§4.8).

### 4.2 The first pooled attempt was destructive, for two measured reasons

With groundedness unfixed and routing unconstrained, the pooled gate over 20 passages read
14.68 to 11.01, a loss of 3.67 points against the run's own base [T1] (-0.92 against the
external 4-bit base [T2]; the two base frames differ by 2.75 points on the same questions
[T3]). Two causes were measured, not inferred.

**The dreams were not the passage's facts.** Mean answer support was 0.533 with 8.35 extracted
pairs per passage [T4] — roughly half of what the fine-tuning stage was given was not in `C`.
The prompt mattered: the quiz prompt solicited answer-bearing question lists where the facts
and recap prompts produced almost no extractable pairs at all [G3, G4].

**The new experts were selected on every text type we probed.** Their routing share was 0.2062
on the passage, 0.2835 on evaluation prompts and 0.2672 on the dreams themselves [T14] — read in
the three-cycle run, whose fine-tuning left the new router columns trainable [T16] —
against a uniform null of 0.0588 [S5]. Measured on text the student never trained on, the
contrast ran the wrong way: 0.1615 on the trained passage against 0.2585 on never-trained
passages (-0.0970), and 0.2385 against 0.2739 on prompts (-0.0354) [T8]. The same contrasts
were already present after warmup alone (-0.0996, -0.0317) [T9], so this was the router's
state, not an artefact of fine-tuning. The base-expert control rules out the obvious
alternative reading: on the same probes the top-4 base experts hold 0.0896 of slots on the
trained passage and 0.0731 on the trained question prompts, and the hottest single base expert
0.0234 and 0.0201 [T10], so the new bank was not behaving like a popular expert — it was
over-selected on every text type we probed.

A first injection probe suggested all fine-tuning arms destroyed the reading [T5], but its four
arms ran sequentially on one model instance without a weight reset, so it measured accumulated
damage, not any single objective; we report it only as the reason the next probe exists. The
clean probe started from a freshly built student: 13 answer-supported pairs, 30 steps, routers
frozen, gave 20.0 to 20.0 (delta 0.0) against a matched ungrounded control's 20.0 to 0.0
[T6]. Groundedness was **necessary but not sufficient** — and the freshness of both arms is
proven by first-step losses of 4.595 and 2.993 rather than near-zero [T7].

A three-cycle run on one passage added the shape of the failure: readings of 20.0 after warmup
and then 0.0, 0.0, 20.0 across the three cycles, with the training dreams memorised to a mean
loss of 0.0217 [T12, T13], and the
same questions reading 20.0 in the evaluation template, 0.0 in a question-answer template and
0.0 in the training template [T15]. The facts were stored and not elicitable, which is why the
next round changed the objective rather than the prompt format.

### 4.3 Three fixes flip the direction, and a confound is named by measurement

The second round changed three things, each aimed at a measured defect:

1. **Dream groundedness by prompt and temperature.** A sweep over prompt variant and
   temperature selected the quiz prompt at temperature 0.7: mean answer support 0.7495 with
   12.33 pairs per passage [G2], against 0.6556 and 15.08 at 0.9 [G1]. Containment was zero in
   every cell that produced content [G5], so the grounding was not achieved by copying `C`.
   The prose prompts, unmeasurable with a pair extractor, were later shown to differ sharply:
   prose overlap medians of 0.025 for the facts prompt, 0.905 for recap and 0.017 at the lower
   temperature [G6].
2. **A routing constraint on the new bank.** A single additive bias on the new router columns,
   bisected to hold the bank's share inside a [0.05, 0.09] band measured on a held-out tuning
   set — held-out passages and held-out question prompts, disjoint from every evaluation frame
   [R4] — converged to -0.75 for all 20 students in every arm [F11]: the tuning-set share fell
   from 0.2205 unbiased [A7] to 0.0577 inside the band [A2]. The band transfers only
   approximately to evaluation text. Here the evaluation readings landed at 0.0783-0.0905
   [F9], inside the band on most passages and above its ceiling on a few; in the step-matched
   session the continued-pretraining and recap arms breach the ceiling by up to 0.0063 while
   the dream arm stays inside [H15], and in this session all three arms reach above 0.09 at
   the top of their ranges — 0.0905, 0.0904 and 0.0907 [F9]. A tuned band is a bias on
   behaviour, not a guarantee about it. The contrastive warmup alternative did not learn
   content selectivity in 25 steps (post-tune contrast -0.008 against a 0.02 bar) [A4], so the
   capped warmup was carried forward on the pre-registered decision tree [A6].
3. **Routers frozen during fine-tuning**, so that the bisected bias is the sole routing control.

The pooled gate then read 13.76 to 14.68, a gain of 0.92 points with an interval of
[-3.67, +5.50] and 4 questions up against 3 down [F1] — INSUFFICIENT by the fixed rule,
but no longer destructive: the held-out reading *improved* from 2.17 to 4.13 (a drop of -1.96
points, inside the 2-point bound) [F7]. The no-selection ablation, however, read 19.27, a gain
of 5.50 points [F3], and the continued-pretraining arm read 14.68 — exactly the dream arm's
value [F2, F4].

That comparison was not yet readable, and we measured why: the arms were not step-matched. The
question-answering arm derived its step count from its pair count and trained 6.20 steps per
passage (range 4-15), while both comparators trained 30.0 [F8]. About five times fewer steps [F13] is
enough on its own to explain a 4.59-point deficit [F5], so neither "the dream arm ties
continued pretraining" nor "the ablation leads" could be read as evidence about dreaming.

### 4.4 Step-matched, the dream arm ties continued pretraining exactly

The third round fixed the step count at 30 for every arm [S8, H12] and added a
dose-comparable dream arm (§4.5). The dream arm then read 13.76 to 17.43, a gain of 3.67
points, interval [-2.75, +10.09], 9 questions up against 5 down [H1] — still INSUFFICIENT
against the +2.0-and-interval-above-zero rule. Continued pretraining, rerun in the same
session with the same schedule, read 17.43 as well [H3]. The paired contrast is **0.00 points,
interval [-6.42, +6.42], 6 questions up and 6 down** [H5]: the two arms are indistinguishable.
Against the no-selection ablation the dream arm sits 1.83 points lower, with an interval that
spans zero [H6, H4].

So the step confound is closed and the conclusion survives it: **on this substrate the dream
stage adds nothing measurable over training on the wake passage itself.** The one quantity
that separates the arms is the dose — 475.4 supervised tokens per passage for the
question-answering arm against 4665.0 for continued pretraining [H11] — and §4.6 explains why
we do not read that as a win.

Non-destruction held in every arm of both sessions: held-out drops of 0.65 points (dream arm),
0.87 (recap arm) and -0.87 (continued pretraining, i.e. an improvement) [H10], and -1.96 in all
three arms of the previous session [F7].

### 4.5 A dose-comparable dream arm is destructive

The dose-comparable arm gave the dream stage the continued-pretraining objective on dream
text, at a comparable dose. Recap-prompt dreams at temperature 0.9, whole-text loss,
30 steps: 5872.3 supervised tokens per passage, 1.26 times continued pretraining's 4665.0
[H14, H11]. The result was 13.76 to 9.17, a loss of 4.59 points with an interval of
[-10.09, +0.92] [H2], and -8.26 points against continued pretraining with an interval of
[-14.68, -2.75] that excludes zero [H7]. Against the no-selection ablation, -10.09 points
[-16.51, -4.59] [H8]. The dream arm with the question-answering objective leads this arm by
8.26 points [+1.83, +15.60] [H9] — but that contrast moves three things at once: the
objective, the dream variant (quiz against recap) and the dose (475.4 against 5872.3
supervised tokens, a factor of 12) [H11], so it locates the difference in the pipeline, not
in any one of them.

Two mechanisms are visible in the arm's own telemetry, and they are the two failure modes the
literalism and copy-filter instruments exist to catch. Literalism fired on 2 of 20 passages, with median 8-gram
containment of 0.5674 and 1.0000 [H16]: at the margin, this prompt degenerates into copying.
And the copy filter had to do real work — 13 of 20 passages trained fewer dreams than were
selected, with 3 and 4 dreams dropped on the two alarm passages [H17]. Mean dream length was
33.9 words (per-passage means 13.0 to 71.7) [H18] against 106.0 for quiz dreams and 120.7 for
passage text, with prose overlap 0.852 [H19], so this arm is a paraphrase source by
construction and still loses. Its warmup had to fall back to prose texts because prose dreams
yield no extractable pairs [H21], and the pair-groundedness instrument reads 0.000 on it
[H20] — an instrument limitation we record rather than resolve, since the prose-overlap
instrument is what actually measured it.

At comparable dose, dream prose is worse than the passage it was dreamt from.

### 4.6 The one favourable reading, and why it is not yet evidence

The dream arm reached the same reading as continued pretraining on about one tenth of the
supervised tokens: 475.4 against 4665.0, a ratio of 0.102 [H13]. Taken at face value this is
the paradigm's most attractive claim: equal recall for a tenth of the supervision.

We do not take it at face value, for a reason that is structural rather than statistical. The
question-answering arm trains on pairs formatted in the evaluation template with the loss
masked to the answer; continued pretraining trains whole-text on the passage. The arm therefore
sees the evaluation format during training and its comparator does not. A reading advantage
could be produced entirely by that alignment, with no additional injectable content. The
paraphrase control has now run (§4.8), and the reading stays **unproven**: on the paraphrase frame
the dream arm's gain over its own base is 2.11 points *below* the continued-pretraining arm's, with
an interval of [-3.81, -0.63] that excludes zero in the comparator's favour [Q23]. The
dose-efficiency reading is not merely unproven in this form; at corpus dose the direction reverses.

### 4.7 Two instrument findings that bound every threshold in this paper

**The same arm reads differently in different sessions.** Continued pretraining ran in two
sessions at identical settings — same surgery, same warmup, same bias, same 30 steps, same
dream cache, same evaluation batch — and read +0.92 points in the first session [F2] and +3.67
in the second [H3]: a 2.75-point swing [X1]. Question by question, the golds agree 109 of 109
and the untrained base reading agrees on its pooled value, but the base *predictions* differ on
18 of 109 and the trained predictions on 53 of 109 [X2], and the entire difference comes from
two passages (7: 0.0 to 16.7; 14: 20.0 to 60.0) — three questions of 109 [X3]. The base pass is
greedy with fixed weights, so the candidates are kernel-level nondeterminism in the 4-bit path
and session randomness in the uncached dream path. §4.9's residual test argues against the first
on the cached path, where the same arm reproduced the earlier session's base and final readings
and all 109 base predictions exactly [V8], and §4.9 attributes that swing to dream-set and
session randomness instead; the two runs here also differed in contention, one alone and one
concurrent with another arm. We record the mechanism as unproven and the consequence as certain:
**a +2.0-point threshold
sits inside this swing**, so every arm contrast in this paper is read on its paired interval,
never on its point estimate, and no number is compared across sessions.

**A borderline contrast did not reproduce.** The no-selection ablation led continued
pretraining by 4.59 points in the first session, with an interval whose lower bound sat exactly
at zero (6 questions up, 1 down) [F6]. In the step-matched session the same contrast read -1.83
points with an interval spanning zero (3 up, 5 down) [X4]. The tempting reading — that raw
dream text carries more injectable signal than the passage at equal steps — was inside session
noise and is not carried forward.

Both findings argue for the design we used rather than against it: 109 questions resolve 0.92
points per question [X5], which is why the pooled frame, the paired interval and the
same-session base were all necessary. They also explain why the 12.4-point paper margin is the
right bar and a +2.0-point threshold is not evidence of anything.

### 4.8 At the paper's corpus scale, on our step dose, the dream arm is the worst of three

The registered protocol spends 6000 steps on one adapter set — thirty times the per-passage
budget of every earlier gate, and a dose we set (30 steps per passage times 200 passages) rather
than one extracted from the paper — over 200 passages and 923 questions, which is the paper's
corpus scale [P9, Q1]. All three arms start from one
initial condition: identical pre-training reads on all four frames (corpus 10.94 exact match /
21.88 token F1, paraphrase 8.88 / 19.58, transfer 0.85 / 8.35, held-out 4.35 / 13.86) [Q4]. The
true no-surgery base on the same 923 questions is 11.48 / 22.11 [Q2], so the surgery and warmup
cost 0.54 points, measured inside the run.

The ordering inverts. Final corpus-frame readings: **dream 0.33, SFT-on-C 4.66, no-selection
11.48**, from an own base of 10.94 [Q5-Q7]. The dream arm loses to training on the passages
themselves by **4.33 points with an interval of [-5.74, -2.93]** (3 questions up, 43 down) [Q8],
and against the no-selection ablation it loses 11.16 points [-13.22, -9.21] [Q10]. Both of the
objectives that touch the facts directly are destructive against their own base — -10.62 points
[-12.68, -8.67] for the dream arm and -6.28 [-8.23, -4.44] for SFT-on-C — while the no-selection
dream-prose ablation survives at +0.54 [-1.41, +2.49], and does not gain [Q9].

The damage is not confined to the frame that was trained on. Held-out readings fall from 4.35 to
0.00 (dream arm) and to 1.09 (SFT-on-C), against a 3.0-point guard, and the with-context read —
the same questions with the passage in view, the reading that says the model can still read —
falls 32.0 and 20.0 points against a 10.0-point guard. The no-selection arm loses 2.17 points
held-out and 6.67 with context, passing both [Q11]. Both fact-facing arms destroy in-context
reading too, which is the strongest evidence that what happened is damage rather than injection.

Three measured facts shape the reading, and one is a mechanism we can only state as an
interpretation. First, **dose and accumulation**: the dream arm scores 94,187 supervised tokens
over 1924 examples revisited 12.5 times, SFT-on-C 3,474,120 tokens over 200 passages revisited
120 times, and the no-selection arm 4,482,846 over 1586 examples [Q12]; the passage arm therefore
supervises 36.9 times more tokens than the dream arm [Q13]. Second, the **loss
trajectories** say the same thing in the opposite direction: the two destructive arms fall to
0.0259 and 0.0246 by their last step (minima 0.0004 and 0.0077), while the surviving arm ends at
0.2428 with a minimum of 0.1526 — at least an order of magnitude above the other two throughout
[Q14]. Interpretation, marked as such: at
this dose the adapter bank becomes a retrieval table for a small number of short answer strings
(dream arm) or for 200 passages (passage arm), and it overrides the general no-context behaviour
that the reading depends on; the no-selection arm's 1586 diverse generated texts cannot be
memorised as cheaply, which is why it alone survives. Third, **literalism was armed and quiet**:
0 alarms in 200 passages in every arm, containment median 0.000 [Q16], so this is not the
copy-versus-store degeneration the literalism instrument exists to catch. The failure is
dose-and-objective, not dream quality.

The mid-training probes show the three arms diverging in kind, not degree [Q15]. Against a probe
base of 11.00: the dream arm reads 0.00 at every probe from step 1000 to 5000 and 1.50 at 6000 —
a total collapse; SFT-on-C oscillates between 3.50 and 7.00, a partially damaged student settling
near half its base; the no-selection arm rises from 11.00 to 14.00 and is the only healthy curve.

**The pre-registered stop rule reads STOP-NEGATIVE** [Q22]. The primary single-hop contrast
excludes zero against the dream arm (-4.33 points [-5.74, -2.93]), which is the rule's first
clause; the second clause — dream arm fails non-destruction while the comparator passes — does
not apply, because both fail it. CONTINUE was unavailable: it required an interval excluding zero
in the dream arm's favour on at least one read-out and not against it on the other.

Two reads must be reported as non-readings, and we report them rather than banking them. The
**transfer frame** nominally favours the dream arm on token F1: +3.17 points [+0.97, +5.47] pooled
over 117 rows, and +3.39 [+0.73, +6.17] on the 84-row compositional stratum [Q21]. But exact match
is 0.00 for the dream arm and 0.00 for SFT-on-C on that frame — the no-selection arm's pooled 1.71
comes entirely from the 33 non-compositional rows — and on the 84-row compositional
stratum it is 0.00 for all three arms. The base token F1 of 8.35 falls to 5.99 (dream arm) and 2.82
(SFT-on-C): a token-F1 interval at floor measures the degeneracy of degenerate answer text, not a
capability. The frame is recorded as invalid as a
capability probe for the reasons in §3.1, and the nominal interval is not used anywhere in this
paper's reasoning. The **paper-kind secondary read** is the corpus-column analogue of the
12.4-point bar: the paper's Sleep leads its removing-dreaming ablation by 8.1 points in that
column (44.3 against 36.2) [P15], and our dream arm against our own still-dreaming no-selection
ablation reads -11.16 points, with the armed bar unreached and the direction reversed [Q24].

Finally, the routing cap that worked at 30 steps does not hold at 6000. Evaluation shares are
0.1098 trained / 0.0953 held-out for the dream arm, 0.0666 / 0.0647 for SFT-on-C and 0.1142 /
0.1096 for no-selection, against a [0.05, 0.09] band [Q17]: the two arms that trained on dream
text breach the ceiling by up to 0.024, and the cap binds least in exactly those arms. A bias
tuned at one dose is not a bias at another.

### 4.9 Perturbation without coupling does not inject

The residual test asked whether the dynamic's most visible ingredient — writing the dreams from
an off-standard route — is enough on its own. Both arms ran serially on one card with one shared
initial condition: the per-passage pre-training readings are identical on 20 of 20 passages, so
the contrast is a dream-text difference and not an initial-condition one [V4].

**R-std**, the as-validated instrument, read 13.7615 to 17.4312, a gain of 3.6697 points over its
own base, interval [-2.75, +10.09], 9 questions up against 5 down, token F1 +3.60 [-2.83,
+10.35], held-out 3.261 to 2.826 (drop 0.43), 9508 supervised tokens [V1]. Those are the earlier
session's dream-arm numbers to four decimal places — same base, same final, same gain — measured
on a different card in a different region [V8]. That is this project's reproducibility anchor, and
it also bounds §4.7: the earlier cross-session swing (+0.92 against +3.67 for the same arm) is not
a property of the card or of the evaluation instrument, and is consistent with dream-set and
session randomness in the uncached path.

**R-pert** read 13.7615 to 13.7615: gain +0.00 points, interval [-6.42, +6.42], 6 up and 6 down,
token F1 +1.59 [-4.59, +7.85], held-out 3.261 to 3.478 (no damage), 9094 supervised tokens [V2].
The paired contrast on the difference of the two arms' gains is **-3.67 points, interval [-10.09,
+2.75]** on exact match and -2.01 [-8.79, +4.44] on token F1 [V3]: **INSUFFICIENT** by the rule
fixed in advance. What the interval does establish is that there is no positive effect anywhere
near the hypothesised size, on either metric, and that the point estimate is negative.

The manipulation was real, which matters because a null from a silent no-op would mean nothing.
Every routed token in the perturbed pass was off-standard (`a1` = 1.0) over 1,148,416 tokens; two
of eight top-k slots were replaced; the substitutes carried a mean probability of 0.0137 against
the 0.0292 of the experts they displaced [V5]; and 0 of 160 dreams are verbatim identical across
the two sets [V7]. The two dream sets also differ in groundedness, and that is a confound we name
rather than bury: answer support fell from 0.744 to 0.673 and extracted pairs per passage rose
from 11.45 to 13.50, with mean dream length 106.0 to 114.9 words and no literalism alarm in either
arm [V7]. The arm therefore varies routing **and** dream factuality together, and both changes
point the same way — towards less injectable content. The honest sentence is "off-standard-routing
dreams, which are also less factual, did not inject", not "routing perturbation is inert".

The evaluation path is provably untouched: with the hook off, the evaluation forward pass is
bit-identical to the same pass with no routing state (maximum absolute difference 0.0), and the
same instrument sees a non-zero perturbation strength move the logits by 9.06 [V6]. The check can
fail, so its pass means something.

Read against §2.2, this is consistent with what the substrate predicts rather than a surprise: we
supplied the perturbation and not the coupling, and the perturbation alone cost the injection it
was supposed to add. It is also consistent with the paper's own ablation, which prices random routing
specifically at 0.1 point [P4, P7].

---

## 5. The substrate thesis

### 5.1 The four outcomes, in the order we measured them

We did not set out with this thesis; we arrived at it by eliminating alternatives, and it is worth
saying what it explains. The claim is that the sleep paradigm's **plumbing** is expressible on a
frozen-router QLoRA mixture of experts while its **coupled dreaming dynamic** is not (§2.2), and
that the four things we measured are each consistent with what that substrate would predict.

| outcome | measurement | why the substrate predicts it |
|---|---|---|
| consolidation bounded | the bank can be capped near its null share, but the trained-minus-held-out share contrast never becomes meaningfully positive: -0.0970 and -0.0354 at the first probe [T8], -0.011 and +0.0149 after the cap [A2, I4] — the largest reading is +0.0149, on a single-passage probe | the new columns are trainable, but they sit on a router whose base is quantisation-frozen: co-training them over-attracts on every text type (§4.2) and was frozen by pre-registered design, so routing can be biased but never trained on the dreams a perturbed route writes (§2.2) |
| injection absent | +3.67 points over own base, indistinguishable from SFT-on-C (0.00, [-6.42, +6.42]) [H5] | the dreams are written by the unexpanded teacher, whose forward pass contains no reserve capacity to route through — consistent with the thesis, though this contrast is also objective- and dose-confounded (§4.6) |
| corpus-scale destruction | dream 0.33 < SFT-on-C 4.66 < no-selection 11.48, from 10.94; dream-minus-passage -4.33 [-5.74, -2.93] [Q5-Q8] | with no coupled dynamic, the adapter bank has nothing to learn but the training set itself, and 6000 steps on one adapter set is a memorisation schedule (§4.8, §5.2) — a reading confounded by the dose (§5.2) |
| perturbation alone did not inject | +0.00 against +3.67, contrast -3.67 [-10.09, +2.75], INSUFFICIENT [V3] | the perturbation moves the route among frozen base experts, away from the standard route, without touching the capacity under test (§4.9); the perturbed dreams were also less grounded, so the null is joint [V7] |

Read as a sequence, the four outcomes are one argument: the plumbing runs (the pipeline builds,
trains, evaluates and passes its own integrity checks), it does not inject, at scale it damages,
and the one ingredient of the dynamic we could add made it worse rather than better. A substrate
that could express the dynamic would be predicted to show at least one of: content-selective
routing, injection above the passage-trained comparator, or a positive effect from perturbed
dream-writing. We observed none of the three.

### 5.2 What this thesis does and does not explain

It **does** explain why the plumbing reproduces so cleanly. Every integrity check in this project
passes: the routing cap bisects to the same bias in every student [F11], the dream cache
reproduces bit-for-bit [V8], the arms share one initial condition [Q4, V4], and the report scripts
recompute their own numbers. The pipeline is not broken. It is a faithful implementation of a
mechanism whose load-bearing part is missing, which is exactly the failure mode that a
mechanism-assay framing exists to make visible.

It **does not** explain the corpus-scale ordering on its own, and we will not pretend otherwise.
The dose is a confound we can name but not remove: the passage arm supervises 36.9 times more
tokens than the dream arm [Q13], and a 200-example training set revisited 120 times [Q12] is a
memorisation schedule by construction. A substrate that expressed the dynamic might well collapse
at this dose too. What our data establish is the narrower and sufficient claim: **at the paper's
corpus scale, at a dose we set (30 steps times 200 passages), on this substrate, the dream stage
is worse than training on the passage.**

It **cannot** attribute the corpus-scale failure to the missing dynamic alone. The paper's own
ablation prices random routing specifically at 0.1 point [P4, P7] and gradient-based selection at
1.0 point [P4, P6], while dreaming as a whole is worth 12.4 [P4, P5]. Dreaming as a whole is
what is load-bearing, and the dynamic is the part this substrate never supplied in full — and
"the dynamic" is itself a compound of perturbation, coupling and routing trained on the dreams
that perturbation writes; the paper's own ablation does not apportion the 12.4 among those parts,
our residual test isolates only the first, and the routing half was trained only on
standard-routed dreams (§2.2).

And one part of the paradigm was **never tested here at all**: the outer loop (ReSTEM), in which
the dream policy itself is trained. Our generator was fixed at the pretrained teacher for every
run in this paper. A trained generator is a different mechanism from a fixed one, and nothing in
our results speaks to it.

---

## 6. Limitations

**This is a mechanism assay, not a capability claim.** The read-out is closed-book recall of
facts from passages read once, scored by exact match rather than the paper's judge model, on two
frames: 20 passages and 109 questions for §4.1-§4.7 and §4.9, and 200 passages and 923 questions
for §4.8. Nothing here licenses a statement about what the model "knows" or how well it answers
questions. Absolute values are not comparable to Table 3 of the paper being replicated, for three
independent reasons: a different backbone, 4-bit quantisation with a frozen base, and a stricter
scoring convention [P1-P5].

**The memory hierarchy, the consolidation step and the outer loop are absent** (§2). A
negative on this substrate bounds *selection plus fine-tuning on a frozen-4-bit expert bank*;
it does not bound the full architecture.

**The step-matched comparison still differs in objective and dose.** At 30 steps the arms
differ by an order of magnitude in supervised tokens (475.4 against 4665.0) [H11], and the
question-answering objective was our design choice to make stored facts elicitable — a choice
that did not make them elicitable. Condition (ii) is therefore a comparison of pipelines, not
of dream content against passage content at equal information.

**Format alignment confounds the dose-efficiency reading** (§4.6). The paraphrase control ran
and did not rescue it: the dream arm's paraphrase-frame gain is 2.11 points below the
continued-pretraining arm's, interval [-3.81, -0.63] [Q23]. The reading is not merely unproven;
in this form it is refuted, and we state it as refuted rather than as pending.

**Cross-session variance bounds every threshold.** The same arm read +0.92 and +3.67 in two
sessions [X1], with 18 of 109 base and 53 of 109 trained predictions differing [X2]. The arms
within the step-matched session were also not run under identical contention, and one arm
(the no-selection ablation) was reused across sessions rather than rerun.

**The prose-dream instrument was built after the fact.** The pair extractor cannot see prose
dreams at all [G3, G4]; the prose-overlap instrument that can was added late [G6, H19], so the
recap arm's groundedness rests on a measure introduced mid-project. Its temperature was chosen
at 0.9 on the strength of an earlier probe whose artefacts did not return from the rented card,
so that choice is supported by the arm's own telemetry [H16, H18, H19] rather than by a
reproducible pre-arm comparison.

**The routing cap bounds but does not eliminate the over-attraction, and it does not survive the
corpus dose.** Post-tune contrasts between trained and held-out text sit between -0.011 and
+0.0149 [A2, I4] — near zero, not at the positive value a selective store needs. The cap also
does not hold exactly: it is tuned on a held-out tuning set and transfers only approximately to
evaluation text, where the 0.09 ceiling is breached in every session — all three arms of the
first pooled session reach 0.0905, 0.0904 and 0.0907 at the top of their ranges [F9], the
step-matched session's continued-pretraining and recap arms breach it by up to 0.0063 [H15], and
at corpus scale the two arms that trained on dream text reach 0.1098 and 0.1142 against the same
ceiling [Q17]. A bias bisected at 30 steps is not a bias at 6000.

**Single-passage frames are unusable as evidence here.** Five questions give 20 points per
question; every decision in §4.1-§4.7 and §4.9 was taken on the pooled 109-question frame, and
§4.8's on 923 questions.

**Perturbation without coupling is not the dynamic.** Our residual test perturbed the route
during generation, but the routing and the reserve capacity were trained on the
**standard-routed** dream set — every arm from §4.3 on freezes the routers during fine-tuning
(§2.2) — and it could not make the perturbed route pass through consolidated capacity, because
the generator is the unexpanded checkpoint and carries no reserve bank in its forward pass at all
[S9]. The verdict is therefore a statement about a generation-time sampling intervention, not
about the paper's coupled mechanism. The confound compounds it: the perturbed dreams were also
less grounded (answer support 0.744 to 0.673) [V7], so the arm varies routing and dream factuality
together, and we did not run the missing control — a less-grounded but standard-routed arm, which
a higher sampling temperature would supply.

**Single seed at corpus scale.** The 200-passage run pre-registered one seed, with a second only on
the INSUFFICIENT branch of the stop rule; the verdict came out STOP-NEGATIVE, so the second seed
was never run [Q22]. The corpus-scale ordering is therefore one draw. The arms share an initial
condition and are compared on paired intervals [Q4], which removes the initial-condition variance
but not seed variance in training, and §4.7 measured how large that can be on the small frame.

**The sample size the corpus run bought is partly spent on a frame we cannot read.** The transfer
frame is at floor for all three arms, so the one read-out that could have distinguished
consolidation from ordinary storage contributes a non-reading (§4.8), and the compositional
stratum is 84 rows. The paraphrase frame, which the protocol intended as the primary single-hop
frame, reached 473 of the required 800 rows and is reported as secondary [Q20].

---

## 7. What would change our mind

**A pre-committed stop rule, and what it read.** The full-protocol run compares the dream arm with
continued pretraining, question by question, on two read-outs: the single-hop frame and the
transfer frame. The rule was fixed before that run was designed, so that a tie cannot be
re-described as a partial success afterwards. At 200 passages and the paper's question scale
[P9, Q1] it stops on this substrate (STOP-NULL) if the paired interval contains zero on
**both** read-outs **and** the point estimate is below 2.0 points in magnitude. A zero-spanning
interval whose point estimate is 2.0 points or more in the dream arm's favour is not a tie but
INSUFFICIENT, and it carries exactly one pre-registered remediation: a seed-1 rerun of the dream
and continued-pretraining arms only, with the decision then taken on the pooled two-seed paired
bootstrap — a second INSUFFICIENT stops the project and is published as STOP-INSUFFICIENT. A loss
that excludes zero on either read-out, or a dream arm that fails the non-destruction guard while
continued pretraining passes it, is STOP-NEGATIVE. The run continues only if the interval excludes
zero in the dream arm's favour on at least one read-out and does not exclude zero against it on
the other.

The registered single-hop frame was the paraphrase frame, and the protocol carried its own
fallback: if fewer than 800 paraphrases were accepted, the primary single-hop frame reverts to the
corpus frame and the paraphrase read is reported as secondary. That is what happened (473 accepted,
51.2 per cent [Q20]). The rule then read **STOP-NEGATIVE**: -4.33 points [-5.74, -2.93] against the
dream arm on the primary frame [Q8]. The transfer frame's nominal token-F1 interval in the dream
arm's favour does not change that read, for the reason given in §4.8 — exact match is at floor on
that frame (0.00 for both arms in the contrast, 0.00 for all three on the compositional stratum) —
and it is recorded in the report as a non-reading rather than used.

**What would have to change our mind.** Four outcomes, in ascending strength: (a) a dream arm
above continued pretraining by the paper's relative margin [P8]; (b) that advantage surviving the
paraphrase control, which would make the dose-efficiency reading real; (c) a transfer advantage on
compositional questions that continued pretraining does not show, which is the only outcome that
would distinguish consolidation from ordinary continued-pretraining-style storage; and (d) a
positive effect from perturbed dream-writing once the routing is trained on the perturbed dreams
rather than merely sampled at generation time. None of (a)-(c) was observed at corpus scale, and
(d) was not testable here: it needs the coupling, not the perturbation alone (§2.2).

**What would not change our mind.** A larger absolute gain in the dream arm without a paired
advantage over continued pretraining; a gain that appears only in the evaluation template; a
token-F1 interval at floor; or any reading that requires comparing across sessions, given §4.7.

**What we would have to see to withdraw the thesis.** A substrate that expresses the full
compound — a route perturbed while the dreams are written, through reserve capacity that is
itself being trained, with the routing trained on the dreams that perturbation produced — and
still shows the ordering we measured, would refute the thesis and leave a substrate-independent
negative. We have built one half of it: the routing and the reserve columns were trained together
in the diagnostic runs of §4.2, against standard-routed dreams, and that half read no better than
its own base at the five-question resolution it was measured at [T16] (§6). The other half,
training on the perturbed distribution, we have not built, and we do not know of an open-weights
substrate at this scale that supplies it.

---

## 8. Pitfalls for future replicators

Every item below produced a plausible green reading on our own runs before an instrument
exposed it. Each is stated as the symptom we actually saw, the instrument that separated the
symptom from the mechanism, the fix, and what it would have cost us to miss. They are the
transferable part of a negative result: the negative is substrate-bound, the instruments are
not.

### 8.1 A healthy loss curve over a router that has learned nothing but "here"

**Symptom.** The warmup loss falls and the new bank's routing share rises, so every diagnostic
that watches the optimiser looks healthy — but forced-inclusion routing puts the new columns
into the top 8 on *every* input, so they are trained on positive text only and nothing ever
tells them not-here. In the three-cycle diagnostic run the bank held 0.2062 of all routing
slots on the trained passage, 0.2835 on evaluation prompts and 0.2672 on the dreams themselves
[T14], and the trained-versus-held-out contrast was already wrong-signed after warmup alone
(-0.0996 and -0.0317) [T9] — the over-attraction was the router's state before fine-tuning
touched it.

**Instrument.** Routing share on trained text against never-trained text, read from the full
68-expert count vector so any control is recomputable, with a base-expert control beside it:
the top-4 base experts hold 0.0896 and 0.0731 of slots and the hottest single base expert
0.0234 and 0.0201 [T10], so a bank at a quarter of all slots is not behaving like a popular
expert. The gradient on the new router columns [W2] tells you whether the warmup trains
anything; the share contrast tells you what it trained.

**Fix.** Cap the bank — one additive bias on the new router columns, bisected to a held-out
share band, with routers frozen during fine-tuning so the bias is the sole routing control
(§4.3). Re-bisect at every new dose: a bias tuned at 30 steps is not a bias at 6000, and the same
0.09 ceiling was breached by 0.1098 and 0.1142 at corpus scale [Q17] (§4.8).

**Cost of missing it.** The first pooled gate read -3.67 points [T1] and the deficit was
attributed to the dreams until the share contrast named a second cause; a groundedness fix
alone could not have rescued it, because the damage was in the router.

### 8.2 Arms that are not step-matched

**Symptom.** A comparator leads the dream arm by 4.59 points [F5] — and against the other
comparator the same 4.59 comes with an interval whose lower bound sits exactly at zero [F6] —
which reads as a finding about dream content.

**Instrument.** Steps per passage, logged per arm [F8]: 6.20 (range 4-15) for the dream arm
against 30.0 and 30.0 for its two comparators, a factor of 4.84 [F13].

**Fix.** A fixed step count for every arm, overriding any step count derived from the arm's own
data (§3.3).

**Cost of missing it.** The ablation lead was an artefact. Step-matched, the same contrast
reads -1.83 points [H6, X4] and the tempting reading — that raw dream text carries more
injectable signal than the passage at equal steps — had to be withdrawn (§4.7).

### 8.3 Steps are not the dose

**Symptom.** Equal readings at a tenth of the supervised tokens [H13] read as efficiency.

**Instrument.** Label tokens the loss actually scored, counted inside the training loop and
summed per passage [H11]: 475.4 (range 210-880) for the answer-masked arm against 4665.0
(2100-7680) for whole-text continued pretraining and 5872.3 (2070-16440) for the recap arm.
The estimate we made before the counter existed put the gap at 40 to 290 times [X8]; measured
on the means, it is 9.8 times [H13]. Equal steps and equal dose are different claims, and only
the first was matched here.

**Fix.** Publish both per arm and read a contrast only where the dose is comparable — which is
why a dose-comparable dream arm exists at all (§4.5).

**Cost of missing it.** Without the counter the recap arm would have been described as "a dream
arm at comparable dose" while scoring 1.26 times continued pretraining's tokens, and the
answer-masked arm's equal reading would have been a win by construction.

### 8.4 Training in the evaluation format

**Symptom.** The answer-masked arm trains on pairs formatted in the evaluation template, gains,
and the gain looks like injection.

**Instrument.** The same questions in three templates on one checkpoint: 20.0 in the evaluation
template, 0.0 in a question-answer template and 0.0 in the training template [T15]; and, at
full power, a paraphrase frame on which template alignment cannot help (§4.8).

**Fix.** Report the objective difference instead of pretending to control it, and decide the
dose-efficiency reading on the paraphrase frame, not on the original one (§4.6, §4.8).

**Cost of missing it.** The one reading favourable to the paradigm would have been published as
a result when it may be entirely a template effect (§4.6).

### 8.5 Probes that share one model instance

**Symptom.** Every fine-tuning arm reads 0.0 after a base of 20.0 and a post-warmup reading of
40.0 [T5] — a clean structural claim that fine-tuning of any kind destroys the reading.

**Instrument.** The first-step loss of the student actually being trained: 4.595 and 2.993, not
near zero [T7]. A reused instance starts near zero because it has already been trained.

**Fix.** Build a fresh student for every probe arm and re-measure.

**Cost of missing it.** An entire structural diagnosis was wrong — the probe measured
accumulated damage across four sequential arms sharing one set of weights, and the clean probe
reads 20.0 to 20.0 against a matched control's 20.0 to 0.0 [T6].

### 8.6 Dreams that degenerate into copies

**Symptom.** A lower sampling temperature produces longer, more on-topic dreams, which looks
like better grounding.

**Instrument.** 8-gram containment of each dream in its source passage, computed every cycle,
with an alarm at 0.5 [R7], beside prose overlap and dream length [H18, H19]. An early probe at
temperature 0.7 read near-copies of the source passages; that probe's artefacts did not return
from the rented card, so the paper's own evidence is the arm's telemetry, which fired the alarm
on 2 of 20 passages at 0.5674 and 1.0000 [H16].

**Fix.** Choose the temperature by measurement rather than by assumption, keep the containment
alarm and the copy filter, and record the filter's work: 13 of 20 passages trained fewer dreams
than were selected [H17].

**Cost of missing it.** Near-copies of `C` are rebuilt continued pretraining, so the dream arm
silently becomes its own comparator and any gain it shows says nothing about the sleep stage.

### 8.7 Comparing across sessions

**Symptom.** The same arm at identical settings reads +0.92 points in one session and +3.67 in
another [X1], so a single-arm gain read against a stored base can land on either side of a
2-point threshold.

**Instrument.** Per-question pairing against the run's own untrained student, plus the
agreement counts: golds identical 109 of 109, untrained base predictions differing on 18 of 109
and trained predictions on 53 of 109 [X2], with the entire difference carried by two passages
[X3].

**Fix.** Own-base paired intervals inside one session, one evaluation batch size throughout
[X6], and no number compared across sessions — including the reuse of an arm trained in an
earlier session [H4].

**Cost of missing it.** A borderline contrast (+4.59 points, interval touching zero [F6]) did
not reproduce (-1.83 points [X4]) and had to be withdrawn; a threshold that sits inside the
swing will invert a verdict.

### 8.8 Stored artefacts that outlive the instrument that wrote them

**Symptom.** A report reads a verdict, label or contrast that an earlier version of the
instrument wrote, and prints green.

**Instrument.** Recompute every verdict from the rows at report time and fail if the stored
value disagrees. A verdict regeneration pass relabelled 3 cells from FAIL to NO_PAIRS with 0
mismatches against the stored support and pair counts [G7], and a sign assertion compares every
contrast against the two arms' own pooled values.

**Fix.** Keep the recompute and the assertion inside the shipped report script, not in a
one-off check, and version the stored artefacts the report reads (Appendix A).

**Cost of missing it.** Our report script labelled a contrast with one sign and computed it with
the other, printing +8.26 points where the arms' own values say -8.26 and marking two
conditions PASS: a false pass on the only favourable reading in this paper (§4.6).

### 8.9 A stop rule anchored below the floor

**Symptom.** The arm collapses totally — the fixed probe reads 0.00 exact match at every
checkpoint from step 1000 to 5000 against a base of 11.00 [Q15] — and the pre-registered abort
rule stays silent, so the run continues and the collapse is read only at the end.

**Instrument.** The rule itself, and the arithmetic that should be done on it before it is armed:
"three consecutive probes more than 10 points below the arm's own base **and** the latest more
than 15 below" [Q27]. With a base of 11.00 points, a delta of -15 is an exact match of -4.0. The
second clause cannot be satisfied by any reading the instrument can produce, so the rule reduces
to its first clause and never fires.

**Fix.** Anchor the tail condition on the available headroom, not on a fixed point count: require
the latest probe to be below half the own base, or express the tail as a fraction of the floor
(minus 10 of a possible 11, here) rather than an absolute -15. It cost nothing in this run — the
arm was going to be read regardless — but a paid arm that should be stopped will not be.

**Cost of missing it.** An unbounded spend on a destroyed arm, and a damage reading published
without its mid-training trajectory to corroborate it.

### 8.10 A perturbation at inference is not a co-trained router

**Symptom.** A routing perturbation during dream generation produces materially different dreams
(0 of 160 verbatim, every routed token off-standard [V5, V7]) and changes nothing measurable in the
student — a null that invites the conclusion "routing perturbation does not matter".

**Instrument.** Two things, and the second is the one that is usually missing. The perturbation
telemetry (`a1`, drawn against displaced probability, token count) proves the manipulation
happened [V5]; the eval-identity check proves the hook is off where the reading is taken, and its
asymmetry control proves the check can fail [V6]. What neither instrument can show is whether the
perturbed route passed through capacity that had been consolidated — here it could not, because
the generator is the unexpanded checkpoint and carries no reserve bank in its forward pass
[S9].

**Fix.** State the coupling explicitly before running the arm: which parameters are perturbed,
which are trained afterwards, and whether the perturbed path touches the capacity under test. If
the answer to the third is "no", the experiment measures a sampling intervention, and should be
reported as one. Control the co-varying property too — here the perturbed dreams were also less
grounded (0.744 to 0.673) [V7], and a higher-temperature standard-routed arm would have separated
the two.

**Cost of missing it.** A null read as "the mechanism does not matter" when it was never the
mechanism under test — and, worse, a positive result would have been read as isolating a mechanism
it had not isolated either.

### 8.11 A frozen teacher regenerates the same dreams every cycle

**Symptom.** A three-cycle run reads as three independent consolidation passes. With the teacher
frozen and one sampling seed it is not: every cycle regenerates the same kept dream set, so the
arm trains one dream set three times. The per-cycle readings still swing without the arm moving
(0.0, 0.0, 20.0 over three cycles) [T12], which invites the reading that the cycle count is the
variable.

**Instrument.** The dream text itself, compared across cycles. Regenerating the input verbatim is
what a frozen generator does with a fixed seed, so a multi-cycle run measures repetition, not
accumulation, unless the seed advances per cycle [T17].

**Fix.** Advance the per-cycle dream seed in the runner, so cycle `c` samples with seed
`seed + 1000 * c` and the warmup runs on cycle 0 only (`scripts/run_cycle.py`, the seed comment
there); then verify that the cycles' dream texts actually differ before reading any per-cycle
number.

**Cost of missing it.** A cycle count silently becomes a repetition count, and the conclusion
that more cycles do not help is an artefact of training three times on one dream set rather than
a statement about accumulation.

---

## 9. Conclusion

We re-hosted a sleep paradigm on a substrate that can express its plumbing and not its coupled
dreaming dynamic, and we measured four things: a reserve bank that can be capped but not taught to
be selective; a dream stage that does not inject on a small frame; a corpus-scale run in which the
dream arm is the worst of three and both fact-facing objectives destroy the closed-book reading;
and a routing perturbation that changed the dreams materially and the student not at all. Each is
consistent with what the substrate predicts, and together they are the argument.

The result we would most want a reader to take is not the negative but its boundary. The paper's
mechanism is a coupling — dreams written from an off-standard route through capacity that is being
trained to hold them — and a substrate that supplies the perturbation, the reserve capacity and the
synthetic text but not the coupling is not a test of that mechanism. It is a test of the parts, and
the parts do not carry it. The paper's own ablations set the scale for that claim: removing random
routing costs 0.1 point while removing dreaming costs 12.4 [P4, P5, P7] — the load-bearing part is
dreaming as a whole, and its coupled core, a perturbed route trained on the dreams it writes, is
the part this substrate never supplied.

What transfers is the instrument set (§8), not the verdict. Eleven failure modes produced plausible
green readings here before an instrument separated the symptom from the mechanism, and every one of
them was found on our own runs rather than in the paper's.

---

## Appendix A — reproduction

All artefacts are under `~/research/qlora-sleep/`. Every measured or derived number in this
paper carries an id that resolves to a row in `numbers.md` (deposit root), which names the file and the
key it was read from.

| paper section | artefacts |
|---|---|
| §2 substrate and surgery | `configs/base.yaml`; `runs/m1_warmup/metrics.jsonl` (stage=student) |
| §4.1 base readings and re-basing | `runs/b0_base_gate20/summary.json`, `runs/b0_base50_20/summary.json`, `runs/m0_base/summary.json` |
| §4.2 first pooled gate and diagnosis | `runs/m3_gate20_qa2/summary.json`; `runs/probe_grounded_fresh/summary.json`; `runs/probe_heldout/summary.json`; `runs/m3_cycle3b/summary.json` |
| §4.3 fixes and second session | `runs/e1_factuality/summary.json`, `runs/e1_quiz07/summary.json`, `runs/e2_arms/summary.json`, `runs/e3_samoa/summary.json`, `runs/e4_sleep`, `runs/e4_cpt`, `runs/e4_nodream` |
| §4.4 step-matched session | `runs/e5_sleep_qa`, `runs/e5_cpt`, `runs/e4_nodream`, `runs/e5_report.json` |
| §4.5 dose-comparable dream arm | `runs/e5_sleep_recap`, `runs/e5_report.json` |
| §4.7 instrument findings | `runs/e4_cpt/summary.json` against `runs/e5_cpt/summary.json`; `runs/e5_report.json` |
| §4.8 corpus-scale protocol | `runs/p6_true_base/summary.json`; `runs/p6_s`, `runs/p6_c`, `runs/p6_n` (`summary.json` + `metrics.jsonl`); `runs/p6_report.json`; `runs/p6_dreams.pt`; `data/squad_corpus200.jsonl`, `data/corpus_para.jsonl`, `data/transfer_twohop.jsonl`; `runs/p6_frames/{build,mining}.json`, `runs/p6_frames/spot_read.txt`; `boxlogs/p6_{s,c,n,dreams}.log` |
| §4.9 residual routing test | `runs/r_std`, `runs/r_pert` (`summary.json` + `metrics.jsonl`); `runs/residual_report.json`; `runs/e4_dreams.pt` against `runs/r_pert_dreams.pt`; `boxlogs/{r_std,r_pert}.log` |
| §8 pitfalls | the artefacts cited in §8.1-8.11: `runs/m3_cycle3b`, `runs/probe_heldout`, `runs/m1_warmup`, `runs/e1_factuality`, `runs/e4_*`, `runs/e5_*`, `runs/e5_report.json`, `runs/p6_*`, `runs/r_std`, `runs/r_pert`, `runs/residual_report.json`; `scripts/run_cycle.py` (the per-cycle dream seed of §8.11) |

Commands that regenerate the statistics (CPU only, no GPU):

```
python3 scripts/bootstrap_gate.py runs/e5_sleep_qa/summary.json        # own-base paired
python3 scripts/bootstrap_gate.py runs/e5_sleep_qa/summary.json runs/e5_cpt/summary.json
python3 scripts/e5_report.py --out runs/e5_report.json                 # nine contrasts
python3 scripts/plan3_report.py --out runs/p6_report.json              # corpus arms, stop rule
python3 scripts/residual_report.py --out runs/residual_report.json     # paired perturbation arms
python3 scripts/regen_factuality_verdicts.py                           # E1 verdict relabel
```

`scripts/e5_report.py`, `scripts/plan3_report.py` and `scripts/residual_report.py` each assert that
every contrast's sign agrees with the two arms' own pooled values, and each recomputes its stored
numbers from the per-question predictions. The assertion exists because an earlier version of the
first script reported the recap-versus-continued-pretraining contrast with the wrong sign and would
have published a false pass (§8.8).

Two artefact defects are recorded rather than repaired, because the measurement survived them and
the repair would cost a run: `runs/r_pert/summary.json` carries `dream_perturb.source = "cache"`
although that session generated its dreams (the true record is `n_passages_with_stats = 20`,
`tokens = 1,148,416` and the log line `[dream] 20 passages in 1084s`), and its
`eval_routing_identity` field is null because the identity check lives inside the passage loop,
which a resumed run skips — the measurement is in `boxlogs/r_pert.log` and in `runs/r_std/summary.json`
on the same build.

## Appendix B — cost accounting

| item | hours | rate | cost |
|---|---|---|---|
| session 1, H100 SXM5 80GB, us-south-2 (M0-E4) | 2.72 | $4.29/h | $11.67 |
| session 2, A100 SXM4 40GB, us-west-2 (E5) | 4.20 | $1.99/h | $8.36 |
| session 3, A100 SXM4 40GB, us-west-2 (full protocol) | 15.92 | $1.99/h | $31.68 |
| session 4, A100 SXM4 40GB, us-east-1 (residual test) | 2.20 | $1.99/h | $4.38 |
| local RX 9070 (M0-M3, rootful podman ROCm container) | — | owned hardware | $0 |
| **total rental** | **25.04** | — | **about $56.09** |

The full protocol's pre-run projection was 13.9 GPU-hours at about $28; the session ran 15.92 hours
at about $31.68, inside the $60-90 envelope approved for it, with the overrun concentrated in the
dream pass (2.85 hours against a 1.8-hour projection). The residual session ran 2.01 GPU-hours of
work in a 2.2-hour window at about $4.38, inside its $10-15 envelope. Earlier projections were
about $5-8 for 3 GPU-hours (E4) and a cap of about $6.4 for the E5 session, which was written for an
H100 at $4.29/h; no H100 capacity was available, so the same money bought a longer A100 session and
the overrun is in GPU-hours, not dollars. Hours and dollars are rounded to two decimals from the
published rates; there is no invoice in the repository, so these rows are RECORD rather than
MEASURED.

---

*Draft status: complete for the small-frame results (M0-E5), the registered full protocol (200
passages, 923 realised questions, three arms, one seed) and the residual routing test. The central
claim is the substrate-expressiveness thesis of §1 and §5. British spelling, ASCII quotes, em
dashes are literal characters.*
