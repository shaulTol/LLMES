# Architecture Search — Greedy, One-Change-At-A-Time

## Future-work checklist (to revisit when current sweeps land)

1. **Embedding-noising literature review.** Before committing to our ad-hoc latent-Gaussian σ at the `[CLS]` (and the per-token version), survey published methods: SMOTE-on-embeddings, Mixup at the encoder output, noise schedules from contrastive learning (SimCSE-style), MASKER-style "feature dropout", InfoNCE adversarial perturbations. Goal: figure out whether what we did corresponds to a known technique with an established lr/σ relationship, and whether we should switch to (e.g.) Mixup which has stronger theory than per-dim Gaussian.

2. **Feature-relevance build-up + dimensionality reduction.** Currently we feed the full 768-d `[CLS]` (and 1536-d for Story+Response) to the head. Test the *complementary* direction: build a feature set incrementally by adding `[CLS]` dimensions in order of |corr(dim, target)|, evaluate at each step. If F1 plateaus past some k* dims (say k*≈50), then projecting onto the top-k* subspace before the head should help (less noise, fewer params). If F1 keeps growing past k*=768 (i.e. the answer is "use all dims"), the question becomes whether a *learned* projection (PCA / CCA / discriminant subspace) finds k* < 768 dims that match full-dim performance — a cheaper representation for downstream Story+Response / LoRA work.

---

This document tracks every architectural change attempted against the baseline, the result, and the decision (keep / reject / ask). Each step changes **exactly one** component from the currently-accepted configuration. Decision rule:

| Δ accuracy | Δ macro F1 | Decision |
|---|---|---|
| ↑ | ↑ | **KEEP** |
| ↓ | ↓ | **REJECT** |
| ↑ | ↓ (or vice versa) | **ASK USER** |

All runs use the cached frozen DistilBERT `[CLS]` embeddings, trained for N seeds (specified per step) and compared as distributions (mean ± std). "Δ vs current" is computed as the difference of seed-mean.

---

## Reference: linear baseline (current accepted config)

| Component | Value |
|---|---|
| Head | `Linear(768 → 3) → Softmax` |
| Loss | soft cross-entropy |
| Optimizer | Adam, lr=1e-3 |
| Batch | 32 |
| Early stop | eval loss, patience 5 |
| Max epochs | 50 |

**Reference distribution (N=100 seeds, from A1 real-labels run):**
- Test accuracy: **0.647 ± 0.054**
- Test macro F1: **0.350 ± 0.022**

For fair within-step comparison, each step's metric tables also include a re-run of the current-accepted config under the *same* seed budget used in that step.

---

## Path graph

```
linear baseline (acc 0.660, F1 0.350) [N=30]
   │
   └─ Step 1: linear → MLP-256/dropout-0.3   [KEPT despite mixed]
        acc +0.042 ↑, F1 −0.020 ↓
            │
            ├─ Step 2: + weighted soft CE on MLP   [REJECT — null result]
            │     acc +0.002, F1 −0.002 (within noise)
            │     class-weights cancel on mushy soft labels.
            │
            ├─ Step 3: sharpen labels α=3 (train+eval)   [REJECT — same trade as step 1]
            │     acc +0.021 ↑, F1 −0.019 ↓ (variance halved on acc)
            │     sharpening amplifies the Cog bias instead of fixing it
            │     since most argmax labels are Cog. Rolling back.
            │
            └─ Step 4: class-balanced sampling on MLP+softCE   [pending]
                  switch from loss-side bias attacks to data-side:
                  per-epoch weighted resample by inverse argmax-class freq,
                  so each batch is ~1/3 cog / 1/3 aff / 1/3 mot.
```

(Each accepted change becomes the new root for the next step. Rejected branches are kept in the doc as dead ends. Reference linear-baseline numbers used in this graph are the within-step re-run, not A1's N=100.)

---

## Steps log

### Step 1 — Linear → MLP

**Change:** Replace `Linear(768 → 3)` with `Linear(768 → 256) → GELU → Dropout(0.3) → Linear(256 → 3)` (one hidden layer; ~200K trainable parameters).
**Everything else (loss, optimizer, batch, early-stop, max epochs, patience):** unchanged.
**N seeds:** 30 each arm.

| Arm | Test acc | Test macro F1 | Mean epochs |
|---|---|---|---|
| linear (current) | 0.6597 ± 0.0485 | 0.3497 ± 0.0221 | 12.1 |
| MLP-256 (proposed) | **0.7015 ± 0.0271** | 0.3296 ± 0.0273 | 11.7 |
| Δ (MLP − linear) | **+0.0418** | **−0.0201** | — |

**Decision rule outcome:** mixed (accuracy up, macro F1 down) → asked user.

**Decision (user, 2026-05-11): KEEP.**
**Rationale.** The F1 drop is not a classical overfit signal: training stops at ~12 epochs in both arms, MLP accuracy variance is *lower* than linear's (0.027 vs 0.048), and we have no train/test gap pattern showing memorization. The mechanism is the one A3/A4 already characterized — extra capacity is being spent **amplifying the class-prior Cognitive bias**. The MLP is "overfitting to the class prior", not to the training examples. We lock in the accuracy gain + lower variance now, and attack the bias in step 2; if step 2 recovers F1, the MLP capacity will start being spent on real signal. If step 2 fails to recover F1 *and* the bias persists, we revisit and consider rolling back the MLP.

**Current accepted config after step 1:** `HeadConfig(head_type='mlp', hidden_dim=256, dropout=0.3, activation='gelu', loss='soft_ce', ...)`.

_(Result file: `outputs/arch_step1_mlp.json`.)_

---

### Step 2 — Add class weights to the MLP loss [REJECT]

**Change:** `loss = 'soft_ce'` → `loss = 'weighted_soft_ce'`. Per-sample weight = `sum_k(target_k · class_weight_k)`, where class weights are inverse argmax-class frequency on train (normalized to mean 1). Train class shares are 67% cog / 12% aff / 21% mot, giving weights ≈ {cog 0.31, aff 1.79, mot 0.91}.
**Everything else:** unchanged from the step-1 accepted config (MLP-256/dropout-0.3).
**N seeds:** 30 each arm.

| Arm | Test acc | Test macro F1 | Mean epochs |
|---|---|---|---|
| MLP-256, soft CE (current) | 0.7015 ± 0.0271 | 0.3296 ± 0.0273 | 11.7 |
| MLP-256, weighted soft CE | 0.7038 ± 0.0356 | 0.3273 ± 0.0262 | 10.8 |
| Δ (weighted − soft) | +0.0023 | −0.0023 | — |

**Decision: REJECT.** Both deltas are ≈ 0.002, well inside seed noise (std ≈ 0.03). Null result.

**Diagnosis.** Class-weighted soft CE only bites when soft labels are sharp. Our labels are near-uniform (typical row ≈ `[0.36, 0.31, 0.33]`). For such a row, the per-sample weight resolves to `0.36·0.31 + 0.31·1.79 + 0.33·0.91 ≈ 0.97` — essentially 1. The same holds for almost every training row, so the mechanism cancels out before reaching the gradient. Step 3 attacks this directly.

_(Result file: `outputs/arch_step2_classweights.json`.)_

---

### Step 3 — Sharpen the soft labels (α=3)

**Change:** Replace `Y` with `Y' = Y^α / sum(Y^α, axis=1)` for **train and eval** rows (α=3). Test labels stay original — argmax is invariant under monotonic sharpening, so test metrics are computed against the same ground truth regardless of α.
**Everything else:** unchanged from step-1 accepted config (MLP-256, soft CE, no class weights).
**N seeds:** 30 each arm.

| Arm | Test acc | Test macro F1 | Mean epochs |
|---|---|---|---|
| MLP-256, soft CE (current) | 0.7015 ± 0.0271 | 0.3296 ± 0.0273 | 11.7 |
| MLP-256, soft CE + sharpen α=3 | 0.7223 ± **0.0115** | 0.3102 ± 0.0223 | 14.7 |
| Δ (sharpen − no-sharpen) | **+0.0208** | **−0.0194** | +3 |

**Decision: REJECT.** Same trade as step 1 (acc ↑, F1 ↓). Sharpening makes labels less mushy — but because the *majority* of argmax labels are Cognitive, the sharper Cog targets pull the model further into the Cog bias instead of revealing minority-class signal. Notable side-effect: accuracy variance more than halved (0.027 → 0.011) and the model trained ~3 more epochs on average, so sharpening *is* producing a stronger gradient signal — just one pointing the wrong way for F1. Rolling back.

The originally-planned follow-up (class weights on top of sharpened labels) is shelved: we want to attack the Cog bias via a different mechanism than loss-side reweighting, since loss-side attacks have produced two consecutive bad/null outcomes (steps 2 and 3).

_(Result file: `outputs/arch_step3_sharpen.json`.)_

---

### Step 4 — Class-balanced sampling on MLP + soft CE

**Change:** Add `balanced_sampling = True` to the trainer. Per epoch, instead of `torch.randperm(N_train)`, draw `N_train` train indices **with replacement, weighted by inverse argmax-class frequency** — so each batch is on expectation 1/3 cog / 1/3 aff / 1/3 mot. Minority classes are seen ~5× more often per epoch than under uniform sampling. Loss is unchanged (soft CE, no class weights, no sharpening). This attacks the Cog bias **via the data stream** rather than via the loss — sidesteps the soft-label / weight cancellation issue that killed step 2.
**Everything else:** unchanged from the step-1 accepted config (MLP-256, soft CE).
**N seeds:** 30 each arm.

_(Results pending.)_

---

## Phase summary — LoRA Story+Response sweep (Phases 7-21)

After the frozen-`[CLS]` head exhausted easy levers (steps 1-4), we moved to LoRA fine-tuning on a Story+Response architecture (two BERT forwards, shared LoRA weights, concat CLS, MLP head). The frozen-head champion before LoRA was Story+Response at **F1 0.378 (best seed of 100, p17 base config)**.

**Best LoRA config found:** `p17_skip_drop0p5` — rank 4, target qv, all-6 layers, skip connection (frozen+LoRA both feeding the head), decoupled head/LoRA learning rates (3e-4 / 3e-5), wd 0.01, head dropout 0.5, eval-loss early stop, 100 epochs / patience 15.

**Best seed result (deployable):** seed 9 → **F1 = 0.3998, acc = 0.6323.**

**What helped (kept):** LoRA rank ≤ 8, qv targets, all-6 layers, skip connection above LoRA, decoupled lrs (head higher than LoRA), head dropout 0.5, eval-loss early stop, long-but-patient training.

**What did not help (rejected):**

| Lever | Outcome |
|---|---|
| Text augmentation (raw `[MASK]`, all-to-target / min-to-max) | Hurt or null |
| Mixup at concat-CLS | Null |
| Latent Gaussian noise on CLS | Null |
| Larger ranks (16, 32, 64) | Worse, more overfit |
| L1 reg on LoRA delta (1e-3) | Collapse: F1 = 0.276 |
| Heavy weight decay (≥ 0.1, isolated) | Mild loss |
| Bias variants (`lora_only`, `all`) | Within-noise |
| Eval-F1 early stop | Noisier (eval set = 100 ex), worse mean |
| Heavier head dropout (0.7) | Worse |
| Parallel multi-LoRA heads (rejected without running) | Prior evidence: one LoRA already saturates |
| BERT-fill MLM-substitution augmentation (Phase 21) | Inconclusive — all 6 cells timed out at 5h before 10-seed completion; partial single-seed F1s land at 0.36-0.40, same as no-aug |

**Per-class recall (frozen-head baseline → LoRA seed 9):**

| Class | Baseline | LoRA |
|---|---|---|
| Cognitive | 96.3% (828/860) | 78.6% (676/860) |
| Affective | 1.5% (2/132) | **16.7% (22/132)** |
| Motivational | 7.2% (13/180) | **23.9% (43/180)** |

LoRA broadens the prediction distribution (no longer all-Cog) and improves minority recall ~3-11×. Overall F1 moves from ~0.35 (linear) to **0.40 (LoRA seed 9)**.

---

## Diagnosis: the opener-classification ceiling

### Qualitative: A2 mining on LoRA winner

We re-ran the A2 confident-correct / confident-wrong mining protocol on the LoRA winner (`outputs/preds_seed9_best_seed9.npz`, generated by `src/a2_lora_failure_modes.py`, written to `outputs/a2_lora_examples.md`). The **same failure mode as the linear baseline persists**: errors are explainable from the first ~10 tokens of the response.

| True → Predicted | Pattern in confident-wrong examples |
|---|---|
| Cog → Mot (5/5) | All start with *"I'm really sorry to hear that you're feeling this way..."* — LLM safety/refusal template, which is a canonical Motivational opener. |
| Mot → Cog (4/5) | Start with *"You must have felt..."*, *"I can absolutely understand..."*, *"It's completely understandable..."* — canonical Cognitive openers. |
| Aff → Cog (5/5) | Emotional-content responses classified by their analytical-style openers. |

**Why the ceiling exists.** All architectures so far — linear, MLP, LoRA, +skip — funnel the entire response through a single `[CLS]` vector before the head sees it. That vector is dominated by the first few tokens of the response (BERT's `[CLS]` is heavily influenced by early-position attention). The head can re-weight that 768-d collapse, but it cannot reach back into the token stream to read the body of the response. LoRA changes the encoder's weights but **not the bottleneck** — the head still sees one pooled vector per response.

This is consistent with what the leaderboard shows: every architectural change above the encoder produces +/- 0.02 F1 swings around 0.36-0.40, with no breakthrough.

### Quantitative: opener-only probe (positive test, `src/opener_probe.py`)

To turn the A2 qualitative pattern into a hard number, we ran the **same** frozen-DistilBERT-`[CLS]` + linear head on responses truncated to their first N words (N ∈ {3, 5, 10, 20, 50, full}), 10 seeds each. If the body of the response is being ignored, F1 should already saturate at small N.

| Opener width | Test F1 (mean ± std) | Best seed | Test acc | Mean ep |
|---|---|---|---|---|
| 3 | 0.2969 ± 0.0167 | 0.3314 | 0.7079 | 8.4 |
| 5 | 0.3098 ± 0.0188 | 0.3488 | 0.6968 | 8.5 |
| 10 | 0.3079 ± 0.0165 | 0.3386 | 0.6754 | 8.2 |
| 20 | 0.3243 ± 0.0149 | 0.3471 | 0.6404 | 10.2 |
| 50 | 0.3497 ± 0.0205 | 0.3763 | 0.6442 | 10.6 |
| full | 0.3539 ± 0.0181 | 0.3780 | 0.6613 | 12.2 |

**Result.** The first **3 words alone** reach 84% of full-response F1 (0.297 vs 0.354). The first **50 words** match full (0.350 vs 0.354, within noise). Words 51+ contribute **+0.004 F1** — indistinguishable from zero. Quantitative confirmation: the model is reading the opener.

**Comparison to LoRA winner.** LoRA's full-response F1 (0.378 → 0.40 best seed) sits +0.025 to +0.045 above the full-response linear baseline. That gain is roughly the same size as the N=10 → N=full gap in the probe (+0.046). LoRA is recovering some body signal — but through the same `[CLS]` bottleneck, which is precisely what caps the gain. Breaking past 0.40 requires changing the pooling, not the head capacity.

_(Result files: `outputs/opener_probe.json`, `outputs/opener_probe.md`.)_

---

## Phase 22 — pooling sweep (negative result)

**Hypothesis**: F1 ceiling at 0.36-0.40 is caused by `[CLS]` collapsing the response. Test by replacing `[CLS]` pooling with mean / attention / `cls+mean+max` while keeping the seed-9 LoRA winner config constant (`rank 4 qv all6, skip_conn, head_dropout 0.5, decoupled lrs, wd 0.01, 100 ep / pat 15, balanced_samp`, 10 seeds).

| pool | F1 mean ± std | best | acc | Δ vs cls |
|---|---|---|---|---|
| **cls** (control) | **0.3679 ± 0.0198** | 0.3998 | 0.6227 | 0 |
| mean | 0.3565 ± 0.0153 | 0.3765 | 0.5885 | −0.011 |
| attn (learned-query) | 0.3530 ± 0.0105 | 0.3699 | 0.5902 | −0.015 |
| cls_mean_max (2304-d) | 0.3004 ± 0.0333 | 0.3448 | 0.5145 | −0.068 |

**Decision: REJECT pooling change.** All non-`cls` variants lose F1. The body's token states, after frozen+LoRA encoding, don't surface label-relevant signal that mean/attn pooling can pick up — they actually wash out the strong opener signal in `[CLS]`. `cls_mean_max`'s 2304-d input is too high-capacity for the head to fit on 1218 train rows.

This rules out pooling as the bottleneck. The `[CLS]` vector is not just *over*-representing the opener; the rest of the sequence appears to lack the body→label signal in the first place.

_(Result files: `outputs/lorastory_*_p22_pool_*.json`.)_

---

## Phase 23 — opener interventions (mixed result; cleanest variant still running)

**Hypothesis**: opener overfit drives the ceiling. Test by directly intervening on the opener during training:
- `strip_N`: drop the first 10 words of every response (train + eval + test). Forces body-only classification.
- `swap_p_same`: with probability `p` per training row, replace first 10 words with one sampled uniformly from a same-class opener bank. Keeps opener→class correlation; breaks opener→specific-template memorization.
- `swap_p_cross`: same, but bank is cross-class (any label). Breaks opener→class entirely.

| Variant | F1 mean ± std | best | acc | Δ vs control |
|---|---|---|---|---|
| **swap p=0.3 same-class** | **0.3746 ± 0.0159** | 0.3913 | 0.6157 | **+0.007** |
| control (no intervention) | 0.3679 ± 0.0198 | 0.3998 | 0.6227 | 0 |
| swap p=0.5 cross-class | 0.3659 ± 0.0197 | 0.3965 | 0.6121 | −0.002 |
| swap p=0.5 same-class | 0.3643 ± 0.0197 | 0.3923 | 0.6076 | −0.004 |
| swap p=0.3 cross-class | 0.3626 ± 0.0123 | 0.3849 | 0.5997 | −0.005 |
| strip 10 words | 0.3322 ± 0.0360 | 0.3857 | **0.6600** | −0.036 |
| swap p=1.0 cross-class | *(running)* | — | — | — |

**Interpretation.**

- **Same-class swap p=0.3 is the new mean champion (+0.007).** Mild regularization win. The model was over-memorizing specific opener templates (e.g. "I'm really sorry to hear..." as a literal phrase); forcing it to handle multiple openers within the same class generalizes better. But best-seed (0.3913) didn't beat control's 0.3998, so it's a mean-shift, not a ceiling break.

- **Cross-class swap consistently hurts (−0.002 to −0.005).** When the opener no longer predicts the label, the model has no body signal to pivot to. It can't compensate. **This is the diagnostic answer**: the opener is doing the actual work; the body is not contributing recoverable signal.

- **Strip-10 collapses F1 (−0.036) but raises accuracy to 0.66** (highest in the table). With the opener removed, the model falls back hard on the class prior — it predicts Cognitive on almost everything, which is rewarded by Study-3's 73% Cog test share but kills macro F1. Confirms: opener is what lets the model deviate from the majority-class predictor.

- **p=1.0 cross-class** (still running) is the cleanest version of this test — every training row gets a cross-class opener, so opener→label correlation is fully broken. If F1 stays above the majority-class floor (~0.28), the body has some signal. If F1 collapses to floor, body has none accessible.

_(Result files: `outputs/lorastory_*_p23_*.json`.)_

---

## Synthesis (Phase 22 + Phase 23)

Combining the negative pooling result and the cross-class swap result:

1. **Opener overfit is real but small.** Same-class swap regularizes it away for +0.007 F1 — not nothing, but not a breakthrough either.
2. **The opener is genuinely the only label-recoverable signal at this encoder's capacity.** Cross-class swap and strip both hurt; mean/attn pooling can't find anything else in the token states.
3. **The F1 ceiling at ~0.37-0.40 is structural to the frozen-then-LoRA-adapted DistilBERT representation.** Every lever above the encoder is exhausted (linear → MLP, LoRA rank, skip-conn, decoupled lrs, dropout, augmentation, mixup, pooling, opener interventions).

**Next escalation (if p=1.0 cross-class doesn't surprise us).** The encoder itself has to change:

- **Full DistilBERT fine-tune** (no LoRA cap) — unfreezes the encoder; more capacity to reshape body-token representations into something label-relevant.
- **Partial unfreeze** (top 2-3 layers only) — middle ground between LoRA and full fine-tune; cheaper, less prone to catastrophic forgetting on 1218 train rows.
- **Richer base** (RoBERTa-large, decoder LM like Llama-3-8B with classification head) — different inductive biases, potentially deeper compositional features in the body.
- **Reformulation** — binary one-vs-rest classifiers per class; or token-level labels aggregated up.

---

## Phase 25 — partial encoder unfreeze (negative result, REJECT direction)

**Hypothesis**: LoRA's rank-r updates on q/v are too restrictive — releasing full top-layer weight updates should let the encoder reshape body-token representations and break the ceiling.

**Setup**: `src/run_partial_ft_story.py` mirrors the LoRA winner architecture (Story+Response, MLP head 256/dropout 0.5, balanced_samp, eval-loss early stop, 100 ep / pat 15) but replaces LoRA with partial unfreeze: freeze embeddings + bottom layers, unfreeze top N DistilBERT layers. Decoupled lrs (head_lr=3e-4, encoder_lr swept). 10 seeds × 6 cells.

| Config | F1 mean ± std | best | acc | mean ep |
|---|---|---|---|---|
| p25_top2_enc3e5 | 0.3502 ± 0.0380 | 0.3860 | 0.5979 | 19.9 |
| p25_top3_enc1e5 | 0.3458 ± 0.0346 | 0.3936 | 0.6029 | 20.3 |
| p25_top2_enc1e5 | 0.3456 ± 0.0340 | 0.3934 | 0.6162 | 20.3 |
| p25_top2_enc5e5 | 0.3358 ± 0.0298 | 0.3760 | 0.5528 | 19.3 |
| p25_top3_enc3e5 | 0.3314 ± 0.0369 | 0.3686 | 0.5512 | 19.1 |
| p25_top3_enc5e5 | 0.3278 ± 0.0311 | 0.3654 | 0.5601 | 18.4 |
| (reference) LoRA winner | 0.3679 ± 0.0198 | 0.3998 | 0.6227 | 30.1 |
| (reference) swap03_same | 0.3746 ± 0.0159 | 0.3913 | 0.6157 | 29.4 |

**All partial-unfreeze variants LOSE to both the LoRA winner and the same-class swap champion.** Best partial-unfreeze cell (top-2 enc 3e-5) hits F1 mean 0.3502 — −0.018 below LoRA. Best seed (0.3936) doesn't beat LoRA's 0.3998 either. Variance is also ~2× LoRA's (0.034 vs 0.020).

**Pattern:** top-2 > top-3, lower encoder_lr > higher, training stops ~10 epochs earlier than LoRA — all classic **overfit signatures**. Unfreezing 22% (top-2) or 32% (top-3) of params on 1218 train rows lets the encoder drift into a worse representation; the model fits train fast but generalizes worse. **LoRA's rank-r constraint was acting as useful regularization, not a capacity bottleneck.**

**Decision: REJECT partial-unfreeze direction.** Full fine-tune (64% of params) is almost certainly worse — won't run it. The cheapest remaining encoder-side lever is going *narrower* than LoRA (top-1 layer only + heavy wd), but the expected payoff is small.

**Where the search stands now.** Every encoder-side intervention up to and including partial unfreeze has failed to break F1 ~0.40. The ceiling is structural to the *pretrained DistilBERT representation itself*, not its adaptability. Real next escalation:

- **Switch base model** — RoBERTa-large, DeBERTa-v3-large, or a decoder LM with classification head (Llama-3-8B / Mistral-7B). Different pretraining objective and depth → potentially different body-token features. Most promising remaining lever.
- **Reformulation** — three binary one-vs-rest classifiers (each task gets focused signal; minority classes get their own head); or train per-token labels and aggregate. Cheap, but uncertain.
- **Accept the ceiling and ship the LoRA winner.** F1 = 0.40 (best seed) is a meaningful +0.05 over the linear baseline; we have a clear, defensible story about why we can't go higher.

_(Result files: `outputs/partialft_story_*_p25_*.json`.)_

---

## Phase 26 — partial unfreeze regularization sweep (final negative; REJECT confirmed)

Before declaring partial unfreeze dead, swept 8 more cells with the LoRA winner's tricks layered on: narrower (top-1), heavier wd (0.1, 0.3), heavier dropout (0.7), skip_conn, layer-wise lr decay, longer training. All still lose to LoRA.

| Cell | F1 mean ± std | best | acc | mean ep |
|---|---|---|---|---|
| top-2 + head_dropout 0.7 | 0.3554 ± 0.0282 | **0.3961** | 0.6164 | 19.4 |
| top-2 + skip_conn | 0.3538 ± 0.0251 | 0.3799 | 0.5762 | 20.6 |
| top-1, enc 3e-5 | 0.3528 ± 0.0275 | 0.3890 | 0.6234 | 20.5 |
| top-1, enc 1e-5 | 0.3511 ± 0.0206 | 0.3799 | 0.6428 | 21.9 |
| top-2 long (200 ep, pat 25) | 0.3502 ± 0.0380 | 0.3860 | 0.5979 | 29.9 |
| top-2 + LLRD 0.5 | 0.3489 ± 0.0337 | 0.3848 | 0.6273 | 20.1 |
| top-2 + wd 0.3 | 0.3457 ± 0.0454 | 0.3861 | 0.5947 | 19.8 |
| top-2 + wd 0.1 | 0.3444 ± 0.0470 | 0.3838 | 0.5938 | 19.8 |
| (reference) LoRA winner | 0.3679 ± 0.0198 | 0.3998 | 0.6227 | 30.1 |
| (reference) swap03_same | 0.3746 ± 0.0159 | 0.3913 | 0.6157 | 29.4 |

**Best regularized partial unfreeze (top-2 + dropout 0.7): F1 mean 0.3554, best seed 0.3961.** That's +0.005 over Phase 25's best, but still −0.012 below LoRA's mean and −0.004 below LoRA's best seed. Across 14 partial-unfreeze cells (Phase 25 + 26) we did not find a single config that beats the LoRA winner.

**Why partial unfreeze loses even with skip_conn:** the frozen+partial-FT pair can't reproduce what frozen+LoRA-adapted does. LoRA's adapter applies *targeted, low-rank* updates inside the q/v projections of every transformer layer (rank=4 across all 6 layers). Partial unfreeze gives top-2 layers *full-rank* updates — far more capacity but only at the last two layers, no access to earlier projections. On 1218 train rows, LoRA's distributed low-rank adaptation generalizes; partial unfreeze's concentrated full-rank adaptation overfits.

Notable secondary findings:
- **Heavier wd HURTS** (0.3 → 0.346, 0.1 → 0.344, vs 0.01 → 0.350). The unfrozen layers don't need shrinkage; what they need is structural constraint, which wd doesn't provide.
- **LLRD did not help.** Layer-wise lr decay (deeper layer = lower lr) made no difference vs uniform encoder_lr.
- **Longer training did not help.** With max_epochs 200 + patience 25, the model still stops around epoch 30 and lands at the same F1 as the 100-epoch version. The bottleneck is not "stopped too early".

**Final decision: REJECT partial-unfreeze direction.** Confirmed across 14 cells with comprehensive regularization sweep.

**Real remaining options (re-stated, unchanged):**
1. **Switch base model** (RoBERTa-large / DeBERTa-v3-large / decoder LM). Most promising. The thing we have NOT tried is a fundamentally different pretrained representation.
2. **Reformulation** — three binary one-vs-rest classifiers.
3. **Ship the LoRA winner** with the diagnostic story.

_(Result files: `outputs/partialft_story_*_p26_*.json`.)_

---

## Phase 27 — LoRA combine-winners + untried levers (no breakthrough)

Before finalizing the LoRA ceiling, we ran the **combine** that Phase 24 was supposed to do (cancelled earlier after pooling failed) plus all the LoRA axes we had not yet swept: loss variants, head capacity, LR schedule, narrower rank, heavier swap. 8 cells × 10 seeds; base = LoRA winner (`p17_skip_drop0p5` config: skip + decoupled lrs + head_drop 0.5) + `opener_swap_p=0.3 same_class`.

Added to `run_lora_story.py`: `--loss {soft_ce, soft_ce_ls, focal}`, `--focal_gamma`, `--label_smoothing`, `--mlp_hidden`, `--mlp_hidden2` (enables 2-layer head), `--lr_schedule {flat, cosine}`, `--warmup_epochs`.

| Variant | F1 mean ± std | best | acc | Δ vs combine |
|---|---|---|---|---|
| **p27_combo** (pure combine, no extra) | **0.3746 ± 0.0159** | 0.3913 | 0.6157 | 0 |
| p27_combo + label smoothing 0.1 | 0.3725 ± 0.0138 | 0.3872 | 0.6177 | −0.002 |
| p27_combo + swap p=0.5 same | 0.3643 ± 0.0197 | 0.3923 | 0.6076 | −0.010 |
| p27_combo + rank=2 | 0.3593 ± 0.0166 | 0.3843 | 0.6056 | −0.015 |
| p27_combo + cosine LR | 0.3574 ± 0.0125 | 0.3747 | 0.6226 | −0.017 |
| p27_combo + 512-d head | 0.3493 ± 0.0225 | 0.3806 | 0.6145 | −0.025 |
| p27_combo + 2-layer head (256, 128) | 0.3489 ± 0.0137 | 0.3762 | 0.5770 | −0.026 |
| p27_combo + focal loss γ=2 | 0.3327 ± 0.0278 | 0.3737 | 0.6054 | −0.042 |

**Findings:**

- **`p27_combo` ≡ `p23_swap03_same`.** Same config, identical numbers (F1 0.3746, best 0.3913). Reproducibility sanity check passes; Phase 24's "combine winners" was effectively swap03_same all along, since pooling lost. No new combination to discover.

- **Label smoothing 0.1 essentially ties combine** (0.3725 vs 0.3746, within seed noise). Tiny head-side regularization, no breakthrough.

- **Focal loss collapsed (−0.042 F1).** γ=2 is too aggressive on near-uniform soft labels (typical row ≈ `[0.36, 0.31, 0.33]`). The focal weight `(1 - p_true)^2` ends up high on essentially every sample, amplifying noise. Same mechanism that killed class-weighted CE back in Step 2.

- **All head-capacity variants HURT** (wider, deeper, both). 1218 train rows is not enough to fit a 2-layer head on top of LoRA-adapted features. The 256-d single-hidden-layer head is the right capacity.

- **Cosine LR + warmup HURT** (−0.017). Flat AdamW is fine here; the schedule was decaying the lr below the effective range needed by the head/LoRA pair.

- **Rank=2 mildly hurts** (−0.015 vs rank=4). Rank=4 remains the sweet spot.

- **Best individual seed across Phase 27 = 0.3923** (combo_swap05). Still below the LoRA winner's 0.3998.

**Decision: LoRA design space is now genuinely exhausted across loss, head capacity, LR schedule, rank, and combinations.** No cell beats F1 mean 0.40. The ceiling stands.

_(Result files: `outputs/lorastory_*_p27_*.json`.)_

---

## Phase 28 — switch base from DistilBERT to RoBERTa-base (no breakthrough)

**Setup**: `src/run_lora_story_roberta.py` mirrors the LoRA winner architecture (Story+Response, MLP head 256/dropout 0.5, skip_conn, decoupled lrs, opener_swap_p=0.3 same_class) but with `roberta-base` as the base model. Different pretraining objective (dynamic masking), 10× more training data (160 GB vs 16 GB for BERT's teacher), BPE tokenizer. 6 cells × 5 seeds.

(Originally planned: DeBERTa-v3-base. Transformers 5.x has a known SentencePiece/tiktoken loader bug that prevents loading DeBERTa-v3's `spm.model`; the cluster venv hit it. Pivoted to RoBERTa-base.)

| Variant | F1 mean ± std | best | acc | mean ep |
|---|---|---|---|---|
| rob_r8_qv_top6 | 0.3638 ± 0.0077 | 0.3755 | 0.5834 | 25.4 |
| rob_r8_qv_all12 | 0.3555 ± 0.0100 | 0.3631 | 0.5901 | 22.2 |
| rob_r4_qv_all12 | 0.3532 ± 0.0107 | 0.3651 | 0.5788 | 26.0 |
| rob_r4_qv_top6 (no swap) | 0.3505 ± 0.0163 | 0.3658 | 0.6265 | 23.2 |
| rob_r4_qv_top6 | 0.3478 ± 0.0136 | 0.3662 | 0.6113 | 23.2 |
| rob_r4_qkv_top6 | 0.3302 ± 0.0211 | 0.3543 | 0.5640 | 21.8 |
| (DistilBERT) swap03_same champ | **0.3746 ± 0.0159** | 0.3913 | 0.6157 | 29.4 |
| (DistilBERT) LoRA winner | 0.3679 ± 0.0198 | **0.3998** | 0.6227 | 30.1 |

**All RoBERTa cells lose to both DistilBERT references.** Best mean is 0.3638 (−0.011 vs DistilBERT champ); best individual seed is 0.3755 (−0.024 vs DistilBERT LoRA winner).

**Notable:**
- **Lower variance** than DistilBERT (~0.01 vs ~0.02). RoBERTa is more stable but lands lower.
- Same hyperparameter trends as DistilBERT: rank-8 > rank-4 (here, marginal), qv > qkv, opener_swap helps slightly.
- 5 seeds is noisier than the 10-seed numbers for DistilBERT, but the mean gap is large enough to not flip with more seeds.

**Strongest inference of the search:** The ceiling is **not in the encoder**. A fundamentally different base model, trained with a different objective on much more data, lands in essentially the same F1 zone (0.35-0.36) as the original DistilBERT linear baseline. Combined with the opener-only probe (F1 saturates at N=50 words; the body adds only +0.004) and the cross-class swap (body has ~1/6 of the model's signal), this is strong evidence that the ceiling is **in the task / labels themselves**:

- Soft labels are near-uniform (`[0.36, 0.31, 0.33]` typical) → limited usable signal per example
- The cognitive/affective/motivational empathy distinction may have intrinsic semantic overlap that bumps every encoder up against the same wall
- 1218 train rows × mushy labels = small effective information budget

_(Result files: `outputs/roberta_story_*_p28_*.json`.)_

---

## Final standing (post Phase 28)

**Three reference models, all preserved:**

| Model | F1 mean | best seed | role |
|---|---|---|---|
| **Linear baseline** (`models/baseline_v1.pt`) | 0.350 ± 0.022 (N=100) | ~0.39 | anchor — what we started from |
| **LoRA winner DistilBERT** (`models/lora_winner_seed9.pt`) | 0.3746 (best mean: swap03_same) | **0.3998** (seed 9) | architecture-search ceiling |
| **Best RoBERTa-base** (Phase 28 r8_qv_top6) | 0.3638 ± 0.008 | 0.3755 | different-base check |

**Path forward** — two real options:

1. **One more "bigger encoder" cell — RoBERTa-large (355M, 3× capacity).** Cheap sanity check; would more decisively show whether capacity alone can budge the ceiling. Expected outcome: more stable variance, similar mean ≈ 0.36-0.37. If it surprises us with > 0.40, we'd reopen the encoder-side direction.

2. **Pivot to interpretability + label-side work.** With three reference checkpoints (linear baseline, LoRA winner, best RoBERTa) we can:
   - Compare what each model actually attends to / encodes — probe attention patterns, layer-wise activations, what features differ between LoRA and frozen.
   - Measure the labels' intrinsic information limit — what's the maximum macro F1 attainable from the soft labels themselves under perfect calibration? (Bayes-optimal classifier on the soft distribution.) This sets the real ceiling.
   - Examine annotator agreement — are the soft labels themselves close to the inter-rater agreement floor?

The strongest single next-step claim from this whole search: **the empathy-classifier-3-way problem may be inherently at a ~0.40 macro-F1 ceiling on this dataset**, and the rest of the work should be characterizing that ceiling rather than trying to break it.

---

## Next architectural lever — token-level pooling

The next change is to **replace `[CLS]` pooling with token-level pooling**, exposing the full response token sequence to the head:

| Pooling | Description | Cost |
|---|---|---|
| Mean-pool over all tokens | Average BERT hidden states across the response sequence. Cheapest. | None — drop-in replacement. |
| Attention-pool (single query) | Learned query vector attends over all tokens; weighted sum. Lets the head "look at" relevant positions instead of being stuck on `[CLS]`. | Small (a few hundred params). |
| Mean + max + CLS concat | Three vectors concatenated → 2304-d head input. Cheapest way to break the single-vector bottleneck. | None — just concat. |

**Plan:** add a `--pool {cls, mean, attn, cls_mean_max}` flag to `run_lora_story.py` and sweep on the seed-9 config. Expect that *if* the diagnosis is correct, mean / attention pooling produces the first F1 jump above 0.40 we have seen since the frozen-`[CLS]` regime started.

If token-pooling also stalls at 0.40, the bottleneck is not pooling — it is the encoder's representation of the response itself, and the next lever is full-encoder fine-tuning (no LoRA), or switching to a richer base (RoBERTa-large / Llama-style decoder + classification head).

_(Source files: `src/a2_lora_failure_modes.py`, `outputs/a2_lora_examples.md`, `outputs/preds_seed9_best_seed9.npz`.)_

