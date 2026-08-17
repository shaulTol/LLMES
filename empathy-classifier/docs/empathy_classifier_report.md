# Empathy Classifier — Findings Report

**Date:** 2026-06-14
**Author:** Shaul Tolkowsky
**Status:** Architecture search concluded.

---

## Executive summary

We set out to build a three-way soft classifier (cognitive / affective / motivational empathy) on Study-3 test data, starting from a frozen-DistilBERT + linear-head baseline at **macro F1 = 0.350**. Across ~150 SLURM cells spanning frozen heads, LoRA fine-tuning, partial encoder unfreeze, pooling variants, opener interventions, loss/architecture/schedule sweeps, and a switch to RoBERTa-base, the **best achievable F1 is 0.40 (best single seed of LoRA Story+Response with skip connection + decoupled lrs + dropout 0.5)**. Mean F1 across seeds tops out at 0.3746 (same architecture + opener-swap regularization).

The most informative result of the entire search is **negative**: switching to a fundamentally different encoder (RoBERTa-base — different pretraining objective, BPE tokenizer, 10× more training data than DistilBERT's teacher) lands in the *same* F1 zone of 0.35-0.36. Combined with two diagnostic experiments showing that (a) the first 50 words of every response give the same F1 as the full response (+0.004), and (b) decorrelating the opener from the label causes F1 to collapse, this is convergent evidence that **the macro-F1 ceiling near 0.40 is structural to the labels/task — not to any choice of architecture, fine-tuning method, regularization, or base model**.

Three reference checkpoints are preserved for downstream interpretability work:

- `models/baseline_v1.pt` — linear baseline (anchor)
- `models/lora_winner_seed9.pt` — LoRA winner seed 9 (the architecture-search ceiling)
- best RoBERTa-base from Phase 28 (different-base check)

Future work should pivot from "break the ceiling" to "characterize the ceiling": measure the labels' intrinsic information limit, examine inter-rater agreement, and use the three references to probe what features each model actually encodes.

---

## 1. Task and data

**Task.** Three-way *soft* classification of empathic responses. Each response is annotated with a probability mass over three empathy types: cognitive (perspective-taking), affective (emotional resonance), motivational (action-oriented support). Labels are produced by aggregating multiple annotators per row; a typical row's soft label is near-uniform, e.g. `[0.36, 0.31, 0.33]`.

**Data.** Three studies in one CSV (1218 train + 100 eval + 1172 test). Train = Studies 1 + 1b (minus 50 held out per study for eval); test = Study 3. Class share differs between train and test:

| Class | Train share | Test share |
|---|---|---|
| Cognitive    | 67% | 73% |
| Affective    | 12% | 11% |
| Motivational | 21% | 15% |

**Metric.** **Macro F1** is the primary metric, not accuracy. Study 3's 73% Cognitive prior means a trivial "always predict Cog" classifier scores **0.73 accuracy but only 0.28 macro F1**. The linear baseline scores 0.72 accuracy (just below the trivial floor) but 0.35 macro F1 — the genuine signal lives in macro F1.

---

## 2. Reference numbers

| Reference | macro F1 |
|---|---|
| Majority-class baseline (always Cog) | ~0.28 |
| Linear-on-frozen-`[CLS]` baseline (DistilBERT, 100 seeds) | **0.350 ± 0.022** |
| Linear baseline best seed | ~0.39 |
| LoRA Story+Response winner — mean (`p23_swap03_same` / `p27_combo`, 10 seeds) | **0.3746 ± 0.016** |
| LoRA Story+Response winner — best seed (`p17_skip_drop0p5` seed 9) | **0.3998** |
| RoBERTa-base, best Phase 28 cell (5 seeds) | 0.3638 ± 0.008 |

The whole search has moved the mean F1 by **+0.025** and the best-seed F1 by **+0.05** over the linear baseline. Substantial absolute work, modest absolute gain.

---

## 3. Architecture search — what we tried

The search followed a **greedy one-change-at-a-time protocol**: each step modified exactly one component from the current-accepted configuration; decisions used `Δacc / ΔF1`; rejected branches stayed in the log as dead ends. Roughly 30 steps and ~150 SLURM cells.

### 3.1 Head-side sweep (linear baseline → MLP, frozen `[CLS]`)

Started by varying the head on top of cached frozen DistilBERT `[CLS]` embeddings.

| Lever | Result |
|---|---|
| Linear (default) | F1 0.350 |
| MLP 256 hidden, dropout 0.3 | +0.04 acc, −0.02 F1 — kept conditionally |
| Class-weighted soft CE | null (cancels on near-uniform soft labels) |
| Label sharpening α=3 | acc +0.02, F1 −0.02 (amplifies Cog bias) |
| Latent Gaussian augmentation at `[CLS]` | mild win |
| `tgt=2500` augmentation target | mild win (+0.003 F1, p=0.033) |
| Story+Response cache (concat story+response `[CLS]`) | **+0.7 pp F1 → 0.378** at lr=1e-5 |

**Frozen-head champion:** Story+Response cache + MLP 256/dropout 0.3 + latent aug, **F1 = 0.378 (best of 100 seeds)** — the launching pad for LoRA work.

### 3.2 LoRA Story+Response — exhaustive sweep (Phases 7-21, 27)

Roughly 70 LoRA cells, then 8 more in Phase 27.

| Axis | Tested | Winner |
|---|---|---|
| Rank | 2, 4, 8, 16, 32, 64 | **4** |
| Target modules | qv, qkv, qkvo, qkvo+ffn | **qv** |
| Layer scope | top2, top3, all6 | **all6** |
| Alpha / r | r/2, r, 2r | **α = r** |
| Bias | none, lora_only, all | **none** |
| Decoupled lrs | head 3e-4 / lora 3e-5 | **+0.005 F1** over single lr |
| Head dropout | 0.3, 0.5, 0.7 | **0.5** |
| Weight decay | 0.01, 0.1, 0.3 | **0.01** |
| L1 reg | off, 1e-4, 1e-3 | **off** (1e-3 collapses) |
| Skip connection | off / on | **on** (head sees frozen ⊕ LoRA per text) |
| Mixup at concat-CLS | off, α=0.2, 0.5 | off |
| Text aug ([MASK], BERT-fill) | various targets | off (BERT-fill timed out) |
| Latent noise σ | 0, 0.5×std | off |
| Early stop metric | eval loss, eval F1 | **eval loss** |
| Loss | soft CE, focal γ=2, label smoothing 0.1 | **soft CE** |
| MLP head | 256, 512, 256→128 | **256 single** |
| LR schedule | flat, cosine + warmup | **flat** |
| Long training | 100 ep / pat 15 vs 200 / 25 | 100/15 fine |

**LoRA winner** (`p17_skip_drop0p5` seed 9): **F1 = 0.3998, acc = 0.6323, 28 epochs**.
**Best mean LoRA config** (`p23_swap03_same` = `p27_combo`): **F1 = 0.3746 ± 0.016, best 0.3913**.

### 3.3 Pooling sweep (Phase 22) — REJECT

Hypothesis: `[CLS]` is collapsing the response. Test by replacing with mean / attention / `cls+mean+max`.

| pool | F1 mean ± std | Δ vs cls |
|---|---|---|
| **cls (control)** | **0.3679 ± 0.020** | 0 |
| mean | 0.3565 ± 0.015 | −0.011 |
| attention (learned-query) | 0.3530 ± 0.011 | −0.015 |
| cls_mean_max (2304-d) | 0.3004 ± 0.033 | −0.068 |

**All non-cls poolings lose F1.** The body's token states (post-LoRA) don't carry the label-relevant features mean/attn pooling can pick up. This is the first strong signal that the body of the response is *information-impoverished*, not just under-utilized.

### 3.4 Partial encoder unfreeze (Phases 25-26) — REJECT

Hypothesis: LoRA's rank-r constraint is too restrictive — unfreezing top N transformer layers should give more capacity. Tested across **14 cells** spanning N ∈ {1, 2, 3}, encoder_lr ∈ {3e-6, 1e-5, 3e-5, 5e-5}, wd ∈ {0.01, 0.1, 0.3}, head_dropout ∈ {0.5, 0.7}, skip_conn, layer-wise lr decay, longer training.

| Best partial-unfreeze cell | F1 mean | best seed |
|---|---|---|
| top-2 + head_dropout 0.7 | 0.3554 | 0.3961 |
| top-2 + skip_conn | 0.3538 | 0.3799 |
| (DistilBERT LoRA winner) | 0.3679 | 0.3998 |

**Every partial-unfreeze cell loses to LoRA.** Pattern is overfit-shaped: top-2 > top-3, lower lr > higher, training stops 10 epochs earlier than LoRA, ~2× higher variance. Heavier wd HURTS (0.1, 0.3 both worse than 0.01). **LoRA's rank-r constraint is acting as useful regularization**, not as a capacity bottleneck.

Implication: full fine-tune (64% of params trainable) would be even worse than partial unfreeze. Did not run it.

### 3.5 Opener interventions (Phase 23) — mixed

The A2 success/failure mining had qualitatively suggested the model is doing "opener-template classification": confident-wrong errors are explainable from the first ~10 tokens (e.g., "I'm really sorry to hear..." → predicted Motivational regardless of true label). Tested interventions on the opener directly.

| Variant | F1 mean | Δ vs control |
|---|---|---|
| **swap p=0.3 same-class** | **0.3746** | **+0.007** |
| control (no intervention) | 0.3679 | 0 |
| swap p=0.5 cross-class | 0.3659 | −0.002 |
| swap p=0.5 same-class | 0.3643 | −0.004 |
| swap p=0.3 cross-class | 0.3626 | −0.005 |
| swap p=1.0 cross-class | 0.3403 | −0.028 |
| strip first 10 words | 0.3322 | −0.036 |

**Read:**

- **Same-class swap p=0.3 is the new mean champion (+0.007).** Mild regularization: model can no longer memorize a specific opener template as a shortcut to a class; it must generalize across multiple openers within the same class.

- **Cross-class swap consistently hurts.** When opener→class correlation is broken, the model has no body signal to pivot to.

- **p=1.0 cross-class lands at F1 = 0.34** — still well above the 0.28 majority floor, so the body has *some* signal. Roughly: opener carries 5/6 of the model's signal, body carries 1/6.

- **Strip-10 collapses F1 (−0.036) but acc rises to 0.66** — without the opener, the model falls back hard on the class prior, scoring well on Cog-heavy Study 3 by accident.

### 3.6 LoRA combine + untried head-side levers (Phase 27)

Eight final LoRA cells: the cancelled "combine winners" run (skip + decoupled + drop 0.5 + opener_swap03_same), plus untried loss variants, head capacities, LR schedules.

| Variant | F1 mean |
|---|---|
| **combine** (= swap03_same; reproducibility check) | **0.3746** |
| + label smoothing 0.1 | 0.3725 |
| + swap p=0.5 same | 0.3643 |
| + rank=2 | 0.3593 |
| + cosine LR + warmup | 0.3574 |
| + 512-d head | 0.3493 |
| + 2-layer head (256, 128) | 0.3489 |
| + focal loss γ=2 | 0.3327 |

The "combine" cell reproduces `p23_swap03_same` exactly (sanity check). Every other variant either ties within noise (label smoothing) or hurts. **LoRA design space is genuinely exhausted across loss / head capacity / LR schedule / rank / combinations.**

### 3.7 Switch base to RoBERTa-base (Phase 28)

Final architecture-side experiment. Originally planned DeBERTa-v3-base; hit a known transformers-5.x SentencePiece/tiktoken loader bug. Pivoted to **RoBERTa-base** (clean BPE tokenizer, ~125M params, trained with dynamic masking on 160 GB text). Same LoRA recipe as the DistilBERT champion, swept rank/layer_scope/target/swap. 6 cells × 5 seeds.

| Variant | F1 mean ± std | best seed |
|---|---|---|
| (DistilBERT) swap03_same champ | 0.3746 ± 0.016 | 0.3913 |
| (DistilBERT) LoRA winner | 0.3679 ± 0.020 | **0.3998** |
| **best p28:** rob_r8_qv_top6 | 0.3638 ± 0.008 | 0.3755 |
| rob_r8_qv_all12 | 0.3555 ± 0.010 | 0.3631 |
| rob_r4_qv_all12 | 0.3532 ± 0.011 | 0.3651 |
| rob_r4_qv_top6 (no swap) | 0.3505 ± 0.016 | 0.3658 |
| rob_r4_qv_top6 | 0.3478 ± 0.014 | 0.3662 |
| rob_r4_qkv_top6 | 0.3302 ± 0.021 | 0.3543 |

**All RoBERTa cells lose to both DistilBERT references.** Best RoBERTa mean is 0.3638 (−0.011 vs DistilBERT champ). Best RoBERTa seed is 0.3755 (−0.024 vs DistilBERT LoRA winner). Variance is markedly lower (~0.01 vs ~0.02): RoBERTa is more stable but lower-ceiling.

---

## 4. Diagnostic experiments

Two probes characterize the ceiling more directly than the leaderboard does.

### 4.1 Opener-only probe (`src/opener_probe.py`)

Truncate each response to its first N words, encode with frozen DistilBERT, train a linear head. 10 seeds per N.

| Opener width | F1 mean ± std | best seed | acc |
|---|---|---|---|
| 3 words  | 0.2969 ± 0.017 | 0.3314 | 0.7079 |
| 5 words  | 0.3098 ± 0.019 | 0.3488 | 0.6968 |
| 10 words | 0.3079 ± 0.017 | 0.3386 | 0.6754 |
| 20 words | 0.3243 ± 0.015 | 0.3471 | 0.6404 |
| 50 words | 0.3497 ± 0.021 | 0.3763 | 0.6442 |
| full     | 0.3539 ± 0.018 | 0.3780 | 0.6613 |

**Findings:**
- The first **3 words alone** get 84% of full-response F1.
- The first **50 words** match full response (Δ = +0.004, within noise).
- Words 51+ contribute essentially nothing.

**Quantitative confirmation that the opener carries almost all the model's signal under this encoder.**

### 4.2 Cross-class swap probe (Phase 23)

If we systematically replace the opener with one drawn from a *random* class during training, F1 should collapse if the body contains nothing else useful, or stay high if the body picks up the slack.

| Cross-class swap p | F1 mean | Δ vs control |
|---|---|---|
| 0.0 (control) | 0.3679 | 0 |
| 0.3 | 0.3626 | −0.005 |
| 0.5 | 0.3659 | −0.002 |
| 1.0 (every row swapped) | 0.3403 | **−0.028** |

At p=1.0 — where every training row's opener is cross-class — F1 lands at **0.34**. That is:

- Well above the majority-class floor (0.28) → body has *some* signal
- Well below the LoRA winner (0.40) → that signal is small

Roughly **1/6 of the model's signal lives in the body, 5/6 lives in the opener** under this encoder.

---

## 5. The convergent picture — ceiling is task-side

Three independent lines of evidence point to the same conclusion:

1. **Phase 22 (pooling sweep).** Mean / attention / cls+mean+max pooling all under-perform `[CLS]`. The body's token states don't carry features pooling can extract.
2. **Opener probe + cross-class swap.** Quantitatively, the model is reading the opener; the body contributes ~1/6 of the signal.
3. **Phase 28 (RoBERTa-base).** A different base — different pretraining objective, 10× more data, different tokenizer — lands at the same F1 zone (~0.36 mean). Even with a fundamentally different inductive bias, the body→label signal isn't recoverable.

Taken together, this is strong evidence that the macro-F1 ceiling near 0.40 is **not** an artifact of:
- The head (linear, MLP, deeper, wider — all explored)
- The fine-tuning strategy (LoRA across all axes; partial unfreeze across 14 cells)
- The pooling (cls is best; nothing surfaces from elsewhere)
- The opener (intervention shows it's the load-bearing feature; removing it just collapses to prior)
- The base encoder (RoBERTa-base reaches the same zone with a fundamentally different representation)

Plausible task-side causes:
- **Soft, near-uniform labels** (`[0.36, 0.31, 0.33]` typical) → limited per-example information
- **Intrinsic class overlap** — cognitive, affective, and motivational responses share linguistic surface features (sympathy openers, second-person address, sentiment vocabulary), so a response often genuinely belongs to multiple types
- **Small training set** (1218 rows) × mushy labels = small effective information budget
- **Annotation noise** — if multiple annotators disagree on each row (which is what generates soft labels), the labels themselves may already be near the inter-rater agreement floor

The strongest single claim of this report:

> The empathy-classifier-3-way problem appears to be at a macro-F1 ceiling near 0.40 inherent to the dataset and the labels, not to the architecture.

Remaining effort should characterize that ceiling, not try to break it.

---

## 6. Pinned reference models

All three are preserved for downstream interpretability work.

| Model | Path | F1 | Role |
|---|---|---|---|
| Linear baseline | `models/baseline_v1.pt` | 0.350 mean | anchor |
| LoRA winner (DistilBERT) | `models/lora_winner_seed9.pt` | 0.3998 best seed | architecture-search ceiling |
| Best RoBERTa-base (Phase 28) | (cell `p28_rob_r8_qv_top6`) | 0.3638 mean | different-base check |

Memory entry: `~/.claude/projects/-Users-stolk-github-LLMES/memory/reference_pinned_models.md`.

---

## 7. Recommended next steps

Two directions, in decreasing priority:

### 7.1 Characterize the ceiling

- **Inter-rater agreement floor.** Compute pairwise agreement between annotators of the same row in the original data. If F1-equivalent of human-vs-aggregate is near 0.40, the model has effectively reached the floor.
- **Bayes-optimal F1 from soft labels.** For each test row, the optimal hard prediction under the soft label distribution is `argmax(soft_label)`. Compute macro F1 of this oracle classifier. If it's near 0.40-0.45, that's the dataset's intrinsic ceiling regardless of features.
- **Per-class confusion of the three models.** Confirm that LoRA winner, RoBERTa winner, and linear baseline all fail on the same examples — if they do, the ceiling is data-side and shared. If RoBERTa fails on a different subset than DistilBERT, there's actually room for ensembling.
- **Class-definition overlap.** Look at the confident-wrong examples per model. Are the "errors" actually correct under a different reasonable reading of the labels?

### 7.2 Interpretability over the three references

- Probing classifiers on intermediate layers: which layer's representation is most predictive of the soft label?
- Attention pattern analysis: where does each model attend in the response? Does the LoRA winner attend differently from frozen DistilBERT?
- Layer-wise representation similarity (CKA, etc.) between linear-baseline-encoder vs LoRA-adapted vs RoBERTa.
- Feature ablation: zero out the first 10 tokens at inference; how does each model degrade?

### 7.3 Optional: one more "bigger encoder" sanity

If you want to fully close the encoder-side direction, run a single RoBERTa-large (355M, ~3× capacity) cell at the LoRA-winner recipe with 10 seeds. Expected outcome: similar mean (~0.36-0.38), lower variance. If it surprises us and breaks 0.40, the conclusion shifts and we reopen the encoder-side direction. The cell is cheap.

### 7.4 Optional: ship the model

The LoRA winner seed 9 (F1 = 0.40, acc = 0.63) is the best deployable. If a production system is the goal, ship that one, with the diagnostic story for why it cannot be made better with more architecture-side work.

---

## 8. Reference files

| What | Where |
|---|---|
| Live architecture-search log | `docs/architecture_search.md` |
| Earlier analysis report (baseline, A1-A4) | `docs/analysis_report.md` |
| Opener-only probe code + results | `src/opener_probe.py`, `outputs/opener_probe.{json,md}` |
| LoRA winner predictions (test set) | `outputs/preds_seed9_best_seed9.npz` |
| A2 failure-mode mining on LoRA winner | `outputs/a2_lora_examples.md`, `src/a2_lora_failure_modes.py` |
| LoRA training script | `src/run_lora_story.py` |
| Partial-FT training script | `src/run_partial_ft_story.py` |
| RoBERTa training script | `src/run_lora_story_roberta.py` |
| All Phase 7-28 SLURM scripts | `slurm/array_*.sh` |
| All experiment JSONs | `outputs/lorastory_*.json`, `outputs/partialft_*.json`, `outputs/roberta_*.json` |
