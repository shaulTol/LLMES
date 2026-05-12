# Empathy Classifier — Analysis & Improvement Plan

Reference baseline (Step 0): frozen DistilBERT + 768→3 linear head, lr=1e-3, MPS, soft cross-entropy. Test accuracy on Study 3 = **71.93%**, but the trivial majority-class floor is **73.4%**. Model collapses to predicting Cognitive. **Floor to beat: 73.4%, not 71.9%.**

---

## Step 0 — Precompute frozen [CLS] embeddings (foundation)

DistilBERT is frozen, so its outputs are deterministic per input. Cache them once → all baseline-style retraining (linear head only) becomes seconds instead of minutes. Required to make 100× label permutation tractable.

Output: `data/processed/cls_embeddings_distilbert.npz` (~2490 × 768 float32 ≈ 7 MB).

**Decisions:**
- Approve caching? (D0.1)
- Also cache mean-pooled embedding as a second feature view, or [CLS] only? (D0.2)
- Track CSV + cached embeddings in git, or keep gitignored as today? (D0.3)

---

## Part A — Exploratory & Model Analysis

### A1. Null distribution via 100× label permutation
Shuffle training labels (only training), retrain linear head on cached embeddings, evaluate on the real Study 3 test set. Repeat 100×. Report distribution of test accuracy + macro-F1.

Goal: confirm/refute that 71.9% is meaningfully above pure noise. Likely answer: the null centers near 73.4% (majority class), and our baseline is *inside* that null — which would formally show the linear baseline has near-zero signal.

**Decisions:**
- Shuffle train labels only (eval/test untouched) or also permute eval? (D1.1)
- Use a within-class shuffle (preserve class balance) or full uniform shuffle? (D1.2)

### A2. Success / failure example mining
Pick confident-correct and confident-wrong predictions on Study 3. Look at text patterns.

**Decisions:**
- N examples per cell (e.g. 5 confident-correct + 5 confident-wrong per class = 30 total)? (D2.1)
- Pre-register hypothesized text features to look for (response length, "I'm sorry / I understand" markers, advice-giving verbs, second-person pronouns, etc.)? (D2.2)

### A3. Quantitative error analysis
- Per-class accuracy + macro-F1 + confusion matrix on **both train and test**
- Are train-error patterns different from test-error patterns?
- Predicted probability distribution per class (histograms)

**Decisions:**
- Save figures to `outputs/`? (D3.1)
- Compare prob distributions against the Step-A1 permuted runs as a sanity check? (D3.2)

### A4. Sensitivity analysis
Two sub-analyses:
- **Feature-noise**: add Gaussian noise (σ ∈ {0.01, 0.1, 0.5, 1.0}) to test-time [CLS] embeddings → degradation curve.
- **Observation importance**: leave-one-out (1218 retrains; tractable thanks to cached embeddings) OR influence-via-gradient on training points.

**Decisions:**
- Full LOO (~minutes with cached embeddings) or gradient-based saliency? (D4.1)
- Noise the whole [CLS] or only top-k dims by correlation with target? (D4.2)

---

## Part B — Improved model

### B1. Pick the architectural change
Candidates (ranked by expected payoff):
1. **Unfreeze encoder** (full fine-tune, lower lr, encoder ~2e-5 and head ~1e-3) — biggest single win
2. Class weights / focal loss
3. Mean-pool over tokens instead of [CLS]
4. Stronger encoder (sentence-transformers `all-mpnet-base-v2`, RoBERTa-base)
5. LLM zero/few-shot

**Decision:** Which single change to commit to for the improved model? (D5.1)

### B2. Two feature sets
1. **Response only** (baseline-style)
2. **Story + Response tuple** — concatenate as `[STORY] <story> [RESPONSE] <response>`; classify the tuple

Both trained with the chosen architecture from B1.

**Decisions:**
- Concatenation format / separator tokens? (D6.1)
- Max length budget (responses often long; stories also non-trivial)? (D6.2)
- Handle "No consent to share" rows in Story column (~present in data) — drop, mask, or keep? (D6.3)

### B3. Evaluation & comparison
Report accuracy + macro-F1 + per-class + confusion for: baseline / B2-set1 / B2-set2. Also report on the Part A1 null benchmark to argue the gain is real.

### B4. Re-apply one Part A analysis to the improved model
**Decision:** Which Part A analysis (A1 null, A3 error patterns, or A4 sensitivity)? (D7.1)

---

## Decisions log (filled as we go)

| ID | Decision | Choice | Date |
|---|---|---|---|
| D0.1 | Cache embeddings? | Yes, cache to `data/processed/` ([CLS] only, frozen DistilBERT) | 2026-05-11 |
| D0.2 | [CLS] only or also mean-pool? | [CLS] only — can add mean-pool later in 30 sec if needed | 2026-05-11 |
| D0.3 | Track data files in git? | Keep gitignored; add `data/processed/*` to .gitignore alongside `data/raw/*` | 2026-05-11 |
| D1.0 | Seed-dependence handling | N=100 seeds for BOTH real-label baseline and permuted-label null; compare distributions | 2026-05-11 |
| D1.1 | Shuffle scope (train-only vs train+eval)? | Train labels only (eval/test untouched) | 2026-05-11 |
| D1.2 | Shuffle type (within-class vs uniform)? | Row-wise shuffle of label triplets (uniform random reassignment of full (cog,aff,mot) vectors) | 2026-05-11 |
| D2.1 | N examples per success/fail cell? | _pending_ | |
| D2.2 | Pre-registered text features? | _pending_ | |
| D3.1 | Save A3 figures? | _pending_ | |
| D3.2 | Compare prob dists vs permuted runs? | _pending_ | |
| D4.1 | LOO vs gradient saliency? | _pending_ | |
| D4.2 | Noise [CLS] in full or top-k? | _pending_ | |
| D5.1 | Improved model architectural change? | _pending_ | |
| D6.1 | Story+Response format? | _pending_ | |
| D6.2 | Max length? | _pending_ | |
| D6.3 | "No consent to share" handling? | _pending_ | |
| D7.1 | Which Part A analysis to re-apply? | _pending_ | |
