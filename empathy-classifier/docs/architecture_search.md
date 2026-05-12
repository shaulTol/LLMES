# Architecture Search — Greedy, One-Change-At-A-Time

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

