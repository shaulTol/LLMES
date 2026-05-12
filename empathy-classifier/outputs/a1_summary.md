# A1 — Label Permutation Null (N=100)

Train labels row-wise shuffled; eval and test untouched. Linear head retrained on cached frozen DistilBERT [CLS] embeddings, lr=1e-3, patience=5.

## Test accuracy (Study 3)
| | Mean | Std | Min | Max | 95% CI |
|---|---|---|---|---|---|
| Real labels  | 0.6466 | 0.0536 | 0.4735 | 0.7278 | [0.4940, 0.7214] |
| Permuted     | 0.6109 | 0.0860 | 0.3456 | 0.7184 | [0.4051, 0.7151] |
| Majority-class floor | 0.7338 | — | — | — | — |

## Macro F1
| | Mean | Std | Min | Max | 95% CI |
|---|---|---|---|---|---|
| Real labels | 0.3500 | 0.0224 | 0.2856 | 0.4021 | [0.3085, 0.3899] |
| Permuted    | 0.3085 | 0.0201 | 0.2382 | 0.3471 | [0.2570, 0.3444] |

## Statistical comparison
- P(null_acc ≥ mean_real_acc) = 0.410
- P(real_acc ≤ mean_null_acc) = 0.200
- Cohen's d (real vs null) on test acc: 0.498
