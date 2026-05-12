# A3 — Quantitative error analysis

Source: `models/baseline_v1.pt` (the same artifact mined in A2).

## Per-split metrics

| Split | N | Accuracy | Macro F1 | Cog acc | Aff acc | Mot acc |
|---|---|---|---|---|---|---|
| TRAIN (Studies 1+1b, N=1218) | 1218 | 0.6511 | 0.3152 | 0.9472 (815) | 0.0280 (143) | 0.0654 (260) |
| EVAL  (held-out 100 from 1+1b) | 100 | 0.7000 | 0.3493 | 0.9437 (71) | 0.0000 (10) | 0.1579 (19) |
| TEST  (Study 3, N=1172) | 1172 | 0.7193 | 0.3265 | 0.9628 (860) | 0.0152 (132) | 0.0722 (180) |

## Confusion matrices (rows=true, cols=pred)

### TRAIN (Studies 1+1b, N=1218)

|         | pred Cog | pred Aff | pred Mot |
|---|---|---|---|
| true Cognitive | 772 | 4 | 39 |
| true Affective | 129 | 4 | 10 |
| true Motivational | 242 | 1 | 17 |

### EVAL  (held-out 100 from 1+1b)

|         | pred Cog | pred Aff | pred Mot |
|---|---|---|---|
| true Cognitive | 67 | 0 | 4 |
| true Affective | 10 | 0 | 0 |
| true Motivational | 16 | 0 | 3 |

### TEST  (Study 3, N=1172)

|         | pred Cog | pred Aff | pred Mot |
|---|---|---|---|
| true Cognitive | 828 | 1 | 31 |
| true Affective | 122 | 2 | 8 |
| true Motivational | 166 | 1 | 13 |

## Predicted-class share (argmax)

| Split | pred Cog | pred Aff | pred Mot |
|---|---|---|---|
| TRAIN | 0.938 | 0.007 | 0.054 |
| EVAL | 0.930 | 0.000 | 0.070 |
| TEST | 0.952 | 0.003 | 0.044 |

## True-class share (argmax of soft labels) — reference

| Split | true Cog | true Aff | true Mot |
|---|---|---|---|
| TRAIN | 0.669 | 0.117 | 0.213 |
| EVAL | 0.710 | 0.100 | 0.190 |
| TEST | 0.734 | 0.113 | 0.154 |

## Predicted-probability distribution summary (mean ± std)

| Split | mean P(cog) | mean P(aff) | mean P(mot) | std P(cog) | std P(aff) | std P(mot) |
|---|---|---|---|---|---|---|
| TRAIN | 0.356 | 0.316 | 0.328 | 0.010 | 0.010 | 0.011 |
| EVAL | 0.356 | 0.317 | 0.327 | 0.010 | 0.010 | 0.011 |
| TEST | 0.357 | 0.316 | 0.327 | 0.012 | 0.011 | 0.011 |

## Artifacts

- `outputs/a3_confusion_matrices.png` — train/eval/test confusion matrices
- `outputs/a3_prob_distributions.png` — predicted-prob histograms per (split, pred-class), colored by true class
- `outputs/a3_pred_share.png` — bar chart of predicted-class shares per split
