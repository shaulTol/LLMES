# A4 — Sensitivity analysis

Source linear head: `models/baseline_v1.pt`.  Clean reference: acc=0.7193, macroF1=0.3265.

## A4a — Feature-noise sweep on top-k correlated [CLS] dims

Top-10 dimensions by max|corr(dim, soft_label_k)| across the 3 label dimensions:
  [42, 390, 70, 261, 494, 353, 335, 168, 502, 333]  with |corr| = [0.20800000429153442, 0.20600000023841858, 0.20600000023841858, 0.20200000703334808, 0.1979999989271164, 0.1940000057220459, 0.19300000369548798, 0.19300000369548798, 0.1899999976158142, 0.1889999955892563]

### Test accuracy grid (mean over 5 noise realizations)

| top-k \ sigma | 0.0 | 0.5 | 1.0 | 2.0 | 5.0 |
|---|---|---|---|---|---|
| top-1 | 0.7193 | 0.7186 | 0.7195 | 0.7191 | 0.7123 |
| top-5 | 0.7193 | 0.7191 | 0.7183 | 0.7133 | 0.6968 |
| top-10 | 0.7193 | 0.7177 | 0.7137 | 0.7020 | 0.6379 |
| top-50 | 0.7193 | 0.7184 | 0.7056 | 0.6645 | 0.5404 |
| top-768 | 0.7193 | 0.6765 | 0.6111 | 0.5014 | 0.4102 |

### Test macro F1 grid (mean over 5 noise realizations)

| top-k \ sigma | 0.0 | 0.5 | 1.0 | 2.0 | 5.0 |
|---|---|---|---|---|---|
| top-1 | 0.3265 | 0.3272 | 0.3309 | 0.3319 | 0.3264 |
| top-5 | 0.3265 | 0.3309 | 0.3306 | 0.3326 | 0.3374 |
| top-10 | 0.3265 | 0.3303 | 0.3273 | 0.3413 | 0.3509 |
| top-50 | 0.3265 | 0.3378 | 0.3335 | 0.3481 | 0.3487 |
| top-768 | 0.3265 | 0.3436 | 0.3485 | 0.3415 | 0.3138 |

## A4b — Leave-one-out importance

Reference (full train, seed=0): test acc 0.6502, macro F1 0.3619.
N retrains: 1218.

### macro-F1 importance summary

- mean: -0.00341, std: 0.01925
- min:  -0.04338 (most-helpful single example to remove → biggest gain when removed)
- max:  +0.05582 (most-harmful single example to remove → biggest drop when removed)
- fraction with |importance| > 0.001: 0.960

### accuracy importance summary

- mean: +0.03646, std: 0.04576
- min:  -0.05034
- max:  +0.13396
- fraction with |importance| > 0.001: 0.987

Top-10 train examples whose removal **hurt** macro-F1 most (most valuable to keep):

| train_idx position | importance_f1 (= base − LOO_f1) |
|---|---|
| 972 (cache idx 1042) | +0.05582 |
| 971 (cache idx 1041) | +0.05292 |
| 943 (cache idx 1011) | +0.04603 |
| 927 (cache idx 995) | +0.04580 |
| 2 (cache idx 2) | +0.04571 |
| 1 (cache idx 1) | +0.04524 |
| 0 (cache idx 0) | +0.04489 |
| 28 (cache idx 29) | +0.04487 |
| 944 (cache idx 1012) | +0.04449 |
| 945 (cache idx 1013) | +0.04449 |

Top-10 train examples whose removal **helped** macro-F1 most (most harmful to keep):

| train_idx position | importance_f1 (= base − LOO_f1) |
|---|---|
| 229 (cache idx 248) | -0.04338 |
| 225 (cache idx 244) | -0.04244 |
| 228 (cache idx 247) | -0.04183 |
| 205 (cache idx 221) | -0.04004 |
| 197 (cache idx 213) | -0.03959 |
| 227 (cache idx 246) | -0.03917 |
| 211 (cache idx 227) | -0.03913 |
| 215 (cache idx 232) | -0.03912 |
| 224 (cache idx 243) | -0.03907 |
| 222 (cache idx 241) | -0.03860 |

## Artifacts

- `outputs/a4_noise_curves.png` — degradation curves under feature noise
- `outputs/a4_loo_histograms.png` — histograms of LOO importance on macro-F1 and accuracy
- `outputs/a4_sensitivity.npz` — raw arrays
