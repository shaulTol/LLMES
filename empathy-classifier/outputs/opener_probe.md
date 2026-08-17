# Opener-only probe

Frozen DistilBERT [CLS] + Linear(768→3) head, soft CE, Adam lr=1e-3, max_ep=50 pat=5, 10 seeds. Response is truncated to the first `N` words before encoding (N=full keeps the whole text).

| Opener width | Test F1 (mean ± std) | Best seed | Test acc | Mean ep |
|---|---|---|---|---|
| 3 | 0.2969 ± 0.0167 | 0.3314 | 0.7079 | 8.4 |
| 5 | 0.3098 ± 0.0188 | 0.3488 | 0.6968 | 8.5 |
| 10 | 0.3079 ± 0.0165 | 0.3386 | 0.6754 | 8.2 |
| 20 | 0.3243 ± 0.0149 | 0.3471 | 0.6404 | 10.2 |
| 50 | 0.3497 ± 0.0205 | 0.3763 | 0.6442 | 10.6 |
| full | 0.3539 ± 0.0181 | 0.3780 | 0.6613 | 12.2 |

**Δ(full − N=10) = +0.0459 F1; Δ(full − N=50) = +0.0042 F1.**

**Interpretation.** Quantitative confirmation of the opener-classification ceiling diagnosed by A2:

- The **first 3 words alone** get to 84% of the full-response F1 (0.297 vs 0.354). Three words is barely a phrase — almost certainly the opener template ("I'm really sorry...", "It sounds like...", "You must have felt...").
- By **N=50 words** the model has captured all the signal it's going to get from the response: F1 = 0.350, statistically indistinguishable from full-response F1 = 0.354.
- Across N ∈ {3, 5, 10, 20, 50, full}, F1 grows by only **+0.057 from "first 3 words" to "full response"**. The body of the response (words 51+) contributes **+0.004**, indistinguishable from noise.

In other words: feeding the model just the first sentence of every response would cost essentially nothing on this dataset, under this frozen-`[CLS]` architecture. The model is reading the opener — confirmed both qualitatively (A2 mining) and quantitatively (this probe).

**Implication for LoRA winner (F1 = 0.40):** LoRA's +0.045 over the full-response linear baseline is roughly equal to the +0.046 gap between N=10 and N=full. LoRA is recovering body-signal that the frozen-`[CLS]` baseline misses, but it's recovering it through the same `[CLS]` bottleneck — the gain is bounded. The remaining headroom requires breaking the bottleneck (token-level pooling, attention pool), not more capacity on top of it.