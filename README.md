# Predicting the Dominant Empathy Type of a Written Response

### A Label-Side Ceiling for Three-Way Empathy Classification

Shaul Tolkowsky, Guy Shiff · 52056 Data Science Project

> This README is the project report in Markdown. The submitted PDF is compiled from
> `empathy-classifier/docs/final_paper.tex` and adds two figures (the exploratory
> panels and the confusion-matrix grid) plus four appendices. See
> [Building the report](#building-the-report).

---

## 1. Abstract

Psychological work on empathy has long treated it as several distinct capacities rather than one. A common division names three: cognitive empathy, the ability to take another person's perspective and understand what they are going through; affective empathy, the sharing of another person's emotional state; and motivational empathy, the wish to help and the readiness to act on it. A written reply to someone's personal disclosure can draw on any of these, and in practice usually draws on more than one at once.

Our data comes from a perception study in which participants read short disclosures and the replies written to them, then rated each reply on all three dimensions. Averaging those ratings across participants and normalising gives every reply a share of cognitive, affective and motivational empathy that sums to one. We use these shares as training targets and ask which type dominates a reply. The dataset holds 2490 replies across three studies. We train on Studies 1 and 1b, which give 1218 replies, and test on Study 3, which gives 1172 replies written for different prompts. Because Cognitive dominates 73% of the test replies, always answering Cognitive already scores 0.73 accuracy, so we report macro F1 throughout and treat accuracy as a sanity check.

We compare two models on two feature sets. The Semester A baseline freezes a DistilBERT encoder and trains a small head on its `[CLS]` vector. The Semester B final model adds LoRA adapters so the encoder itself can adapt. Each is run on the reply alone and on the reply concatenated with the disclosure it answers. The final model improves on the baseline under both feature sets, from 0.336 to 0.400 macro F1, against 0.282 for the majority-class floor. The two changes interact, since the disclosure helps only once the encoder can adapt to it. The gains stay small in absolute terms, and our analysis explains why: both models decide mainly from the opening phrase of a reply. Replacing that opening with one drawn from a different class costs the fine-tuned model 0.028 macro F1, roughly a third of the signal it holds above the floor. Applying ProxySPEX (Butler et al., 2025) recovers the same three-word opener features from two different encoders. A resampling estimate of the label-noise floor, 0.342 macro F1, sits just under our best model, which leads us to place the ceiling in the labels rather than in model capacity.

---

## 2. Exploratory analysis (Semester A)

The dataset contains 2490 responses across three studies. Each response pairs a *story* (the discloser's message) with a *response* (the empathic reply), and carries three annotator-derived scores that we normalise into a soft label over the three empathy types. Two properties of the data shaped every later modelling decision.

**The dominant type barely dominates.** A typical label looks like `[0.36, 0.31, 0.33]`. Participants rarely agreed that a reply was purely one type, and we read this as real rather than as noise. A reply that says "that sounds exhausting, and I'm here if you need anything" genuinely contains both cognitive and motivational empathy. It also caps how much any classifier can achieve, because the argmax of a near-tie is close to arbitrary.

**The class prior is skewed and it shifts.** Cognitive dominates in both splits, but more so in Study 3 (73%) than in Studies 1+1b (67%). This is why we do not report accuracy as the headline metric: always predicting Cognitive scores 0.73 accuracy but only 0.282 macro F1, and a model can raise its accuracy purely by leaning harder on the prior.

**The text is short and templated.** Replies run 45 to 110 words, generated under a small number of prompt conditions. Word-frequency and style analysis showed heavy reuse of a few opening formulas ("It sounds like…", "I'm really sorry to hear…"), with lexical diversity low and roughly equal across the three types. We noted at the time that a classifier might key on those openings. Section 5 shows that it did.

---

## 3. Training and evaluation process (Semester A)

**Splits.** Train = Studies 1 + 1b minus 50 held-out responses per study (1218 rows); validation = those 100 held-out rows; test = all of Study 3 (1172 rows). Testing on a different study rather than a random split is deliberate: it measures whether the model generalises to new prompts, which is the use case we care about. It also makes the numbers lower than a random split would produce.

**Objective.** Soft cross-entropy against the full label distribution, rather than cross-entropy against the argmax. Hard labels destroy performance (Section 7.1), so keeping the label distribution intact matters.

**Metric.** Macro F1 on the argmax of the predicted distribution. Macro rather than micro because the minority classes (Affective 11%, Motivational 15% at test) are exactly what we want the model to get right, and micro-averaging would let the Cognitive class hide their failure. We report accuracy alongside it only as a sanity check against the majority-class floor.

**Protocol.** Early stopping on validation loss (patience 15, cap 100 epochs), then a single evaluation on Study 3. Every cell of Section 4 is one training run at a fixed seed, and the confusion matrix shown for that cell comes from the same run, so the table and the figure describe one object. Seed 9 is the seed our Semester B results were presented on and we keep it throughout, with one exception noted below. Runs do vary by seed, so gaps of a few thousandths between cells should not be read as real.

**Reference points.** Three floors and two non-neural references anchor every comparison. Without them, a model scoring 0.68 accuracy looks successful when it is in fact below the 0.734 trivial floor.

**Table 1.** Reference points on the Study 3 test set.

| Reference | Accuracy | Macro F1 |
|---|---|---|
| Always answer Cognitive | 0.734 | 0.282 |
| Permuted training labels | n/a | 0.309 |
| Label-noise floor (rating resampling) | n/a | 0.342 |
| TF-IDF counts with logistic regression | n/a | 0.350 |
| Hand-written regex over opening phrases | n/a | 0.361 |
| Linear baseline: frozen `[CLS]` with linear head | 0.684 | 0.336 |

The permutation null is the most useful of these. Training the same architecture on shuffled labels still reaches 0.309, so the linear baseline contains only a small amount of learned signal. The label-noise floor is more informative still: two independent draws from the same annotation distribution agree only at 0.342 macro F1, so the whole band our models occupy is within roughly 0.06 of the point where the labels stop distinguishing anything. The two non-neural references sharpen the point: a plain TF-IDF logistic regression ties the linear baseline at 0.350, and a hand-written regex over opener phrases reaches 0.361, above the linear baseline and within 0.017 of the best cell of Table 2, with no learning at all.

Throughout this report we call the last row the **linear baseline**. The frozen encoder with an MLP head introduced in Section 4 is a stronger and separate model, and we call it the **frozen+MLP baseline**.

---

## 4. Prediction models

**Baseline model (Semester A).** A frozen DistilBERT encoder produces one `[CLS]` vector per text, and a trained linear head turns it into a distribution over the three types. The encoder never updates. This is the model our Semester A work settled on and the one we carried into Semester B as the comparison point, and it is the linear baseline of Table 1. We also built a stronger frozen variant with an MLP-256 head, and we return to it below because it matters for how much credit fine-tuning deserves.

**Final model (Semester B).** The same encoder carrying LoRA adapters of rank 4 on the query and value projections of all six attention blocks, about 74k trainable parameters against 66M frozen. A skip connection gives the head both the adapted and the original `[CLS]` vector. Full configuration: AdamW, head learning rate 3e-4 and adapter learning rate 3e-5, weight decay 0.01, batch size 32, MLP-256 head at dropout 0.5, balanced sampling, at most 100 epochs with early-stopping patience 15, which in practice stops around epoch 30. A greedy search that changed one component at a time over roughly 150 cluster runs produced this configuration.

**Feature sets.** *Response only* is the 768-dimensional `[CLS]` vector of the reply on its own. *Story+Response* encodes the disclosure and the reply separately and concatenates them into a 1536-dimensional vector, so the head can see the situation the reply was written for. Each was tuned on its own learning rate. We tried a third option as well, replacing DistilBERT with RoBERTa-base so the embedding itself changed rather than its input, and it reached 0.364. That sits inside the same band as everything else, so we report it as a negative result and keep the two `[CLS]` feature sets in the grid.

**Table 2.** Macro F1 on Study 3 for two models and two feature sets. Every cell is a single training run, and the confusion matrix for that same run appears in Figure 2 of the PDF.

| Model | Response only | Story+Response | Gain from story |
|---|---|---|---|
| Baseline: frozen `[CLS]`, linear head | 0.336 | 0.333 | −0.003 |
| Final: LoRA r4 qv all-6 | 0.352 | **0.400** | +0.048 |
| Gain from LoRA | +0.016 | +0.067 | |

**Reading the grid.** The final model improves on the baseline under both feature sets, by 0.016 on the reply alone and by 0.067 once the disclosure is included, reaching 0.400 against the baseline's 0.336. The two axes interact rather than adding up. Story context does nothing for the frozen linear head, which loses 0.003 when given it, and helps the fine-tuned model by 0.048. That pattern makes sense: a linear map over a 1536-dimensional concatenation has twice the parameters and no way to relate the two halves, while LoRA can adapt the encoder so the story representation carries something the head can use. The combination of fine-tuning and story context is therefore worth more than either alone, which is the main modelling result of Semester B.

**One qualification on how much credit fine-tuning deserves.** The strongest frozen model we ever built was not the linear baseline but a frozen encoder with an MLP-256 head on Story+Response features, which reaches 0.378. Measured against that model rather than against the Semester A baseline, the LoRA run is ahead by 0.022 rather than by 0.067. We report this because it changes the interpretation: fine-tuning does beat the baseline we carried forward, and a good part of that gain is reachable with a better head on a frozen encoder. Section 5 explains why, and the explanation is the same for both models.

**Which model we prefer.** The LoRA Story+Response model, as the final model, on the grounds that it has the best result of the four cells. It costs more to train, since the encoder no longer runs once into a cache, and it carries wider variance from run to run. Where training cost matters more than the last few hundredths of macro F1, the frozen MLP head on Story+Response features is the practical choice.

---

## 5. Model analysis

We analyse the linear baseline and the final LoRA model with two approaches, reading confident errors by hand and perturbing the input, and we ran both on both models. The frozen+MLP variant appears alongside them wherever we measured it. Two probes exist for one model only, the opener swap for LoRA and the encoder noise for the linear baseline, so those support conclusions within a model rather than across models.

### 5.1 Approach one: mining confident errors

For each model we extracted the five most confident correct and five most confident wrong predictions per class and read them. The result was the same for every model and is the central finding of the project: **predictions track the opening phrase of the response, not its content.** Responses beginning "It sounds like…" or "It's clear that…" are predicted Cognitive; those beginning "I'm really sorry to hear…" and later containing "remember" or "try" are predicted Motivational.

Counting how many of the five confident errors per class are explained by their opener template: 5/5 in every class for the linear baseline; 5/5, 5/5 and 4/5 for the LoRA model; and 3/5, 4/5 and 4/5 for the frozen+MLP Story+Response baseline.

**Table 3.** The LoRA model's five most confident wrong predictions on true-Cognitive test rows. All five open "I'm really sorry" and all five are predicted Motivational, regardless of the response that follows.

| Test idx | Predicted | True | Opening words | Soft label |
|---|---|---|---|---|
| 60 | Mot | Cog | I'm really sorry that you're feeling this way, but I'm unable… | [0.46, 0.17, 0.37] |
| 1163 | Mot | Cog | I'm really sorry to hear that you're feeling this way, but… | [0.35, 0.29, 0.35] |
| 128 | Mot | Cog | I'm really sorry to hear what you're going through, and I… | [0.37, 0.37, 0.26] |
| 842 | Mot | Cog | I'm really sorry to hear that you've been feeling this way… | [0.40, 0.27, 0.33] |
| 873 | Mot | Cog | I'm really sorry to hear that you're feeling underappreciated… | [0.37, 0.33, 0.30] |

Source: `outputs/a2_lora_examples.md`.

**Comparing the models.** LoRA changes the symptom without changing the disease. The linear baseline predicts Cognitive on 95% of test inputs and recovers almost no Affective cases; the frozen+MLP baseline already spreads wider; the LoRA model spreads wider still, lifting Affective recall from about 2% to 17% and Motivational from 7% to 24%. But its confident errors are still explained by the opener, and the templates involved are the same ones. Fine-tuning redistributed the model's bias; it did not give the model a new basis for deciding.

### 5.2 Approach two: input-perturbation sensitivity

**Table 4.** Effect of input perturbations on macro F1. Unlike Table 2 these are averages over 10 runs per cell, because the effects are a few hundredths of macro F1 and a single run cannot separate them from run-to-run variation. Each column is read against its own control, and the three controls differ.

| Perturbation (training + eval) | Linear baseline | Frozen+MLP | Final (LoRA) |
|---|---|---|---|
| None (control) | 0.354 | 0.368 | 0.368 |
| Strip the first 10 words | 0.339 (−0.015) | 0.360 (−0.009) | 0.332 (−0.036) |
| Swap opener across classes, p = 0.3 |, |, | 0.363 (−0.005) |
| Swap opener across classes, p = 0.5 |, |, | 0.366 (−0.002) |
| Swap opener across classes, p = 1.0 |, |, | 0.340 (−0.028) |

**Both perturbations of the opener hurt, and nothing else does.** The LoRA penalty is large relative to the learned signal the linear baseline has over the permutation null. Sizing the effect against the always-Cognitive floor of 0.282: fully breaking the opener-to-label mapping takes the LoRA model from 0.368 to 0.340, removing about a third of its above-floor signal. The opener is therefore the single most load-bearing feature, but the body of the response is not unread.

**The features are barely used at all.** On the linear baseline, no single `[CLS]` dimension correlates with the target above |r| = 0.21, adding noise to the most correlated dimension changes nothing, and about 50 dimensions must be corrupted before accuracy moves. Heavy noise on all 768 dimensions *raises* macro F1 from 0.327 to 0.349 while dropping accuracy from 0.72 to 0.61, the damage is to the Cognitive bias, not to real signal. This directly motivated the balanced sampling and augmentation used in the final model.

**What the two approaches jointly establish.** Error mining says the models decide on openers; perturbation says the decision is fragile to opener content but robust to everything else. The two models differ in how they distribute their predictions but not in what they are reading. This is why the four cells of Table 2 land within 0.02 of each other.

---

## 6. Research task: ProxySPEX feature recovery

**The method.** ProxySPEX (Butler et al., 2025) explains a model by treating its output as a function over binary masks of the input tokens and extracting the largest Fourier coefficients of that function. A coefficient on a token subset *T* measures the joint influence of those tokens together, so the method captures interactions rather than per-token attributions. Computing the spectrum exactly is exponential; ProxySPEX instead fits a gradient-boosted-tree proxy on (mask, output) pairs and reads the coefficients off the proxy in closed form, needing about O(n log n) model queries.

**Why it fits our problem.** Sections 5.1 and 5.2 gave converging evidence that our models key on opener phrases, but both are indirect: one reads examples, the other deletes text. Neither says *which* tokens the model combines. Opener templates are multi-word units, so a method that measures token interactions rather than single-token importance is the right instrument.

**How we applied it.** We ran ProxySPEX over the first ten tokens of each Study 3 response, with subsets up to size three and the top ten coefficients retained per example. Two models were run at full budget, the LoRA final model and a RoBERTa-base variant trained with the same recipe, each with 20 examples per class and 256 masks. Implementation is in `src/proxyspex_opener.py`.

**Results.** The recovered spectrum is dominated by three-token subsets. For the LoRA model's Cognitive examples, the 200 retained coefficients split into 127 triplets, 62 pairs and 11 singletons, and the other classes and the RoBERTa run look the same. The dominance of triplets is mechanistic: openers such as "I'm / really / sorry" are three-word units and removing any one word breaks the template, so the variance of the prediction lives in three-word coefficients. And two architecturally different encoders converge on the same features, hard to explain as a quirk of any one model, easy to explain as the labels rewarding exactly this structure.

**Do the models fail on the same examples?** On Affective rows all three reference models are wrong together 107 times out of 132, but the independence null already predicts 103.4, so this is not shared confusion, each model simply has a marginal error rate above 83% on that class. On Cognitive rows all three are wrong on only 11 of 860 against a null of 1.0 (p ≈ 10⁻⁸): genuine shared confusion on a small set of ambiguous rows. The explanation for Affective is in the labels: mean label entropy on Affective rows is 1.57 of a possible 1.585 bits.

---

## 7. Conclusions

**What worked.** Treating macro F1 and the permutation null as the real metrics, rather than accuracy, kept us honest: the linear baseline's 0.65 mean accuracy, and even the 0.72 of its luckiest seed, is at or below the 0.734 trivial floor, and only the macro-F1 view revealed it. Giving the model the disclosure alongside the reply helped, though only once the encoder could adapt to it, which took us a while to see. Checking a result on more than one seed before believing it stopped us several times from claiming gains that later evaporated. Caching frozen encoder outputs turned a five-minute experiment into a one-second one and is what made a search of this size possible.

**What did not work.** Nearly everything on the model side. LoRA, partial encoder unfreezing, attention and mean pooling, deeper and wider heads, class-weighted and focal losses, label sharpening, random forests, chunked features, and a switch to RoBERTa-base all landed in the same 0.35 to 0.40 band. Training on hard labels was actively harmful, collapsing macro F1 to 0.166. The most useful result of the whole project is negative, and it took roughly 150 runs to establish with confidence.

**Limitations.** Our estimate of the ceiling is a proxy rather than a Bayes limit. The label-noise floor of 0.342 comes from resampling two independent hard labels per row from its soft label; because the source ratings are continuous averages rather than categorical votes, standard inter-annotator agreement statistics cannot be computed and we could not validate the proxy against them. The test set is one study, so generalisation here means generalisation to a single new prompt set. The numbers we report are single runs rather than averages, so a few thousandths either way should not be read as meaningful.

**Future directions.** Stop trying to break the ceiling and start measuring it properly: obtain raw per-annotator ratings so a real agreement statistic can replace our resampling proxy, extend the error-overlap analysis to per-example agreement on the 107 all-three-wrong Affective rows, and consider whether a different label schema, multi-label rather than a forced mixture, would carry more information per response.

### 7.1 What we learned from the process

The clearest lesson is that expressive features buy nothing when the data is limited. Every step up the capacity ladder returned almost nothing: a linear head on frozen `[CLS]` gives macro F1 0.350, an MLP-256 head gives 0.374, adding the disclosure gives 0.378, LoRA fine-tuning gives 0.375, and RoBERTa-base, with a different pretraining objective on roughly ten times the corpus, gives 0.364. That is a span of 0.028 macro F1 across changes that ranged from swapping a head to swapping the entire encoder, and the last two steps were negative. The second lesson is that our hyperparameter search was far more gradual and less interpretable than we expected, and that several conclusions of the form "this architecture does not work" were really "the training schedule was wrong". Story+Response is the clearest case: at the learning rate tuned for response-only features (3e-5) it scored 0.373 against response-only's 0.374, so we briefly held evidence that the disclosure was useless, and only after retuning to 5e-6 did it reach 0.381. Similarly, our original MLP was early-stopping at a mean of 11.7 epochs under lr 1e-3 with patience 5, so its 0.330 said more about the schedule than the architecture. We think this means some of our earlier negative results were simply not trustworthy at the time we drew them, and we do not know how many of the rejected branches would survive a retuned schedule.

The third lesson is that one metric is not enough, and that the choice of metric changes the story. The same baseline checkpoint scores 0.719 accuracy and 0.327 macro F1; the first number looks like a working classifier and the second shows it is below the majority-class rate of 0.734, which is the real floor. The two metrics can even move in opposite directions on the same intervention: adding Gaussian noise to all 768 encoder features at sigma = 1 *raises* macro F1 from 0.327 to 0.349 while dropping accuracy from 0.719 to 0.611, because the damage lands on the Cognitive bias rather than on signal. We also trained against soft labels but evaluated through an argmax, and that gap matters more than we assumed: training the same model on hard labels instead collapses macro F1 to 0.166, so the calibration in the soft targets is doing real work that our evaluation never measures. Finally, extracting an emotional signal from a small, human-rated dataset is genuinely noisy, and we now believe the ceiling sits in the labels rather than the model. Training on shuffled labels still reaches 0.309 against the real baseline's 0.350, leaving about 0.041 of learned signal. The baseline's mean output across the test set is [0.36, 0.32, 0.33] and its per-dimension standard deviation is near 0.01, so it barely moves with its input. Leave-one-out retraining over all 1218 training rows gives a mean influence of −0.003 macro F1 with a standard deviation of 0.019 and a worst case of ±0.05, so no single example is decisive and the influence is diffuse rather than concentrated. And the labels themselves are close to maximum entropy where it matters most: Affective rows average 1.57 bits of label entropy out of a possible 1.585, with mean probability 0.36 on the true class, and of 132 Affective test rows exactly one is classified correctly by all three reference models. When human raters given affect-targeted prompts still return near-ties, a classifier trained on the mean of those ratings has very little left to recover.

---

## References

- Butler, L. et al. (2025). *ProxySPEX: Inference-Efficient Interpretability via Sparse Feature Interactions in LLMs.*
- Hu, E. J. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
- Liu, Y. et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach.*
- Sanh, V., Debut, L., Chaumond, J., Wolf, T. (2019). *DistilBERT, a distilled version of BERT.*

---

## Repository layout

```
empathy-classifier/
  docs/final_paper.tex    the report source (compiles to the submitted PDF)
  docs/make_figures.py    generates the figures the report embeds
  src/                    training, analysis and figure code (see below)
  outputs/                one result artifact behind every number in the report
  data/                   raw CSV and cached embeddings (not in git, see below)
```

### `src/`, what produces what

| Report section | Code |
|---|---|
| §3 linear baseline | `train.py`, `evaluate.py`, `data.py`, `model.py` |
| §4 frozen+MLP baseline | `head_trainer.py`, `arch_search.py`, `run_balanced_experiments.py`, `run_scale_aug.py`, `run_story_100seeds.py`, `run_story_experiments.py`, `run_story_lower_lr.py`, `run_arch_step2.py` |
| §4 final LoRA model | `run_lora.py`, `run_lora_story.py`, `run_lora_story_roberta.py` |
| §5.1 error mining | `a2_lora_failure_modes.py`, `a2_frozen_mlp_failure_modes.py` |
| §5.2 perturbations | `baseline_strip_probe.py`, `frozen_mlp_strip_probe.py`, `opener_probe.py`, `a4_sensitivity.py` |
| Table 1 permutation null | `a1_permutation_null.py` |
| §6 research task | `proxyspex_opener.py` |
| Confusion matrices | `dump_test_preds.py`, `dump_grid_preds.py`, `docs/make_figures.py` |
| Embedding caches | `cache_embeddings.py`, `build_story_caches.py` |

## Data (not in this repository)

Two things the code needs are deliberately not committed:

1. **The raw dataset**, expected at
   `empathy-classifier/data/raw/Supplementary Data - Responses and Measures - all experiments (1).csv`.
   It is the supplementary data of Rubin et al. (*Nature Human Behaviour*, 2025) and is
   not ours to redistribute. Obtain it from that paper. Only Figure 1 and the
   training scripts read it.
2. **Cached `[CLS]` embeddings** in `empathy-classifier/data/processed/` (~1.1 GB, too
   large for git). The frozen-head scripts load these rather than re-running the
   encoder. Regenerate them from the CSV with:

   ```bash
   cd empathy-classifier
   python src/cache_embeddings.py      # response-only, 768-d
   python src/build_story_caches.py    # story+response, 1536-d
   ```

Model checkpoints (~1 GB of `.pt` files) are also excluded.

## Building the report

```bash
cd empathy-classifier
pip install -r ../requirements.txt
python docs/make_figures.py                            # -> docs/_paper_figs/*.png
pdflatex -output-directory=docs docs/final_paper.tex   # run twice, for cross-references
```

`make_figures.py` reads the raw CSV for Figure 1 and the `outputs/preds_*.npz`
prediction files for the confusion matrices. Every other number in the report is a
literal in `final_paper.tex`, each traceable to a file in `outputs/`.

For Overleaf, upload `docs/final_paper.tex` together with the generated
`docs/_paper_figs/` folder. The preamble avoids `titlesec` so it builds on a minimal
TeX install as well.

The full research history, roughly 150 cluster runs, 50 SLURM job scripts, the
rejected architecture branches and the running search log, lives on the
[`process`](../../tree/process) branch.
