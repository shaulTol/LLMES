# Predicting the Dominant Empathy Type of a Written Response

Shaul Tolkowsky, Guy Shiff · 52056 Data Science Project

> This README is the project report in Markdown. The submitted PDF is compiled from
> `empathy-classifier/docs/final_paper.tex` and adds two figures (the exploratory
> panels and the confusion-matrix grid) plus four appendices. See
> [Building the report](#building-the-report).

---

## 1. Abstract

Rubin et al. (2025) work with a three-way split of empathy: a cognitive component, understanding what another person is going through; an affective component, feeling with them; and a motivational component, caring enough to want to help. We take that split and train a classifier to predict which of the three dominates a written reply to someone's personal disclosure.

Their study asked whether empathy is valued differently depending on whether people believe it came from a person or from an AI. Participants read short disclosures together with the replies written to them, and rated every reply on all three components, five questionnaire items per component. Each reply therefore carries one participant's scores, and normalising the three component means leaves a share of cognitive, affective and motivational empathy that sums to one. The largest of the three is what we call the dominant type. Those shares turn out close to even, with the dominant component usually holding a little more than a third of the mass, so the task is harder than a three-way choice sounds. We train on Studies 1 and 1b, which give 1218 replies, and test on Study 3, which gives 1172 replies written for different prompts. Cognitive dominates 73% of the test replies, so always answering Cognitive already scores 0.73 accuracy, and we report macro F1 throughout with accuracy kept only as a sanity check.

We compare two models on two feature sets. The Semester A baseline freezes a DistilBERT encoder and trains a small head on its `[CLS]` vector. The Semester B final model adds LoRA adapters so the encoder itself can adapt. Each is run on the reply alone and on the reply concatenated with the disclosure it answers. The final model improves on the baseline under both feature sets, from 0.336 to 0.400 macro F1, against 0.282 for the majority-class floor. The two changes interact, since the disclosure helps only once the encoder can adapt to it. The gains stay small in absolute terms, and our analysis puts the reason in the data rather than in the model. Applying ProxySPEX (Butler et al., 2025) recovers the same three-word opener features from two different encoders, and every intervention we tried that removes those features costs macro F1 instead of forcing the model onto better ones. Swapping DistilBERT for RoBERTa-base lands in the same range, and on the Affective replies, where the ratings are closest to a three-way tie, all three of our models fail together. Taken together this points towards the labels rather than model capacity as what limits the result.

---

## 2. Exploratory analysis (Semester A)

Every reply in the dataset comes paired with the disclosure it answers, which we call the *story*, and with three numbers describing how one participant scored it on the cognitive, affective and motivational components. Those numbers reach us already aggregated: each participant answered a 15-item questionnaire, five items per component, and the supplementary data reports the per-component means rather than the individual items. We take them as given and divide each by the sum of the three, which turns a triple such as `[9.0, 8.6, 8.6]` into shares `[0.34, 0.33, 0.33]`. This is what makes the label a probability distribution and therefore a valid target for the soft cross-entropy of Section 3. It also discards the overall intensity of the ratings, so a reply scored `[9.0, 8.6, 8.6]` and one scored `[3.0, 2.9, 2.9]` become almost the same target and the model learns the balance between the three components rather than how empathic the reply was judged to be. That suits the question we ask, which is which component dominates.

**The dominant type barely dominates.** A typical label looks like `[0.36, 0.31, 0.33]`. Participants rarely scored a reply high on one component alone, and we read this as real rather than as noise. A reply saying "that sounds exhausting, and I am here if you need anything" genuinely carries perspective-taking and an offer of help at the same time. It also limits what any classifier can reach, because picking the largest of three near-equal numbers is close to arbitrary.

**The class prior is skewed, and it moves between splits.** Cognitive leads in both, at 67% of training replies and 73% of test replies. This drove our choice of metric. Always answering Cognitive scores 0.734 accuracy and 0.282 macro F1, so accuracy rewards a model for leaning on the prior, while macro F1 exposes it.

**The text is short and templated.** Replies run 45 to 110 words and were produced under a handful of prompt conditions. Word-frequency and style analysis showed the same few openings reused constantly, among them "It sounds like…" and "I'm really sorry to hear…", with lexical diversity low and roughly equal across the three types. We noted at the time that a classifier might key on those openings. Section 5 shows that it did.

---

## 3. Training and evaluation process (Semester A)

**Splits.** Training uses Studies 1 and 1b less 50 held-out replies per study, giving 1218 rows. Those 100 held-out rows form the validation set. Study 3 is the test set, all 1172 rows. We tested on a separate study instead of a random split because we care whether the model handles new prompts, and a random split would have measured something easier. The reported numbers are lower as a result.

**Objective.** Soft cross-entropy against the full rating distribution. We keep the distribution intact because collapsing it to a single label throws away information about how close the three types were, and Section 7 shows what that costs.

**Metric.** Macro F1 over the three classes, computed on the argmax of the predicted distribution. We chose macro over micro averaging because Affective and Motivational cover only 11% and 15% of the test set, and micro averaging would let good Cognitive performance mask complete failure on the other two. Accuracy appears alongside it only as a check against the majority-class floor.

**Protocol.** Early stopping on validation loss with patience 15 and a cap of 100 epochs, then evaluation on Study 3. Every cell of Section 4 is one training run at a fixed seed, and the confusion matrix shown for that cell comes from the same run, so the table and the figure describe one object rather than two. Seed 9 is the seed our Semester B results were presented on and we keep it throughout, with one exception noted in Appendix D. Runs vary by seed, so gaps of a few thousandths between cells should not be read as real.

**Table 1.** Reference points on the Study 3 test set, ordered by macro F1.

| Reference | Accuracy | Macro F1 |
|---|---|---|
| Always answer Cognitive | 0.734 | 0.282 |
| Permuted training labels | 0.611 | 0.309 |
| Hand-written regex over opening phrases | 0.567 | 0.343 |
| Linear baseline: frozen `[CLS]` with linear head | 0.684 | 0.336 |

The permutation null is the most useful of these. Training the same architecture on shuffled labels still reaches 0.309, so the linear baseline's 0.336 contains only about 0.027 of learned signal. The regex row sharpens this further: a dozen hand-written patterns over opening phrases reach 0.343, slightly above the linear baseline, with no learned representation involved at all. Note also that accuracy and macro F1 disagree across the whole table. The highest accuracy belongs to the row that answers Cognitive every time and scores the worst macro F1 of any, which is the clearest argument for the metric we chose.

---

## 4. Prediction models

**Baseline model (Semester A).** A frozen DistilBERT encoder produces one `[CLS]` vector per text, and a trained linear head turns it into a distribution over the three types. The encoder never updates, and the baseline uses neither balanced sampling nor augmentation, so it is the plain configuration.

**Final model (Semester B).** The same encoder carrying LoRA adapters of rank 4 on the query and value projections of all six attention blocks, about 74k trainable parameters against 66M frozen. A skip connection gives the head both the adapted and the original `[CLS]` vector. Full configuration: AdamW, head learning rate 3e-4 and adapter learning rate 3e-5, weight decay 0.01, batch size 32, MLP-256 head at dropout 0.5, balanced sampling, and at most 100 epochs with early-stopping patience 15, which in practice stops around epoch 30. Balanced sampling here means a weighted sampler that draws training rows with replacement, weighting each row inversely to how common its dominant type is. Batches therefore come out roughly even across the three types rather than following the 67/12/21 split of the training data.

**Feature sets.** *Response only* is the 768-dimensional `[CLS]` vector of the reply on its own. *Story+Response* encodes the disclosure and the reply separately and concatenates them into a 1536-dimensional vector. Each was tuned on its own learning rate. We tried a third option as well, replacing DistilBERT with RoBERTa-base so the embedding itself changed rather than its input, and it reached 0.376. That sits inside the same band as everything else, so we report it as a negative result.

**Table 2.** Macro F1 on Study 3 for two models and two feature sets. The two upper rows are models, one per row, evaluated under each feature set; every one of those four cells is a single training run. The final row and the final column are differences between the cells beside them, not models or feature sets.

| Model | Response only | Story+Response | *Gain from story* |
|---|---|---|---|
| Baseline: frozen `[CLS]`, linear head | 0.336 | 0.333 | −0.003 |
| Final: LoRA r4 qv all-6 | 0.352 | **0.400** | +0.048 |
| *Gain from LoRA* | +0.016 | +0.067 | |

**Reading the grid.** The final model improves on the baseline under both feature sets, by 0.016 on the reply alone and by 0.067 once the disclosure is included, reaching 0.400 against the baseline's 0.336. The two axes interact rather than adding up. Story context does nothing for the frozen linear head, which loses 0.003 when given it, and helps the fine-tuned model by 0.048. That pattern makes sense: a linear map over a 1536-dimensional concatenation has twice the parameters and no way to relate the two halves, while LoRA can adapt the encoder so the story representation carries something the head can use. The combination of fine-tuning and story context is therefore worth more than either alone, which is the main modelling result of Semester B.

**One qualification on how much credit fine-tuning deserves.** The strongest frozen model we built was not the linear baseline but a frozen encoder with an MLP-256 head on Story+Response features and latent-Gaussian augmentation to 2500 examples per class, which reaches 0.377. That model gets its class balance from the augmentation rather than from a sampler, so both of our stronger models correct the skewed prior in some way and neither result should be read as coming from architecture alone. Measured against it rather than against the Semester A baseline, the LoRA run is ahead by 0.023 rather than by 0.067.

**Which model we prefer.** The LoRA Story+Response model, as the final model, on the grounds that it has the best macro F1 of the four cells. It costs more to train, since the encoder no longer runs once into a cache. Where training cost matters more than the last few hundredths of macro F1, the frozen MLP head on Story+Response features is the practical choice.

Per-class recall from the confusion matrices: the linear head answers Cognitive almost everywhere and recovers close to nothing on Affective, while the LoRA panels spread their answers across all three classes. Affective is the class every configuration fails hardest on.

---

## 5. Model analysis

We analyse the linear baseline and the final LoRA model by perturbing their inputs and watching what the perturbation costs.

If the opener drives the answer, damaging the opener should hurt more than damaging anything else. We tested this two ways during training. The first removes the first ten words of every reply, and we ran it on both models, so it supports a comparison between them. The second replaces those words with an opener taken from a reply of a different class, at three rates, and we ran it on the final model only, so it speaks to that model alone.

**Table 3.** Removing the opener, applied at training and evaluation time. Each column is read against its own control, and the two controls differ.

| | Linear baseline | Final (LoRA) |
|---|---|---|
| None (control) | 0.354 | 0.368 |
| Strip the first 10 words | 0.339 (−0.015) | 0.332 (−0.036) |

**Table 4.** Replacing the opener with one drawn from a reply of a different class, at three rates, on the final model.

| | Final (LoRA) |
|---|---|
| None (control) | 0.368 |
| Swap opener across classes, p = 0.3 | 0.363 (−0.005) |
| Swap opener across classes, p = 0.5 | 0.366 (−0.002) |
| Swap opener across classes, p = 1.0 | 0.340 (−0.028) |

**Damaging the opener hurts both models.** Stripping it costs the linear baseline 0.015 and the final model 0.036, and breaking the opener-to-label mapping entirely with a cross-class swap costs the final model 0.028. Set against the 0.027 of learned signal the linear baseline holds over the permutation null, these are large. The reading the data support is that the fine-tuned encoder depends on the opener more than the frozen one does. Measured against the always-Cognitive floor of 0.282, breaking the opener mapping takes the LoRA model from 0.368 to 0.340, which removes about a third of its above-floor signal. The opener carries the most weight of any single feature, and the body of the reply still contributes the rest.

**The opener may be a shallow feature the models lean on too heavily, but removing it does not help.** The obvious remedy is to train the shortcut out. It does not work. Stripping the opener costs the fine-tuned model 0.036, and swapping openers across classes costs 0.005 at p = 0.3 and 0.028 at p = 1.0. The one opener intervention that helps is the same-class swap used in the final recipe, worth 0.007, and that one leaves the template-to-label mapping intact while varying the wording. Read against the majority-class floor of 0.282, the stripped model still holds 0.050 of the 0.086 the control holds above the floor, so the body of the reply carries most of what the model knows even though the opener is the single most load-bearing feature. Removing the shortcut does not reveal better features underneath; it removes signal and leaves the rest where it was.

---

## 6. Research task: ProxySPEX feature recovery

**The method.** ProxySPEX (Butler et al., 2025) explains a model by treating its output as a function over binary masks of the input tokens and extracting the largest Fourier coefficients of that function. A coefficient on a token subset *T* measures what those tokens contribute jointly, so the method reports interactions between tokens instead of per-token attributions. The exact spectrum is exponential to compute, so ProxySPEX fits a gradient-boosted-tree proxy on (mask, output) pairs and reads the coefficients off the proxy in closed form, using about O(n log n) model queries.

**Why it fits our problem.** Section 5 points at opening phrases, and it does so indirectly, by deleting text and measuring what that costs. It does not identify which tokens the model combines. Opener templates are multi-word units, so a method built around token interactions is the right instrument here. We also wanted a test we could apply unchanged to different encoders.

**How we applied it.** We ran ProxySPEX over the first ten tokens of each Study 3 reply, allowing subsets up to size three and keeping the top ten coefficients per example. Two models ran at full budget, the LoRA final model and a RoBERTa-base variant trained with the same recipe, each on 20 examples per class with 256 masks. The linear baseline ran only as a reduced smoke test. The implementation is `src/proxyspex_opener.py`.

**Results.** Three-token subsets dominate the recovered spectrum. For the LoRA model's Cognitive examples the 200 retained coefficients split into 127 triplets, 62 pairs and 11 singletons, and the other classes and the RoBERTa run look the same. The dominance of triplets is mechanical: an opener such as "I'm / really / sorry" is a three-word unit, so removing any one word breaks the template and the variance of the answer collects in three-word coefficients. More importantly, two architecturally different encoders converge on the same features. We would not expect a quirk of one model to reproduce across encoders, so the more likely reading is that the labels reward this structure, though two encoders is a thin basis for saying so.

**Do the models fail on the same replies?** We compared the test errors of the three reference models against an independence null. On Affective rows all three are wrong together 107 times out of 132, and independence already predicts 103.4, so this reflects each model having a marginal error rate above 83% on that class. On Cognitive rows all three are wrong on only 11 of 860 against a null of 1.0, p ≈ 10⁻⁸, which is genuine shared confusion over a small set of ambiguous rows. On Motivational rows all three are wrong 111 times out of 180 against a null of 96.0, p = 0.015. For Affective the labels themselves look like the likeliest explanation: mean label entropy on Affective rows is 1.57 bits of a possible 1.585.

---

## 7. Conclusions

**What worked.** Treating macro F1 and the permutation null as the real metrics kept us honest. The linear baseline sits at or below the 0.734 trivial floor on accuracy, and only the macro F1 view showed it. Optimising one metric is fine, but reading only that metric hides what the model is doing: our baseline checkpoint scores 0.719 accuracy and 0.327 macro F1, and the two even move in opposite directions when we add Gaussian noise to all 768 encoder features, which raises macro F1 to 0.349 while dropping accuracy to 0.611 because the noise lands on the Cognitive bias rather than on signal. Giving the model the disclosure alongside the reply helped, though only once the encoder could adapt to it, which took us a while to see. Correcting the skewed class prior mattered too, though less in the headline number than in what the model predicts. The sampler barely moves macro F1 on its own, yet both models that beat the linear baseline correct the prior in some way, one with a sampler and one with augmentation, and the confusion matrices show the effect plainly as answers spreading off Cognitive and onto the two minority types. Tuning also paid, more than we expected for something we had treated as bookkeeping: retuning the learning rate for the Story+Response features moved them from 0.373 to 0.381, and fixing the early-stopping schedule moved the MLP head off 0.330. Re-running a promising configuration before believing it stopped us several times from claiming gains that did not hold up. Caching the frozen encoder's outputs turned a five-minute experiment into a one-second one, and a search of this size would have been impossible without it.

**The search itself was harder to read than we expected.** A step that helps at one stage can stop helping at another for reasons that are not obvious in advance. Story+Response is the clearest case: at the learning rate tuned for response-only features it scored 0.373 against response-only's 0.374, so we briefly held evidence that the disclosure was useless, and retuning brought it to 0.381. Our original MLP was stopping early at a mean of 11.7 epochs, so its 0.330 described the schedule more than the architecture. Latent augmentation is the case we understand best in hindsight. It helps a frozen encoder, where the `[CLS]` vectors are a fixed point cloud and Gaussian noise genuinely smooths the boundary the head draws through them. Once LoRA is training that cloud moves at every step, so noise calibrated to the original representation is calibrated to nothing, and the head cannot separate augmentation noise from the encoder actually learning. The lever became null and we had recorded it as a gain. Some of our earlier negative results were therefore not trustworthy at the moment we drew them.

**What did not work.** Almost everything on the model side. LoRA, partial encoder unfreezing, attention and mean pooling, deeper and wider heads, class-weighted and focal losses, label sharpening, random forests, chunked features and a switch to RoBERTa-base all landed in the same 0.35 to 0.40 band. Training on hard labels actively hurt, dropping macro F1 to 0.166. The most useful result of the project is a negative one, and establishing it with any confidence took about 30 steps of search.

**Limitations.** We never measured the ceiling we keep pointing at, and we could not have done so with this dataset. By the ceiling we mean the best macro F1 any predictor could reach given how unstable the target is, and the way to measure it is to have several people rate the same reply and then score one rater's answer against another's under the same metric. No model should be expected to predict a label better than a second person does. Every reply here carries the scores of exactly one participant, so there is no disagreement to measure. Every statement we make about the ceiling is therefore an inference from convergent evidence rather than a result, and the resampling check of Appendix D is a rough indication and nothing stronger. The test set is one study, so generalisation here means generalisation to a single new prompt set. The numbers we report are single runs rather than averages, so a few thousandths either way should not be read as meaningful, and the LoRA response-only cell predates part of the final recipe.

**Future directions.** The obvious next step is to measure the limit rather than infer it, and that needs a small collection effort rather than a better model. Having two or three people label a few hundred replies with the three-way question directly would give a human-against-human macro F1 and settle where the ceiling sits, and it is cheap next to re-running the original study. Beyond that, extending the error-overlap analysis to per-example agreement on the 107 Affective rows all three models miss, and asking whether a different label schema, multi-label rather than a forced mixture, would carry more information per reply.

**What we learned from the process.** Noisy, limited data sets the ceiling, and no amount of feature engineering or model sophistication moves it. Swapping a linear head for an MLP, for LoRA fine-tuning, and finally for a different pretrained encoder moved macro F1 from 0.336 to 0.400, against a majority-class floor of 0.282 that costs nothing at all, and training on shuffled labels already reaches 0.309.

The labels are the reason, and they are hard in two compounding ways. Judging emotional content in text is difficult for people and not only for models, since whether a reply understands someone, feels with them or wants to help them are overlapping readings of the same sentences. On top of that, nobody in this study was ever asked which type a reply belonged to. Participants answered fifteen questionnaire items, five per component, and the component scores are averages over those items; we then normalise the three and take the largest. The target is a construct assembled two steps away from anything a person decided, each reply is rated by a single participant rather than a panel, and different replies are rated by different participants, so between-rater variation enters the dataset as noise we can neither measure nor average away. Affective rows average 1.57 bits of label entropy out of a possible 1.585, and of 132 Affective test rows exactly one is answered correctly by all three of our models. Given labels like these, we would not expect any model to do much better.

---

## References

- Rubin, M., Li, J. Z., Zimmerman, F., Ong, D. C., Goldenberg, A., & Perry, A. (2025). *Comparing the value of perceived human versus AI-generated empathy.* Nature Human Behaviour. https://doi.org/10.1038/s41562-025-02247-w
- Butler, L. et al. (2025). *ProxySPEX: Inference-Efficient Interpretability via Sparse Feature Interactions in LLMs.*
- Hu, E. J. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
- Liu, Y. et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach.*
- Sanh, V., Debut, L., Chaumond, J., Wolf, T. (2019). *DistilBERT, a distilled version of BERT.*

The PDF also carries four appendices: the architecture search chain and rejected
directions, additional diagnostics (opener probe, hard-label training, leave-one-out
influence), further confusion matrices, and a provenance note for every number that needs
one.

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

### `src/` — what produces what

| Report section | Code |
|---|---|
| §3 linear baseline | `train.py`, `evaluate.py`, `data.py`, `model.py` |
| §3 permutation null | `a1_permutation_null.py` |
| §3 opener regex | `nonneural_baselines.py` |
| §4 frozen heads | `head_trainer.py`, `arch_search.py`, `run_balanced_experiments.py`, `run_scale_aug.py`, `run_story_100seeds.py`, `run_story_experiments.py`, `run_story_lower_lr.py`, `run_arch_step2.py` |
| §7 capacity ladder, Appendix A chain | `chain_single_seed.py` |
| §4 final LoRA model | `run_lora.py`, `run_lora_story.py`, `run_lora_story_roberta.py` |
| §5 perturbations | `baseline_strip_probe.py`, `frozen_mlp_strip_probe.py`, `opener_probe.py` |
| §6 research task | `proxyspex_opener.py` |
| Appendix diagnostics | `a4_sensitivity.py`, `a2_lora_failure_modes.py`, `a2_frozen_mlp_failure_modes.py` |
| Confusion matrices | `dump_test_preds.py`, `dump_grid_preds.py`, `docs/make_figures.py` |
| Embedding caches | `cache_embeddings.py`, `build_story_caches.py` |

## Data (not in this repository)

Two things the code needs are deliberately not committed:

1. **The raw dataset**, expected at
   `empathy-classifier/data/raw/Supplementary Data - Responses and Measures - all experiments (1).csv`.
   It is the supplementary data of Rubin et al. (*Nature Human Behaviour*, 2025) and is
   not ours to redistribute. Obtain it from that paper.
2. **Cached `[CLS]` embeddings** in `empathy-classifier/data/processed/` (~1.1 GB, too
   large for git). Regenerate them from the CSV with:

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

The full research history, including every step of the search and the rejected branches,
lives on the [`process`](../../tree/process) branch.
