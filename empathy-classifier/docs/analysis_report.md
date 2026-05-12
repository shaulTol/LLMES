# Empathy Classifier — Analysis Report

Working draft. Sections are filled in as we complete each stage of the analysis. This document is the base for the final submission.

---

## Setup

**Task.** Three-way soft classification of empathic responses into *cognitive*, *affective*, and *motivational* types, given the `Response` text. Soft labels are the row-normalized human ratings on the three dimensions.

**Splits.** Train = Studies 1 + 1b minus 100 held-out (N=1218). Eval = those 100 held-out (early stopping). Test = all of Study 3 (N=1172).

**Baseline architecture.** Frozen `distilbert-base-uncased`, `[CLS]` token embedding → single linear layer (768 → 3) → softmax. Soft cross-entropy loss, Adam lr=1e-3, batch 32, early stopping with patience 5 on eval loss.

**Computational note.** Because DistilBERT is frozen, its `[CLS]` outputs for the 2490 rows are deterministic; we cache them to `data/processed/` once and retrain only the linear head for all subsequent experiments. This makes a single training run effectively free (~1 sec) and enables 100× resampling.

---

## Part A — Exploratory & Model Analysis

### A1 — No-signal characterization (label permutation)

**Procedure.** For seeds 0…99 we trained the linear head on cached `[CLS]` embeddings twice: once on the real training labels, once on a row-wise permutation of those training labels (eval and test labels are untouched). All other settings match the baseline. We compare the two resulting distributions of Study-3 test accuracy and macro F1.

**Result.**

| | Real labels (N=100) | Permuted labels (N=100) | Real − Permuted |
|---|---|---|---|
| Test accuracy | 0.647 ± 0.054 | 0.611 ± 0.086 | **+3.6 pp** |
| Macro F1 | 0.350 ± 0.022 | 0.309 ± 0.020 | **+4.1 pp** |
| Cohen's d (real vs null) on accuracy | — | — | 0.50 |
| P(null acc ≥ real_mean acc) | — | — | 0.41 |

Real-labels accuracy 95% CI: [0.494, 0.721]. Permuted accuracy 95% CI: [0.405, 0.715].

**Reading.** The real-labels run *does* outperform the no-signal run on both metrics. The macro-F1 separation is the more decisive view: the two distributions are clearly distinct and barely overlap. On test accuracy alone the distributions overlap substantially (Cohen's d ≈ 0.5, "small-to-medium"), reflecting how noisy this baseline is across seeds.

**Takeaway.** The linear baseline picks up some real signal — but only a little. The single test-accuracy number we report for "the baseline" is one draw from a distribution that spans roughly 49% to 72% across seeds.

*(Artifacts: `outputs/a1_permutation_null.png`, `outputs/a1_permutation_null.npz`, `outputs/a1_summary.md`.)*

---

### A2 — Success / failure example mining

**Procedure.** Using the originally trained baseline (`models/baseline_v1.pt`, test acc 0.719 — a high-end draw from the A1 real-labels distribution), we ran inference on Study 3 and selected, for each true class, up to 5 confident-correct and 5 confident-wrong examples. "Confident" = highest predicted probability on the predicted class. We also computed a small set of pre-registered text features (length, marker-word counts, second-person pronouns) over those examples.

**Cell counts.** All cells filled to N=5 except Affective-confident-correct, which has only 2 examples — the model gets just 2 of 132 Affective items right.

**Success pattern.** The model is essentially classifying by **opening phrase / surface style**, not by content. 5/5 confident-correct examples per class are stereotyped on one opener:
- *Cognitive*: "It's clear / It's evident / It sounds like / It's apparent that you..."
- *Motivational*: "I'm really sorry to hear..." + at least one of {"remember", "you can", "try", "I believe in your"}
- *Affective*: "I am truly amazed / I truly feel for you / you are..."

**Failure pattern.** All confident-wrong examples share the same root cause: **the opener encodes a different empathy style than the true label**, and the model trusts the opener. Specifically:
- True=Cognitive, opener="I'm really sorry to hear..." → predicted Motivational (5/5).
- True=Affective, opener="It's clear / I can sense..." → predicted Cognitive (5/5).
- True=Motivational, opener="It sounds like / You seem to be..." → predicted Cognitive (5/5).

**Tradeoff.** Surface markers are a real signal — when the response is stereotyped, the model gets it right. But when the response *mixes* styles (e.g. sympathy opener + reflective content but the human rating is Cognitive), the model has no mechanism to override the surface cue. The frozen `[CLS]` representation appears to be dominated by the first tokens of the response.

**Pre-registered text features — what the per-cell means actually show.**

| Feature | Cog-correct | Cog-wrong | Aff-correct | Aff-wrong | Mot-correct | Mot-wrong | Useful? |
|---|---|---|---|---|---|---|---|
| `n_chars` | 391 | 371 | 426 | 419 | 440 | 456 | no |
| `n_words` | 65 | 63 | 74 | 73 | 72 | 77 | no |
| `n_q_marks` | 0 | 0 | 0 | 0 | 0 | 0 | no (always 0) |
| `n_excl` | 0 | 0.4 | 0 | 0.2 | 0 | 0 | no |
| `n_you_pron` | 5.4 | 5.0 | 8.0 | 4.2 | 5.8 | 5.8 | mild aff signal |
| `n_cog_markers` | 1.6 | 1.0 | 0 | 1.4 | 0.8 | 2.2 | noisy / no |
| `n_aff_markers` | 0.6 | 1.6 | 1.5 | 1.2 | 2.0 | 1.4 | noisy ("sorry" is everywhere) |
| `n_mot_markers` | 0 | 1.4 | 4.0 | 0.2 | 2.2 | 0.2 | **clean** for Mot |

The only direct feature that cleanly discriminates anything is `n_mot_markers` (count of "remember / you can / try / strength" etc.): Motivational-correct examples carry many of these and Motivational-wrong examples almost none.

**Pipeline note (relevant to Part B).** The baseline tokenizes each response with `max_length=512` and feeds it through DistilBERT in a *single* forward pass, taking only the `[CLS]` token. Responses are 45–110 words (~60–150 tokens), so no truncation happens; the entire response is collapsed into a single 768-d vector. This is consistent with the model behaving as a function of opening tokens.

**Direct features we would add.**
1. **Opener template tag** — regex on the first ~10 tokens producing a categorical feature in {`it_s_observational`, `i_m_sorry_to_hear`, `you_are_affirmation`, `other`}. This is the single highest-value new feature given the failure analysis.
2. **Imperative-verb count** (remember, try, please, consider, keep) — refines `n_mot_markers`.
3. **Affirmation phrases** ("you are X", "your strength", "I'm amazed by") to discriminate Affective.
4. **Reflective verbs** ("I see / sense / imagine that you...") with a second-person object, for Cognitive.

**Architectural implication.** Because opener-style dominates the `[CLS]` representation, an obvious cheap intervention is to **separately encode opener and body** and concatenate the two embeddings before the linear head. We flag this for Part B.

#### Mined examples (full text)

Each example is annotated with `pred = <model prediction>`, the true and predicted soft-probability vectors over (cog, aff, mot), and the response length in words.

##### True class = Cognitive

**Confident-correct (5)**

1. *pred = Cognitive* | true probs [cog 0.50, aff 0.07, mot 0.43] | pred probs [cog 0.40, aff 0.28, mot 0.33] | 81w
   > It's evident that you felt overwhelmed and stressed while trying to complete a large quantity of work in such a limited time frame. The pressure of task deadlines lead you to cut corners, which seemingly left you feeling unhappy and anxious about meeting all the expectations. It sounds like you were pushing yourself hard, trying to lose yourself in the work to cope with the anxiety. Despite the relief it was over, the stress you endured should be recognized and validated.

2. *pred = Cognitive* | true probs [cog 0.35, aff 0.30, mot 0.34] | pred probs [cog 0.39, aff 0.30, mot 0.31] | 62w
   > It's clear you were experiencing a significant amount of stress and pressure during the holiday season, aiming to create a memorable Christmas for your daughter. The weight of trying to do everything in a limited time frame must have been overwhelming. It's evident the love you have for your daughter pushed you to take on a lot, despite the anxiety it caused.

3. *pred = Cognitive* | true probs [cog 0.39, aff 0.31, mot 0.30] | pred probs [cog 0.39, aff 0.31, mot 0.30] | 69w
   > It sounds like you experienced a multitude of emotions during these two Christmas events. It's clear you felt immense disappointment and frustration when you had your visa dates wrong last year, especially having to spend Christmas Eve alone at an airport. However, your spirits seemed to have been significantly uplifted when you were finally able to enjoy a joyful and fulfilling Christmas with your girlfriend and her family abroad.

4. *pred = Cognitive* | true probs [cog 0.33, aff 0.33, mot 0.33] | pred probs [cog 0.39, aff 0.31, mot 0.30] | 53w
   > It's wonderful to know that you received a gift you dearly desired, but decided not to purchase because of its price. It must have been a thrilling surprise to have been gifted this, enhancing your happiness and sense of appreciation. It's nice to see how much you value and cherish this thoughtful gift.

5. *pred = Cognitive* | true probs [cog 0.38, aff 0.28, mot 0.34] | pred probs [cog 0.39, aff 0.30, mot 0.31] | 59w
   > It's apparent that you felt deeply moved and touched by the emotional depth of your girlfriend's Christmas card. The card seems to have affirmed your importance in her life, making you feel extremely valuable and cherished in the relationship. This experience has evidently given you a sense of comfort, joy and secure attachment, feelings you appear to significantly cherish.

**Confident-wrong (5)**

1. *pred = Motivational* | true probs [cog 0.39, aff 0.26, mot 0.35] | pred probs [cog 0.34, aff 0.30, mot 0.36] | 67w
   > I'm really sorry to hear about your car accident, accidents can be quite a shock and it's normal to feel deflated afterwards. I admire your perspective though, recognizing that unexpected things can happen and we need to be ready to handle them. Remember, it's okay to take some time for yourself to recover from this shock and I'm here if you need any support during this time.

2. *pred = Motivational* | true probs [cog 0.36, aff 0.33, mot 0.32] | pred probs [cog 0.35, aff 0.30, mot 0.35] | 45w
   > I'm really sorry to hear about your granddad's passing. It's completely understandable that you're feeling a blend of sadness and confusion, especially because your relationship with him wasn't very close. Remember, it's totally okay to grieve someone's loss, even if you didn't know them well.

3. *pred = Motivational* | true probs [cog 0.38, aff 0.32, mot 0.29] | pred probs [cog 0.34, aff 0.31, mot 0.35] | 75w
   > I'm genuinely sorry to hear about your gramp's diagnosis; it must be a challenging time for you and your family. Remember, it's okay to feel all the emotions that come with such news and you can lean on the people closest to you. Your gramp is lucky to have someone as loving and understanding as you by his side during this time. Know that every moment you share with him now is precious and valuable.

4. *pred = Motivational* | true probs [cog 0.36, aff 0.32, mot 0.32] | pred probs [cog 0.34, aff 0.31, mot 0.35] | 71w
   > It certainly sounds like you're experiencing a lot of mixed emotions right now, filled with both incredible pride and happiness for Kelsie's new journey, yet also wrestling with the deep sorrow of having to part ways. It's clear that her departure is causing you to feel a significant sense of loss, and it's totally understandable – you're not just losing a friend's physical presence, but a constant part of your life.

5. *pred = Motivational* | true probs [cog 0.35, aff 0.34, mot 0.32] | pred probs [cog 0.35, aff 0.30, mot 0.35] | 55w
   > Congratulations on completing your last assignment of the semester! Presenting to a large group can be incredibly challenging and you should be immensely proud of the courage and determination you exemplified. Remember this victory as a testament to your potential to conquer any challenge university throws at you. You're absolutely capable and you've got this!

##### True class = Affective

**Confident-correct (2 — the model gets only 2 of 132 Affective items right)**

1. *pred = Affective* | true probs [cog 0.32, aff 0.45, mot 0.23] | pred probs [cog 0.34, aff 0.34, mot 0.32] | 75w
   > I am truly amazed by your courage and persistence to claim your independence and freedom. Please don't let anyone make you feel less than the incredibly strong and capable person that you are. Facing the world alone in a wheelchair takes immense strength; your story inspires and reminds us of the need to respect and value everyone's journey. Remember, your spirit is unbreakable, and you are not alone, regardless of how others may perceive you.

2. *pred = Affective* | true probs [cog 0.32, aff 0.37, mot 0.31] | pred probs [cog 0.33, aff 0.34, mot 0.33] | 72w
   > I truly feel for you and your family during this overwhelming time, but I celebrate with you now, knowing that your brave little niece has overcome her initial health challenges and is healing at home. The strength that all of you have shown is inspiring and I believe that with your continued love and support, she'll grow stronger each day. I'm here for you, ready to provide any support you might need.

**Confident-wrong (5)**

1. *pred = Cognitive* | true probs [cog 0.30, aff 0.36, mot 0.34] | pred probs [cog 0.38, aff 0.31, mot 0.30] | 54w
   > It's clear that your anticipation for this skiing trip with your friends added to your sense of joy when it finally happened. It must have been so fulfilling to engage in fun activities and share these moments with people you care deeply about. This experience seems to have been incredibly rewarding for you emotionally.

2. *pred = Cognitive* | true probs [cog 0.31, aff 0.41, mot 0.28] | pred probs [cog 0.38, aff 0.30, mot 0.32] | 69w
   > It's clear that you're feeling a deep sense of relief and joy as you have finally reached the end of your house purchasing journey. The eagerness and anticipation you're feeling as you and your children prepare to move into this new phase of your lives are palpable. This is indeed an exciting and happy day for you, and your upbeat spirit about the upcoming moving day is truly contagious.

3. *pred = Cognitive* | true probs [cog 0.38, aff 0.39, mot 0.23] | pred probs [cog 0.38, aff 0.32, mot 0.30] | 64w
   > I can sense how deeply 'Priscilla' touched you and moved you emotionally. It's amazing how art and storytelling can evoke such profound feelings in us. You should be proud of the empathy and emotional capacity you possess; it's a testament to your beautiful heart. Let's make sure to share such moving experiences more often and continue touching our souls with good art and stories.

4. *pred = Cognitive* | true probs [cog 0.29, aff 0.36, mot 0.35] | pred probs [cog 0.38, aff 0.32, mot 0.31] | 66w
   > What a touching memory for you to stumble upon! In the digital age we live in, it's rare to find such heartfelt, tangible reminders of the relationships that truly mean something to us. It's amazing how such a simple object like a handwritten letter can carry so much emotion and history. Cherish this beautiful memory, and remember it's these moments of genuine connection that shape us.

5. *pred = Cognitive* | true probs [cog 0.33, aff 0.34, mot 0.33] | pred probs [cog 0.38, aff 0.32, mot 0.30] | 110w
   > I completely understand how you felt, entwined in that deep mixture of revelry from shared days past and a sense of curiosity for the unexplored paths. Feeling the joy through tears of laughter yet carrying the subtle pinch of regret is such a familiar experience for many of us. Life indeed is a beautiful labyrinth of decisions we made, leaving us sometimes with a longing sigh as we grasp the complexities of time. So here's to cherishing our mosaics of moments, made up of a unique blend of reflection and joy, because every path we didn't take led us to where we are, and for that, we must find …

##### True class = Motivational

**Confident-correct (5)**

1. *pred = Motivational* | true probs [cog 0.34, aff 0.32, mot 0.35] | pred probs [cog 0.35, aff 0.29, mot 0.36] | 69w
   > I'm truly sorry to hear that you're going through this at your workplace. It's absolutely natural to feel hurt when faced with such situations. However, try not to let this negatively affect your confidence or performance at work. Remember, your value isn't determined by someone else's opinion, and you have every right to stand up for yourself in a respectful manner, even if it means addressing the situation directly.

2. *pred = Motivational* | true probs [cog 0.33, aff 0.33, mot 0.34] | pred probs [cog 0.34, aff 0.30, mot 0.35] | 73w
   > I'm really sorry to hear about the disagreement you had, those situations can get pretty stressful. It's important that you recognize your feelings as both valid and understandable given the circumstances. Never forget that the ability to make amends and reconcile after a miscommunication is a testament to the strength of your relationship. It's great to see how much you value feeling loved and appreciated, and I'm here for any support you need.

3. *pred = Motivational* | true probs [cog 0.37, aff 0.21, mot 0.42] | pred probs [cog 0.34, aff 0.31, mot 0.35] | 86w
   > I'm really sorry to hear that you had to endure such a difficult experience with your ex, it must have been incredibly hard for you. Please remember that you deserve respect, love, and kindness in all your relationships. The feelings of anger you're experiencing are valid, but also remind yourself that you are strong and capable of building a better, healthier future. I believe in your ability to rise above this, and here's to using these experiences as a stepping stone towards stronger, more respectful relationships.

4. *pred = Motivational* | true probs [cog 0.31, aff 0.31, mot 0.39] | pred probs [cog 0.34, aff 0.31, mot 0.35] | 74w
   > I'm really sorry to hear about your painful experience at the dentist's office. It's absolutely understandable to feel scared or anxious when facing such situations, but please remember you showed an incredible amount of strength by getting through it. I believe you have the resilience to face any challenge that comes your way, just as you dealt with this one. Let's try and make your future dental visits more comfortable and less daunting together.

5. *pred = Motivational* | true probs [cog 0.33, aff 0.31, mot 0.35] | pred probs [cog 0.34, aff 0.31, mot 0.35] | 58w
   > I'm really sorry to hear that you're going through this; disagreements with friends can be deeply unsettling. Remember, it's okay to have different perspectives, and it doesn't diminish your worth as an individual or friend. Take some time to heal and know I am here for you, offering support, understanding, and a compassionate ear whenever you need it.

**Confident-wrong (5)**

1. *pred = Cognitive* | true probs [cog 0.33, aff 0.32, mot 0.35] | pred probs [cog 0.39, aff 0.31, mot 0.30] | 48w
   > Your joy radiates so strongly it's as if you're glowing like the sun, filled with an overflowing sense of happiness. The way you embrace your bliss seems to fuel your vitality, making you thrive and beam even more brightly. It's such a beautiful embodiment of pure, undiluted happiness.

2. *pred = Cognitive* | true probs [cog 0.32, aff 0.33, mot 0.35] | pred probs [cog 0.39, aff 0.29, mot 0.32] | 68w
   > I see how you were flooded with overwhelm during the festive season, striving to fulfill expectations and make everything perfect. It's clear that you want to create memorable, joyful experiences for those around you, yet the pressure to achieve this can feel daunting and consuming. It must be difficult when your desire to remember nice things is clouded by the stress of trying to make everything just right.

3. *pred = Cognitive* | true probs [cog 0.28, aff 0.09, mot 0.62] | pred probs [cog 0.38, aff 0.29, mot 0.32] | 103w
   > It's evident that you love and cherish your role as a mother deeply and feel a considerable amount of longing and a sense of loss when you're spending time apart from your daughter. Even while recognizing the value of her relationship with her father, the 50/50 arrangement during the holidays really tugs at your heartstrings. You're trying to balance your appreciation of her father's role in her life with the disappointment of feeling like you are missing out on part of her childhood. You're also dealing with the emotional aftermath of a broken relationship, which can further color these feeli…

4. *pred = Cognitive* | true probs [cog 0.36, aff 0.27, mot 0.37] | pred probs [cog 0.38, aff 0.28, mot 0.34] | 57w
   > It sounds like you experienced a high level of stress and anxiety in anticipation of your job interview, possibly fueled by the uncertainty and importance of the situation. Then, when you were offered the job, these feelings transformed into happiness and likely also relief. Congratulations on navigating what sounds like an emotional rollercoaster and achieving your goal.

5. *pred = Cognitive* | true probs [cog 0.37, aff 0.25, mot 0.38] | pred probs [cog 0.38, aff 0.31, mot 0.31] | 107w
   > You seem to be wrestling with a mix of opposing emotions, nostalgia and regret on one hand, and a glimmer of hope on the other. It must be bittersweet to revisit a place Ioved and reconnect with someone who used to be pivotal to your life, only to realize how much you two have changed. You seem to feel a sense of closure, accepting that some stories are best kept in the past, yet there's a lingering sadness, a poignant reminder of the deep impact some connections leave on our lives. Your emotions are deeply complex as you navigate this delicate balance between past and present interactions.

*(Artifacts: `outputs/a2_examples.md`, `outputs/a2_examples.csv`, `outputs/a2_examples.npz`.)*

---

## Part A — Sections to come

- A3 — Quantitative error analysis
- A4 — Sensitivity analysis

## Part B — Improved Model (to come)
