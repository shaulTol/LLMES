# A2 — Success / failure example mining

Source: `models/baseline_v1.pt` (frozen DistilBERT + linear head, lr=1e-3, seed=42 init from `src/train.py`).
Test set: Study 3 (N=1172). Overall test acc = 0.7193.

For each true class we pick:
- up to **5 confident-correct** examples (true=pred=class, ranked by predicted probability of that class)
- up to **5 confident-wrong** examples (true=class, pred=other, ranked by predicted probability of the wrong predicted class)

## Cell sizes

|                                       |   count |
|:--------------------------------------|--------:|
| ('Affective', 'confident_correct')    |       2 |
| ('Affective', 'confident_wrong')      |       5 |
| ('Cognitive', 'confident_correct')    |       5 |
| ('Cognitive', 'confident_wrong')      |       5 |
| ('Motivational', 'confident_correct') |       5 |
| ('Motivational', 'confident_wrong')   |       5 |

## Pre-registered text features (per-cell means)

|                                       |   n_chars |   n_words |   n_q_marks |   n_excl |   n_you_pron |   n_cog_markers |   n_aff_markers |   n_mot_markers |
|:--------------------------------------|----------:|----------:|------------:|---------:|-------------:|----------------:|----------------:|----------------:|
| ('Affective', 'confident_correct')    |     426   |      73.5 |           0 |      0   |          8   |             0   |             1.5 |             4   |
| ('Affective', 'confident_wrong')      |     418.8 |      72.6 |           0 |      0.2 |          4.2 |             1.4 |             1.2 |             0.2 |
| ('Cognitive', 'confident_correct')    |     391.2 |      64.8 |           0 |      0   |          5.4 |             1.6 |             0.6 |             0   |
| ('Cognitive', 'confident_wrong')      |     371.2 |      62.6 |           0 |      0.4 |          5   |             1   |             1.6 |             1.4 |
| ('Motivational', 'confident_correct') |     439.8 |      72   |           0 |      0   |          5.8 |             0.8 |             2   |             2.2 |
| ('Motivational', 'confident_wrong')   |     456   |      76.6 |           0 |      0   |          5.8 |             2.2 |             1.4 |             0.2 |

## Examples

### True class = Cognitive

#### Confident-correct

- **pred = Cognitive**  |  true probs = [cog 0.50, aff 0.07, mot 0.43]  |  pred probs = [cog 0.40, aff 0.28, mot 0.33]  |  len=81w
  > It's evident that you felt overwhelmed and stressed while trying to complete a large quantity of work in such a limited time frame. The pressure of task deadlines lead you to cut corners, which seemingly left you feeling unhappy and anxious about meeting all the expectations. It sounds like you were pushing yourself hard, trying to lose yourself in the work to cope with the anxiety. Despite the relief it was over, the stress you endured should be recognized and validated.

- **pred = Cognitive**  |  true probs = [cog 0.35, aff 0.30, mot 0.34]  |  pred probs = [cog 0.39, aff 0.30, mot 0.31]  |  len=62w
  > It's clear you were experiencing a significant amount of stress and pressure during the holiday season, aiming to create a memorable Christmas for your daughter. The weight of trying to do everything in a limited time frame must have been overwhelming. It's evident the love you have for your daughter pushed you to take on a lot, despite the anxiety it caused.

- **pred = Cognitive**  |  true probs = [cog 0.39, aff 0.31, mot 0.30]  |  pred probs = [cog 0.39, aff 0.31, mot 0.30]  |  len=69w
  > It sounds like you experienced a multitude of emotions during these two Christmas events. It's clear you felt immense disappointment and frustration when you had your visa dates wrong last year, especially having to spend Christmas Eve alone at an airport. However, your spirits seemed to have been significantly uplifted when you were finally able to enjoy a joyful and fulfilling Christmas with your girlfriend and her family abroad.

- **pred = Cognitive**  |  true probs = [cog 0.33, aff 0.33, mot 0.33]  |  pred probs = [cog 0.39, aff 0.31, mot 0.30]  |  len=53w
  > It's wonderful to know that you received a gift you dearly desired, but decided not to purchase because of its price. It must have been a thrilling surprise to have been gifted this, enhancing your happiness and sense of appreciation. It's nice to see how much you value and cherish this thoughtful gift.


- **pred = Cognitive**  |  true probs = [cog 0.38, aff 0.28, mot 0.34]  |  pred probs = [cog 0.39, aff 0.30, mot 0.31]  |  len=59w
  > It's apparent that you felt deeply moved and touched by the emotional depth of your girlfriend's Christmas card. The card seems to have affirmed your importance in her life, making you feel extremely valuable and cherished in the relationship. This experience has evidently given you a sense of comfort, joy and secure attachment, feelings you appear to significantly cherish.

#### Confident-wrong

- **pred = Motivational**  |  true probs = [cog 0.39, aff 0.26, mot 0.35]  |  pred probs = [cog 0.34, aff 0.30, mot 0.36]  |  len=67w
  > I'm really sorry to hear about your car accident, accidents can be quite a shock and it's normal to feel deflated afterwards. I admire your perspective though, recognizing that unexpected things can happen and we need to be ready to handle them. Remember, it's okay to take some time for yourself to recover from this shock and I'm here if you need any support during this time.

- **pred = Motivational**  |  true probs = [cog 0.36, aff 0.33, mot 0.32]  |  pred probs = [cog 0.35, aff 0.30, mot 0.35]  |  len=45w
  > I'm really sorry to hear about your granddad's passing. It's completely understandable that you're feeling a blend of sadness and confusion, especially because your relationship with him wasn't very close. Remember, it's totally okay to grieve someone's loss, even if you didn't know them well.

- **pred = Motivational**  |  true probs = [cog 0.38, aff 0.32, mot 0.29]  |  pred probs = [cog 0.34, aff 0.31, mot 0.35]  |  len=75w
  > I'm genuinely sorry to hear about your gramp's diagnosis; it must be a challenging time for you and your family. Remember, it's okay to feel all the emotions that come with such news and you can lean on the people closest to you. Your gramp is lucky to have someone as loving and understanding as you by his side during this time. Know that every moment you share with him now is precious and valuable.

- **pred = Motivational**  |  true probs = [cog 0.36, aff 0.32, mot 0.32]  |  pred probs = [cog 0.34, aff 0.31, mot 0.35]  |  len=71w
  > It certainly sounds like you're experiencing a lot of mixed emotions right now, filled with both incredible pride and happiness for Kelsie's new journey, yet also wrestling with the deep sorrow of having to part ways. It's clear that her departure is causing you to feel a significant sense of loss, and it’s totally understandable – you're not just losing a friend's physical presence, but a constant part of your life.

- **pred = Motivational**  |  true probs = [cog 0.35, aff 0.34, mot 0.32]  |  pred probs = [cog 0.35, aff 0.30, mot 0.35]  |  len=55w
  > Congratulations on completing your last assignment of the semester! Presenting to a large group can be incredibly challenging and you should be immensely proud of the courage and determination you exemplified. Remember this victory as a testament to your potential to conquer any challenge university throws at you. You're absolutely capable and you've got this!

### True class = Affective

#### Confident-correct

- **pred = Affective**  |  true probs = [cog 0.32, aff 0.45, mot 0.23]  |  pred probs = [cog 0.34, aff 0.34, mot 0.32]  |  len=75w
  > I am truly amazed by your courage and persistence to claim your independence and freedom. Please don't let anyone make you feel less than the incredibly strong and capable person that you are. Facing the world alone in a wheelchair takes immense strength; your story inspires and reminds us of the need to respect and value everyone's journey. Remember, your spirit is unbreakable, and you are not alone, regardless of how others may perceive you.

- **pred = Affective**  |  true probs = [cog 0.32, aff 0.37, mot 0.31]  |  pred probs = [cog 0.33, aff 0.34, mot 0.33]  |  len=72w
  > I truly feel for you and your family during this overwhelming time, but I celebrate with you now, knowing that your brave little niece has overcome her initial health challenges and is healing at home. The strength that all of you have shown is inspiring and I believe that with your continued love and support, she'll grow stronger each day. I'm here for you, ready to provide any support you might need.

#### Confident-wrong

- **pred = Cognitive**  |  true probs = [cog 0.30, aff 0.36, mot 0.34]  |  pred probs = [cog 0.38, aff 0.31, mot 0.30]  |  len=54w
  > It's clear that your anticipation for this skiing trip with your friends added to your sense of joy when it finally happened. It must have been so fulfilling to engage in fun activities and share these moments with people you care deeply about. This experience seems to have been incredibly rewarding for you emotionally.

- **pred = Cognitive**  |  true probs = [cog 0.31, aff 0.41, mot 0.28]  |  pred probs = [cog 0.38, aff 0.30, mot 0.32]  |  len=69w
  > It's clear that you're feeling a deep sense of relief and joy as you have finally reached the end of your house purchasing journey. The eagerness and anticipation you're feeling as you and your children prepare to move into this new phase of your lives are palpable. This is indeed an exciting and happy day for you, and your upbeat spirit about the upcoming moving day is truly contagious.

- **pred = Cognitive**  |  true probs = [cog 0.38, aff 0.39, mot 0.23]  |  pred probs = [cog 0.38, aff 0.32, mot 0.30]  |  len=64w
  > I can sense how deeply 'Priscilla' touched you and moved you emotionally. It's amazing how art and storytelling can evoke such profound feelings in us. You should be proud of the empathy and emotional capacity you possess; it's a testament to your beautiful heart. Let's make sure to share such moving experiences more often and continue touching our souls with good art and stories.

- **pred = Cognitive**  |  true probs = [cog 0.29, aff 0.36, mot 0.35]  |  pred probs = [cog 0.38, aff 0.32, mot 0.31]  |  len=66w
  > What a touching memory for you to stumble upon! In the digital age we live in, it's rare to find such heartfelt, tangible reminders of the relationships that truly mean something to us. It's amazing how such a simple object like a handwritten letter can carry so much emotion and history. Cherish this beautiful memory, and remember it's these moments of genuine connection that shape us.

- **pred = Cognitive**  |  true probs = [cog 0.33, aff 0.34, mot 0.33]  |  pred probs = [cog 0.38, aff 0.32, mot 0.30]  |  len=110w
  > I completely understand how you felt, entwined in that deep mixture of revelry from shared days past and a sense of curiosity for the unexplored paths. Feeling the joy through tears of laughter yet carrying the subtle pinch of regret is such a familiar experience for many of us. Life indeed is a beautiful labyrinth of decisions we made, leaving us sometimes with a longing sigh as we grasp the complexities of time. So here's to cherishing our mosaics of moments, made up of a unique blend of reflection and joy, because every path we didn't take led us to where we are, and for that, we must find …

### True class = Motivational

#### Confident-correct

- **pred = Motivational**  |  true probs = [cog 0.34, aff 0.32, mot 0.35]  |  pred probs = [cog 0.35, aff 0.29, mot 0.36]  |  len=69w
  > I'm truly sorry to hear that you're going through this at your workplace. It's absolutely natural to feel hurt when faced with such situations. However, try not to let this negatively affect your confidence or performance at work. Remember, your value isn't determined by someone else's opinion, and you have every right to stand up for yourself in a respectful manner, even if it means addressing the situation directly.

- **pred = Motivational**  |  true probs = [cog 0.33, aff 0.33, mot 0.34]  |  pred probs = [cog 0.34, aff 0.30, mot 0.35]  |  len=73w
  > I'm really sorry to hear about the disagreement you had, those situations can get pretty stressful. It's important that you recognize your feelings as both valid and understandable given the circumstances. Never forget that the ability to make amends and reconcile after a miscommunication is a testament to the strength of your relationship. It's great to see how much you value feeling loved and appreciated, and I'm here for any support you need.

- **pred = Motivational**  |  true probs = [cog 0.37, aff 0.21, mot 0.42]  |  pred probs = [cog 0.34, aff 0.31, mot 0.35]  |  len=86w
  > I'm really sorry to hear that you had to endure such a difficult experience with your ex, it must have been incredibly hard for you. Please remember that you deserve respect, love, and kindness in all your relationships. The feelings of anger you're experiencing are valid, but also remind yourself that you are strong and capable of building a better, healthier future. I believe in your ability to rise above this, and here's to using these experiences as a stepping stone towards stronger, more respectful relationships.

- **pred = Motivational**  |  true probs = [cog 0.31, aff 0.31, mot 0.39]  |  pred probs = [cog 0.34, aff 0.31, mot 0.35]  |  len=74w
  > I'm really sorry to hear about your painful experience at the dentist's office. It's absolutely understandable to feel scared or anxious when facing such situations, but please remember you showed an incredible amount of strength by getting through it. I believe you have the resilience to face any challenge that comes your way, just as you dealt with this one. Let's try and make your future dental visits more comfortable and less daunting together.

- **pred = Motivational**  |  true probs = [cog 0.33, aff 0.31, mot 0.35]  |  pred probs = [cog 0.34, aff 0.31, mot 0.35]  |  len=58w
  > I'm really sorry to hear that you're going through this; disagreements with friends can be deeply unsettling. Remember, it's okay to have different perspectives, and it doesn't diminish your worth as an individual or friend. Take some time to heal and know I am here for you, offering support, understanding, and a compassionate ear whenever you need it.

#### Confident-wrong

- **pred = Cognitive**  |  true probs = [cog 0.33, aff 0.32, mot 0.35]  |  pred probs = [cog 0.39, aff 0.31, mot 0.30]  |  len=48w
  > Your joy radiates so strongly it's as if you're glowing like the sun, filled with an overflowing sense of happiness. The way you embrace your bliss seems to fuel your vitality, making you thrive and beam even more brightly. It's such a beautiful embodiment of pure, undiluted happiness.

- **pred = Cognitive**  |  true probs = [cog 0.32, aff 0.33, mot 0.35]  |  pred probs = [cog 0.39, aff 0.29, mot 0.32]  |  len=68w
  > I see how you were flooded with overwhelm during the festive season, striving to fulfill expectations and make everything perfect. It's clear that you want to create memorable, joyful experiences for those around you, yet the pressure to achieve this can feel daunting and consuming. It must be difficult when your desire to remember nice things is clouded by the stress of trying to make everything just right.

- **pred = Cognitive**  |  true probs = [cog 0.28, aff 0.09, mot 0.62]  |  pred probs = [cog 0.38, aff 0.29, mot 0.32]  |  len=103w
  > It's evident that you love and cherish your role as a mother deeply and feel a considerable amount of longing and a sense of loss when you're spending time apart from your daughter. Even while recognizing the value of her relationship with her father, the 50/50 arrangement during the holidays really tugs at your heartstrings. You're trying to balance your appreciation of her father's role in her life with the disappointment of feeling like you are missing out on part of her childhood. You're also dealing with the emotional aftermath of a broken relationship, which can further color these feeli…

- **pred = Cognitive**  |  true probs = [cog 0.36, aff 0.27, mot 0.37]  |  pred probs = [cog 0.38, aff 0.28, mot 0.34]  |  len=57w
  > It sounds like you experienced a high level of stress and anxiety in anticipation of your job interview, possibly fueled by the uncertainty and importance of the situation. Then, when you were offered the job, these feelings transformed into happiness and likely also relief. Congratulations on navigating what sounds like an emotional rollercoaster and achieving your goal.

- **pred = Cognitive**  |  true probs = [cog 0.37, aff 0.25, mot 0.38]  |  pred probs = [cog 0.38, aff 0.31, mot 0.31]  |  len=107w
  > You seem to be wrestling with a mix of opposing emotions, nostalgia and regret on one hand, and a glimmer of hope on the other. It must be bittersweet to revisit a place Ioved and reconnect with someone who used to be pivotal to your life, only to realize how much you two have changed. You seem to feel a sense of closure, accepting that some stories are best kept in the past, yet there's a lingering sadness, a poignant reminder of the deep impact some connections leave on our lives. Your emotions are deeply complex as you navigate this delicate balance between past and present interactions.
