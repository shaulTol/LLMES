# ProxySPEX on opener — linear_baseline

Checkpoint: `/Users/stolk/github/LLMES/empathy-classifier/src/../models/baseline_v1.pt`

Config: {'model': 'baseline', 'k_per_cell': 2, 'n_opener_words': 10, 'n_masks': 100, 'k_order_max': 3, 'batch_size': 16, 'n_eval_proxy': 4096, 'out_tag': 'baseline_smoke'}


## Top tokens by aggregated |F| (interaction strength) per true class


### True = Cognitive  (4 examples)

Interaction-order distribution (top-10 per example): {1: 2, 2: 13, 3: 25}
Top tokens: `this` (0.15), `way,` (0.11), `really` (0.09), `I'm` (0.08), `sorry` (0.07), `that` (0.06), `feeling` (0.06), `a` (0.06), `you're` (0.06), `can` (0.06), `you` (0.05), `emotional` (0.05), `how` (0.05), `I` (0.04), `real` (0.04)
Top positions: pos1:14, pos9:14, pos7:13, pos3:11, pos0:9, pos6:9, pos4:9, pos5:8, pos2:8, pos8:8

### True = Affective  (4 examples)

Interaction-order distribution (top-10 per example): {1: 4, 2: 10, 3: 26}
Top tokens: `you` (0.10), `I` (0.10), `have` (0.09), `and` (0.07), `must` (0.06), `this` (0.06), `joy` (0.06), `distressing` (0.04), `for` (0.04), `an` (0.04), `experienced` (0.04), `during` (0.04), `that` (0.04), `warmth` (0.04), `extremely` (0.04)
Top positions: pos7:15, pos9:13, pos0:12, pos1:10, pos5:10, pos4:10, pos3:9, pos6:8, pos2:8, pos8:7

### True = Motivational  (4 examples)

Interaction-order distribution (top-10 per example): {1: 3, 2: 10, 3: 27}
Top tokens: `this` (0.12), `I'm` (0.08), `hear` (0.07), `I` (0.06), `witnessing` (0.06), `You` (0.06), `through` (0.05), `your` (0.05), `to` (0.05), `going` (0.05), `heavy` (0.05), `phase` (0.05), `so` (0.05), `sense` (0.04), `have` (0.04)
Top positions: pos0:15, pos9:14, pos5:12, pos1:11, pos4:10, pos7:9, pos2:9, pos3:8, pos8:8, pos6:8

## Per-example top-3 interactions (sample)


### idx=635 [Cog/corre] true=Cognitive
opener: `It's evident that you initially experienced a sense of uneasiness`
  - F=+0.013  |T|=2  tokens=['evident', 'experienced']  (positions [1, 5])
  - F=-0.012  |T|=3  tokens=['that', 'sense', 'uneasiness']  (positions [2, 7, 9])
  - F=+0.012  |T|=2  tokens=['that', 'of']  (positions [2, 8])

### idx=1159 [Cog/corre] true=Cognitive
opener: `I can understand how this experience was a real emotional`
  - F=+0.019  |T|=3  tokens=['can', 'this', 'a']  (positions [1, 4, 7])
  - F=-0.015  |T|=3  tokens=['I', 'how', 'experience']  (positions [0, 3, 5])
  - F=-0.015  |T|=2  tokens=['can', 'emotional']  (positions [1, 9])

### idx=60 [Cog/wrong] true=Cognitive
opener: `I'm really sorry that you're feeling this way, but I'm`
  - F=+0.022  |T|=3  tokens=['really', 'this', "I'm"]  (positions [1, 6, 9])
  - F=-0.014  |T|=2  tokens=['way,', "I'm"]  (positions [7, 9])
  - F=+0.013  |T|=2  tokens=['that', 'feeling']  (positions [3, 5])

### idx=1163 [Cog/wrong] true=Cognitive
opener: `I'm really sorry to hear that you're feeling this way,`
  - F=-0.017  |T|=2  tokens=['really', 'sorry']  (positions [1, 2])
  - F=+0.016  |T|=2  tokens=['sorry', 'that']  (positions [2, 5])
  - F=-0.015  |T|=2  tokens=['hear', 'way,']  (positions [4, 9])

### idx=208 [Aff/corre] true=Affective
opener: `I truly feel for you and your family during this`
  - F=-0.021  |T|=3  tokens=['I', 'truly', 'this']  (positions [0, 1, 9])
  - F=-0.013  |T|=2  tokens=['and', 'this']  (positions [5, 9])
  - F=+0.012  |T|=2  tokens=['for', 'your']  (positions [3, 6])

### idx=750 [Aff/corre] true=Affective
opener: `I can almost feel the warmth and joy you experienced`
  - F=-0.014  |T|=1  tokens=['feel']  (positions [3])
  - F=-0.014  |T|=3  tokens=['I', 'warmth', 'joy']  (positions [0, 5, 7])
  - F=-0.012  |T|=2  tokens=['I', 'experienced']  (positions [0, 9])

### idx=919 [Aff/wrong] true=Affective
opener: `It must have been extremely distressing for you to return`
  - F=-0.014  |T|=3  tokens=['extremely', 'distressing', 'you']  (positions [4, 5, 7])
  - F=+0.013  |T|=1  tokens=['must']  (positions [1])
  - F=+0.012  |T|=3  tokens=['It', 'have', 'you']  (positions [0, 2, 7])

### idx=1012 [Aff/wrong] true=Affective
opener: `What an incredibly powerful and moving experience that must have`
  - F=-0.015  |T|=3  tokens=['What', 'that', 'must']  (positions [0, 7, 8])
  - F=-0.013  |T|=3  tokens=['incredibly', 'experience', 'that']  (positions [2, 6, 7])
  - F=-0.012  |T|=3  tokens=['incredibly', 'that', 'have']  (positions [2, 7, 9])

### idx=510 [Mot/corre] true=Motivational
opener: `I'm truly sorry to hear that you're going through this`
  - F=+0.016  |T|=3  tokens=['truly', 'to', 'through']  (positions [1, 3, 8])
  - F=-0.014  |T|=3  tokens=["I'm", 'truly', "you're"]  (positions [0, 1, 6])
  - F=-0.014  |T|=3  tokens=['hear', 'that', 'going']  (positions [4, 5, 7])

### idx=1118 [Mot/corre] true=Motivational
opener: `I understand how challenging this new parenting phase can be`
  - F=+0.015  |T|=2  tokens=['I', 'new']  (positions [0, 5])
  - F=-0.014  |T|=3  tokens=['challenging', 'this', 'phase']  (positions [3, 4, 7])
  - F=-0.014  |T|=3  tokens=['understand', 'can', 'be']  (positions [1, 8, 9])

### idx=499 [Mot/wrong] true=Motivational
opener: `I'm so elated to hear this wonderful news about your`
  - F=+0.014  |T|=3  tokens=['elated', 'about', 'your']  (positions [2, 8, 9])
  - F=+0.014  |T|=3  tokens=['so', 'this', 'your']  (positions [1, 5, 9])
  - F=-0.014  |T|=3  tokens=["I'm", 'to', 'your']  (positions [0, 3, 9])

### idx=30 [Mot/wrong] true=Motivational
opener: `You must have felt a heavy sense of sadness witnessing`
  - F=+0.019  |T|=3  tokens=['You', 'felt', 'sense']  (positions [0, 3, 6])
  - F=+0.017  |T|=2  tokens=['You', 'witnessing']  (positions [0, 9])
  - F=-0.014  |T|=3  tokens=['have', 'felt', 'heavy']  (positions [2, 3, 5])