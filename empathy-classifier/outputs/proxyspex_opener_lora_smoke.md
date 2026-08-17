# ProxySPEX on opener — lora_winner

Checkpoint: `/Users/stolk/github/LLMES/empathy-classifier/src/../models/lora_winner_seed9.pt`

Config: {'model': 'lora_winner', 'k_per_cell': 1, 'n_opener_words': 10, 'n_masks': 50, 'k_order_max': 3, 'batch_size': 8, 'n_eval_proxy': 4096, 'out_tag': 'lora_smoke'}


## Top tokens by aggregated |F| (interaction strength) per true class


### True = Cognitive  (2 examples)

Interaction-order distribution (top-10 per example): {1: 1, 2: 6, 3: 13}
Top tokens: `I'm` (0.06), `this` (0.06), `that` (0.05), `way,` (0.05), `It's` (0.05), `feeling` (0.05), `experienced` (0.04), `you` (0.04), `you're` (0.04), `really` (0.03), `of` (0.03), `sorry` (0.02), `evident` (0.02), `sense` (0.02), `uneasiness` (0.02)
Top positions: pos5:8, pos2:6, pos7:6, pos0:6, pos9:5, pos3:5, pos1:4, pos8:4, pos4:4, pos6:4

### True = Affective  (2 examples)

Interaction-order distribution (top-10 per example): {1: 2, 2: 5, 3: 13}
Top tokens: `you` (0.09), `this` (0.06), `I` (0.05), `for` (0.05), `distressing` (0.04), `have` (0.04), `during` (0.04), `your` (0.04), `and` (0.03), `extremely` (0.03), `must` (0.03), `family` (0.02), `been` (0.02), `truly` (0.02), `It` (0.01)
Top positions: pos7:7, pos5:7, pos4:6, pos0:5, pos6:5, pos3:5, pos1:4, pos9:4, pos8:4, pos2:4

### True = Motivational  (2 examples)

Interaction-order distribution (top-10 per example): {1: 0, 2: 3, 3: 17}
Top tokens: `this` (0.09), `hear` (0.08), `to` (0.07), `I'm` (0.07), `going` (0.05), `through` (0.05), `your` (0.04), `you're` (0.04), `so` (0.03), `about` (0.03), `truly` (0.03), `elated` (0.02), `wonderful` (0.02), `sorry` (0.01), `that` (0.01)
Top positions: pos8:7, pos4:7, pos9:7, pos3:6, pos5:6, pos0:6, pos1:5, pos7:5, pos6:5, pos2:3

## Per-example top-3 interactions (sample)


### idx=635 [Cog/corre] true=Cognitive
opener: `It's evident that you initially experienced a sense of uneasiness`
  - F=+0.013  |T|=2  tokens=['evident', 'experienced']  (positions [1, 5])
  - F=-0.013  |T|=3  tokens=['that', 'sense', 'uneasiness']  (positions [2, 7, 9])
  - F=+0.012  |T|=2  tokens=['that', 'of']  (positions [2, 8])

### idx=60 [Cog/wrong] true=Cognitive
opener: `I'm really sorry that you're feeling this way, but I'm`
  - F=+0.022  |T|=3  tokens=['really', 'this', "I'm"]  (positions [1, 6, 9])
  - F=-0.014  |T|=2  tokens=['way,', "I'm"]  (positions [7, 9])
  - F=+0.013  |T|=3  tokens=["you're", 'this', 'way,']  (positions [4, 6, 7])

### idx=208 [Aff/corre] true=Affective
opener: `I truly feel for you and your family during this`
  - F=-0.020  |T|=3  tokens=['I', 'truly', 'this']  (positions [0, 1, 9])
  - F=-0.014  |T|=3  tokens=['you', 'family', 'during']  (positions [4, 7, 8])
  - F=-0.013  |T|=1  tokens=['your']  (positions [6])

### idx=919 [Aff/wrong] true=Affective
opener: `It must have been extremely distressing for you to return`
  - F=-0.013  |T|=3  tokens=['extremely', 'distressing', 'you']  (positions [4, 5, 7])
  - F=+0.013  |T|=1  tokens=['must']  (positions [1])
  - F=+0.012  |T|=3  tokens=['It', 'have', 'you']  (positions [0, 2, 7])

### idx=510 [Mot/corre] true=Motivational
opener: `I'm truly sorry to hear that you're going through this`
  - F=+0.017  |T|=3  tokens=['truly', 'to', 'through']  (positions [1, 3, 8])
  - F=+0.015  |T|=3  tokens=['hear', 'going', 'this']  (positions [4, 7, 9])
  - F=-0.014  |T|=2  tokens=['sorry', 'hear']  (positions [2, 4])

### idx=499 [Mot/wrong] true=Motivational
opener: `I'm so elated to hear this wonderful news about your`
  - F=+0.012  |T|=3  tokens=['so', 'this', 'your']  (positions [1, 5, 9])
  - F=-0.012  |T|=2  tokens=['so', 'elated']  (positions [1, 2])
  - F=-0.012  |T|=3  tokens=["I'm", 'to', 'your']  (positions [0, 3, 9])