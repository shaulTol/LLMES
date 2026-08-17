# ProxySPEX on opener — roberta_winner

Checkpoint: `/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/src/../models/roberta_winner_seed0.pt`

Config: {'model': 'roberta_winner', 'k_per_cell': 10, 'n_opener_words': 10, 'n_masks': 256, 'k_order_max': 3, 'batch_size': 32, 'n_eval_proxy': 4096, 'out_tag': 'roberta_winner'}


## Top tokens by aggregated |F| (interaction strength) per true class


### True = Cognitive  (20 examples)

Interaction-order distribution (top-10 per example): {1: 10, 2: 62, 3: 128}
Top tokens: `that` (0.32), `really` (0.31), `I'm` (0.31), `I` (0.28), `sorry` (0.25), `this` (0.23), `feeling` (0.21), `your` (0.20), `and` (0.20), `you` (0.20), `you're` (0.19), `can` (0.15), `It's` (0.13), `to` (0.12), `hear` (0.11)
Top positions: pos9:58, pos7:56, pos5:54, pos0:54, pos6:53, pos1:52, pos2:52, pos4:49, pos3:45, pos8:45

### True = Affective  (20 examples)

Interaction-order distribution (top-10 per example): {1: 16, 2: 64, 3: 120}
Top tokens: `I` (0.33), `the` (0.28), `you` (0.28), `can` (0.22), `and` (0.21), `to` (0.20), `feel` (0.14), `your` (0.14), `must` (0.13), `of` (0.11), `that` (0.11), `have` (0.09), `completely` (0.09), `truly` (0.09), `heartwarming` (0.09)
Top positions: pos0:59, pos2:55, pos9:54, pos6:54, pos1:51, pos3:51, pos5:50, pos7:50, pos8:42, pos4:38

### True = Motivational  (20 examples)

Interaction-order distribution (top-10 per example): {1: 18, 2: 39, 3: 143}
Top tokens: `that` (0.31), `I'm` (0.29), `hear` (0.25), `can` (0.25), `I` (0.23), `you're` (0.23), `you` (0.19), `sorry` (0.19), `to` (0.18), `going` (0.17), `really` (0.16), `this` (0.14), `understand` (0.14), `through` (0.14), `your` (0.13)
Top positions: pos5:62, pos9:61, pos1:60, pos0:56, pos2:51, pos7:50, pos4:48, pos8:48, pos3:47, pos6:42

## Per-example top-3 interactions (sample)


### idx=635 [Cog/corre] true=Cognitive
opener: `It's evident that you initially experienced a sense of uneasiness`
  - F=+0.013  |T|=2  tokens=['evident', 'experienced']  (positions [1, 5])
  - F=-0.012  |T|=3  tokens=["It's", 'that', 'you']  (positions [0, 2, 3])
  - F=+0.011  |T|=2  tokens=['that', 'of']  (positions [2, 8])

### idx=1159 [Cog/corre] true=Cognitive
opener: `I can understand how this experience was a real emotional`
  - F=+0.018  |T|=3  tokens=['can', 'this', 'a']  (positions [1, 4, 7])
  - F=-0.016  |T|=2  tokens=['can', 'emotional']  (positions [1, 9])
  - F=-0.014  |T|=3  tokens=['I', 'how', 'experience']  (positions [0, 3, 5])

### idx=379 [Cog/corre] true=Cognitive
opener: `I can see how you might be feeling frustrated and`
  - F=+0.017  |T|=3  tokens=['can', 'be', 'and']  (positions [1, 6, 9])
  - F=+0.015  |T|=2  tokens=['frustrated', 'and']  (positions [8, 9])
  - F=+0.015  |T|=3  tokens=['I', 'feeling', 'frustrated']  (positions [0, 7, 8])

### idx=490 [Cog/corre] true=Cognitive
opener: `It sounds like you were already thrilled about your favorite`
  - F=+0.019  |T|=3  tokens=['sounds', 'you', 'your']  (positions [1, 3, 8])
  - F=+0.017  |T|=3  tokens=['It', 'like', 'about']  (positions [0, 2, 7])
  - F=+0.017  |T|=2  tokens=['you', 'favorite']  (positions [3, 9])

### idx=764 [Cog/corre] true=Cognitive
opener: `It's completely understandable that you may have experienced a mix`
  - F=-0.018  |T|=2  tokens=['that', 'experienced']  (positions [3, 7])
  - F=+0.017  |T|=2  tokens=['may', 'have']  (positions [5, 6])
  - F=-0.016  |T|=1  tokens=['experienced']  (positions [7])

### idx=523 [Cog/corre] true=Cognitive
opener: `I completely understand why you'd feel angry and upset; being`
  - F=-0.017  |T|=3  tokens=['why', 'feel', 'and']  (positions [3, 5, 7])
  - F=+0.013  |T|=2  tokens=['I', 'being']  (positions [0, 9])
  - F=-0.013  |T|=3  tokens=['I', 'angry', 'and']  (positions [0, 6, 7])

### idx=1035 [Cog/corre] true=Cognitive
opener: `I can really identify with the stress and nervousness you`
  - F=-0.018  |T|=2  tokens=['can', 'the']  (positions [1, 5])
  - F=+0.017  |T|=2  tokens=['really', 'identify']  (positions [2, 3])
  - F=-0.014  |T|=3  tokens=['I', 'identify', 'you']  (positions [0, 3, 9])

### idx=1008 [Cog/corre] true=Cognitive
opener: `It must be an emotional time for you, observing your`
  - F=+0.018  |T|=3  tokens=['time', 'for', 'observing']  (positions [5, 6, 8])
  - F=-0.016  |T|=3  tokens=['an', 'emotional', 'for']  (positions [3, 4, 6])
  - F=+0.015  |T|=3  tokens=['must', 'be', 'your']  (positions [1, 2, 9])

### idx=653 [Cog/corre] true=Cognitive
opener: `It's evident that you experienced a rollercoaster of emotions during`
  - F=+0.018  |T|=2  tokens=['rollercoaster', 'emotions']  (positions [6, 8])
  - F=+0.017  |T|=2  tokens=['rollercoaster', 'of']  (positions [6, 7])
  - F=-0.015  |T|=1  tokens=['rollercoaster']  (positions [6])

### idx=335 [Cog/corre] true=Cognitive
opener: `It sounds like your trip to Edinburgh was not only`
  - F=+0.014  |T|=3  tokens=['It', 'Edinburgh', 'not']  (positions [0, 6, 8])
  - F=-0.014  |T|=2  tokens=['to', 'was']  (positions [5, 7])
  - F=+0.013  |T|=2  tokens=['trip', 'not']  (positions [4, 8])

### idx=60 [Cog/wrong] true=Cognitive
opener: `I'm really sorry that you're feeling this way, but I'm`
  - F=+0.022  |T|=3  tokens=['really', 'this', "I'm"]  (positions [1, 6, 9])
  - F=+0.015  |T|=2  tokens=['feeling', 'way,']  (positions [5, 7])
  - F=-0.014  |T|=2  tokens=['way,', "I'm"]  (positions [7, 9])

### idx=1163 [Cog/wrong] true=Cognitive
opener: `I'm really sorry to hear that you're feeling this way,`
  - F=+0.016  |T|=2  tokens=['sorry', 'that']  (positions [2, 5])
  - F=-0.015  |T|=3  tokens=['feeling', 'this', 'way,']  (positions [7, 8, 9])
  - F=-0.015  |T|=3  tokens=["I'm", 'sorry', 'hear']  (positions [0, 2, 4])

### idx=128 [Cog/wrong] true=Cognitive
opener: `I'm really sorry to hear what you're going through, and`
  - F=-0.016  |T|=3  tokens=['really', 'to', 'hear']  (positions [1, 3, 4])
  - F=+0.015  |T|=3  tokens=['hear', 'what', 'and']  (positions [4, 5, 9])
  - F=+0.014  |T|=3  tokens=['really', "you're", 'and']  (positions [1, 6, 9])

### idx=842 [Cog/wrong] true=Cognitive
opener: `I'm really sorry to hear that you've been feeling this`
  - F=-0.016  |T|=3  tokens=["I'm", 'sorry', 'this']  (positions [0, 2, 9])
  - F=+0.016  |T|=2  tokens=["you've", 'been']  (positions [6, 7])
  - F=-0.015  |T|=1  tokens=['sorry']  (positions [2])

### idx=873 [Cog/wrong] true=Cognitive
opener: `I'm really sorry to hear that you're feeling underappreciated and`
  - F=-0.016  |T|=2  tokens=['feeling', 'and']  (positions [7, 9])
  - F=-0.014  |T|=3  tokens=["I'm", 'really', 'and']  (positions [0, 1, 9])
  - F=+0.014  |T|=3  tokens=["you're", 'feeling', 'and']  (positions [6, 7, 9])

### idx=1040 [Cog/wrong] true=Cognitive
opener: `I'm truly sorry that you're feeling misunderstood and unfairly judged`
  - F=-0.019  |T|=3  tokens=['truly', "you're", 'judged']  (positions [1, 4, 9])
  - F=-0.014  |T|=3  tokens=["I'm", 'that', 'unfairly']  (positions [0, 3, 8])
  - F=+0.013  |T|=3  tokens=['sorry', "you're", 'and']  (positions [2, 4, 7])

### idx=774 [Cog/wrong] true=Cognitive
opener: `I'm really sorry to hear about your granddad's passing. It's`
  - F=+0.018  |T|=3  tokens=['about', "granddad's", 'passing.']  (positions [5, 7, 8])
  - F=-0.013  |T|=2  tokens=['your', "It's"]  (positions [6, 9])
  - F=+0.012  |T|=2  tokens=['to', 'hear']  (positions [3, 4])

### idx=629 [Cog/wrong] true=Cognitive
opener: `I understand that this new, unexpected change in your birth`
  - F=+0.020  |T|=2  tokens=['new,', 'unexpected']  (positions [4, 5])
  - F=+0.017  |T|=3  tokens=['new,', 'in', 'birth']  (positions [4, 7, 9])
  - F=+0.017  |T|=3  tokens=['that', 'new,', 'your']  (positions [2, 4, 8])

### idx=408 [Cog/wrong] true=Cognitive
opener: `I'm deeply moved by the emotional struggle your friend is`
  - F=+0.016  |T|=3  tokens=['moved', 'struggle', 'is']  (positions [2, 6, 9])
  - F=-0.014  |T|=3  tokens=['moved', 'your', 'is']  (positions [2, 7, 9])
  - F=-0.012  |T|=3  tokens=['deeply', 'moved', 'emotional']  (positions [1, 2, 5])

### idx=957 [Cog/wrong] true=Cognitive
opener: `I'm so sorry for your loss, and I can really`
  - F=-0.018  |T|=3  tokens=['so', 'and', 'really']  (positions [1, 6, 9])
  - F=-0.017  |T|=2  tokens=['so', 'I']  (positions [1, 7])
  - F=-0.013  |T|=3  tokens=['for', 'I', 'can']  (positions [3, 7, 8])

### idx=208 [Aff/corre] true=Affective
opener: `I truly feel for you and your family during this`
  - F=-0.023  |T|=3  tokens=['I', 'truly', 'this']  (positions [0, 1, 9])
  - F=-0.016  |T|=3  tokens=['I', 'for', 'this']  (positions [0, 3, 9])
  - F=+0.016  |T|=3  tokens=['I', 'you', 'your']  (positions [0, 4, 6])

### idx=750 [Aff/corre] true=Affective
opener: `I can almost feel the warmth and joy you experienced`
  - F=-0.013  |T|=3  tokens=['feel', 'warmth', 'you']  (positions [3, 5, 8])
  - F=+0.013  |T|=3  tokens=['can', 'warmth', 'joy']  (positions [1, 5, 7])
  - F=-0.012  |T|=1  tokens=['feel']  (positions [3])

### idx=734 [Aff/corre] true=Affective
opener: `That's wonderful! It's inspiring to see how much joy you`
  - F=-0.017  |T|=3  tokens=['wonderful!', 'see', 'you']  (positions [1, 5, 9])
  - F=+0.014  |T|=3  tokens=['to', 'how', 'much']  (positions [4, 6, 7])
  - F=+0.014  |T|=2  tokens=["It's", 'you']  (positions [2, 9])

### idx=1151 [Aff/corre] true=Affective
opener: `I can't begin to imagine the terror and desperation you`
  - F=+0.014  |T|=3  tokens=["can't", 'begin', 'you']  (positions [1, 2, 9])
  - F=-0.013  |T|=3  tokens=['to', 'and', 'desperation']  (positions [3, 7, 8])
  - F=-0.012  |T|=3  tokens=['begin', 'desperation', 'you']  (positions [2, 8, 9])

### idx=869 [Aff/corre] true=Affective
opener: `Seeing your son making his sister laugh must really touch`
  - F=-0.017  |T|=3  tokens=['son', 'laugh', 'touch']  (positions [2, 6, 9])
  - F=-0.014  |T|=1  tokens=['son']  (positions [2])
  - F=+0.013  |T|=2  tokens=['Seeing', 'son']  (positions [0, 2])

### idx=1037 [Aff/corre] true=Affective
opener: `I'm truly touched by the strength and courage you show`
  - F=-0.014  |T|=2  tokens=['strength', 'show']  (positions [5, 9])
  - F=-0.014  |T|=3  tokens=['strength', 'and', 'show']  (positions [5, 6, 9])
  - F=+0.011  |T|=1  tokens=['touched']  (positions [2])

### idx=264 [Aff/corre] true=Affective
opener: `Your journey through those countless hours of breastfeeding and the`
  - F=-0.014  |T|=2  tokens=['through', 'the']  (positions [2, 9])
  - F=+0.014  |T|=3  tokens=['Your', 'those', 'breastfeeding']  (positions [0, 3, 7])
  - F=+0.013  |T|=3  tokens=['journey', 'countless', 'breastfeeding']  (positions [1, 4, 7])

### idx=285 [Aff/corre] true=Affective
opener: `Wow, it's truly heartwarming to hear about your daughter's progress.`
  - F=+0.016  |T|=3  tokens=['Wow,', 'truly', 'about']  (positions [0, 2, 6])
  - F=+0.014  |T|=2  tokens=['Wow,', 'your']  (positions [0, 7])
  - F=-0.013  |T|=3  tokens=['Wow,', 'about', 'progress.']  (positions [0, 6, 9])

### idx=599 [Aff/corre] true=Affective
opener: `I can feel the depth of your relief and gratitude`
  - F=+0.015  |T|=3  tokens=['I', 'can', 'the']  (positions [0, 1, 3])
  - F=-0.014  |T|=1  tokens=['depth']  (positions [4])
  - F=-0.013  |T|=1  tokens=['your']  (positions [6])

### idx=521 [Aff/corre] true=Affective
opener: `That's incredibly heartwarming to hear, the connection you've built with`
  - F=+0.016  |T|=3  tokens=["That's", 'heartwarming', "you've"]  (positions [0, 2, 7])
  - F=+0.015  |T|=3  tokens=['to', 'the', 'built']  (positions [3, 5, 8])
  - F=+0.015  |T|=3  tokens=["That's", 'incredibly', 'heartwarming']  (positions [0, 1, 2])