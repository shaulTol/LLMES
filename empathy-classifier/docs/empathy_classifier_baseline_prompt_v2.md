# Task: Empathy Type Classifier Baseline

Build a simple baseline classifier. Organize as a proper repo with Python scripts.

## Step 0: Repo Organization

Create this structure:

```
empathy-classifier/
├── data/
│   └── raw/
│       └── (move CSV here)
├── src/
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── models/
│   └── (saved checkpoints)
├── outputs/
│   └── (figures, results)
├── docs/
│   └── (papers, documentation)
├── notebooks/
│   └── (move existing .ipynb files here, for reference only)
├── requirements.txt
├── .gitignore
└── README.md
```

**Actions:**
- Create the folder structure above
- Move existing files to appropriate folders
- Create `requirements.txt`: torch, transformers, pandas, numpy, scikit-learn, seaborn, matplotlib
- Create `.gitignore`: `__pycache__`, `.ipynb_checkpoints`, `*.pt`, `data/raw/*`
- Create minimal `README.md` with project title and one-line description

## Step 1: Build the Baseline

### `src/data.py`
- Load CSV from `data/raw/Supplementary Data - Responses and Measures - all experiments (1).csv`
- Filter to StudyNum in ['1', '1b', '3']
- Text column: 'AI_Response'
- Label columns: 'EmpathyQ_Cognitive', 'EmpathyQ_Affective', 'EmpathyQ_Motivational'
- Create soft label: normalize the 3 scores to sum to 1
- Splits:
  - Train: Studies 1 + 1b, minus 50 random samples from each
  - Eval: those held-out 50+50
  - Test: Study 3
- RANDOM_SEED=42
- Expose one function: `get_dataloaders(batch_size=32)` → returns train_loader, eval_loader, test_loader
- Keep split logic easy to modify (few lines, top of file)

### `src/model.py`
- Frozen DistilBERT (`distilbert-base-uncased`)
- Single linear layer: 768 -> 3
- Softmax output
- Use [CLS] token embedding
- One class: `BaselineModel(nn.Module)`

### `src/train.py`
- Loss: soft cross-entropy = `-sum(y_true * log(y_pred))`
- Optimizer: Adam, lr=1e-3
- Batch size: 32
- Early stopping: patience=5 on eval loss
- Max 50 epochs
- Print train/eval loss each epoch
- Save best model to `models/baseline_v1.pt`
- Run with: `python src/train.py`

### `src/evaluate.py`
- Load model from `models/baseline_v1.pt`
- Run on test set (Study 3)
- Metrics:
  - Hard accuracy: argmax pred vs argmax true
  - Per-class accuracy
  - Confusion matrix (sklearn + seaborn plot)
- Save confusion matrix to `outputs/baseline_v1_confusion.png`
- Print summary to stdout
- Run with: `python src/evaluate.py`

## Code Style
- No classes except the model
- No config files, no argparse, no CLI flags
- Constants at top of each file
- Inline comments only where non-obvious
- If something fails, fail loud — no silent error handling
- Each file < 100 lines

## Output
- `python src/train.py` → prints losses, saves model
- `python src/evaluate.py` → prints accuracy, saves confusion matrix

Do not over-engineer. I want to read through all 4 files in 5 minutes and verify correctness.
