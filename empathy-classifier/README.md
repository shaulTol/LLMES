# Empathy Type Classifier

Baseline classifier for predicting empathy types (cognitive, affective, motivational) from text responses using a frozen DistilBERT model.

## Repository Structure

```
empathy-classifier/
├── data/
│   └── raw/
│       └── Supplementary Data - Responses and Measures - all experiments (1).csv
│           └── Contains the dataset with text responses and empathy ratings
├── src/
│   ├── data.py          # Data loading, preprocessing, and train/eval/test splits
│   ├── model.py          # Neural network model definition (frozen DistilBERT + linear classifier)
│   ├── train.py          # Training script with early stopping
│   └── evaluate.py       # Evaluation script that computes accuracy and confusion matrix
├── models/
│   └── (saved model checkpoints go here, e.g., baseline_v1.pt)
├── outputs/
│   └── (figures and results go here, e.g., confusion matrix plots)
├── notebooks/
│   └── (Jupyter notebooks for exploratory analysis and reference)
├── docs/
│   └── (Documentation, papers, and project instructions)
├── requirements.txt      # Python package dependencies
├── .gitignore           # Files to ignore in version control
└── README.md            # This file
```

## What This Project Does

This project trains a machine learning model to classify text responses into three empathy types:
- **Cognitive**: Understanding and perspective-taking
- **Affective**: Emotional resonance and connection
- **Motivational**: Support and encouragement

The model uses a pre-trained DistilBERT (a smaller, faster version of BERT) with frozen weights, meaning we only train a small linear classifier on top of the pre-trained embeddings.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure data is in place:**
   - The CSV file should be in `data/raw/` (already included)

## Usage

### Training

Train the model on Studies 1 and 1b, with evaluation on held-out samples:
```bash
python src/train.py
```

This will:
- Load and preprocess the data
- Split into train/eval/test sets
- Train the model with early stopping (patience=5)
- Save the best model to `models/baseline_v1.pt`
- Print training and evaluation loss for each epoch

### Evaluation

Evaluate the trained model on the test set (Study 3):
```bash
python src/evaluate.py
```

This will:
- Load the saved model from `models/baseline_v1.pt`
- Evaluate on test set (Study 3 data)
- Print overall accuracy and per-class accuracy
- Generate and save a confusion matrix to `outputs/baseline_v1_confusion.png`

## Data Splits

The data is split as follows:
- **Train**: Studies 1 + 1b, minus 50 random samples from each (held out for evaluation)
- **Eval**: 50 samples from Study 1 + 50 samples from Study 1b (used for early stopping)
- **Test**: All of Study 3 (used for final evaluation)

This split allows us to:
1. Train on Studies 1 and 1b (general prompts)
2. Test generalization to Study 3 (targeted prompts for specific empathy types)

## Model Architecture

- **Base**: Frozen DistilBERT (`distilbert-base-uncased`)
- **Classifier**: Single linear layer (768 → 3)
- **Output**: Softmax probabilities over 3 empathy types
- **Features**: Uses the [CLS] token embedding from DistilBERT

## Training Details

- **Loss function**: Soft cross-entropy (handles soft labels that sum to 1)
- **Optimizer**: Adam with learning rate 1e-3
- **Batch size**: 32
- **Max epochs**: 50
- **Early stopping**: Patience of 5 epochs on evaluation loss
- **Random seed**: 42 (for reproducibility)

## File Descriptions

- **`src/data.py`**: 
  - Loads CSV data
  - Filters to Studies 1, 1b, and 3
  - Creates soft labels (normalizes empathy scores to sum to 1)
  - Implements data splits
  - Provides `get_dataloaders()` function for PyTorch DataLoaders

- **`src/model.py`**: 
  - Defines `BaselineModel` class
  - Loads pre-trained DistilBERT and freezes it
  - Adds a trainable linear classifier on top

- **`src/train.py`**: 
  - Main training script
  - Implements training loop with early stopping
  - Saves best model based on evaluation loss

- **`src/evaluate.py`**: 
  - Loads trained model
  - Computes accuracy metrics
  - Generates confusion matrix visualization

## Output Files

After running the scripts, you'll find:
- `models/baseline_v1.pt`: Saved model weights
- `outputs/baseline_v1_confusion.png`: Confusion matrix visualization
