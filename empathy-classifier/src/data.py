import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Splits: Train = Studies 1 + 1b (minus 50 each), Eval = 50+50 held-out, Test = Study 3
EVAL_SAMPLES_PER_STUDY = 50

class EmpathyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(label)
        }

def get_dataloaders(batch_size=32):
    # Load data
    import os
    # Try to find data file - works both locally and in Colab
    script_dir = os.path.dirname(__file__)
    data_path = os.path.join(script_dir, '..', 'data', 'raw', 'Supplementary Data - Responses and Measures - all experiments (1).csv')
    
    # If not found, try absolute path from current working directory
    if not os.path.exists(data_path):
        data_path = os.path.join(os.getcwd(), 'data', 'raw', 'Supplementary Data - Responses and Measures - all experiments (1).csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find data file. Tried: {data_path}")
    
    df = pd.read_csv(data_path)
    df = df[df['StudyNum'].isin(['1', '1b', '3'])].copy()
    
    # Extract text and labels
    texts = df['Response'].fillna('').astype(str).tolist()
    labels = df[['cognitive', 'affective', 'motivational']].values.astype(float)
    
    # Create soft labels: normalize to sum to 1
    label_sums = labels.sum(axis=1, keepdims=True)
    label_sums[label_sums == 0] = 1
    soft_labels = labels / label_sums
    
    # Splits
    study1_idx = df[df['StudyNum'] == '1'].index.tolist()
    study1b_idx = df[df['StudyNum'] == '1b'].index.tolist()
    study3_idx = df[df['StudyNum'] == '3'].index.tolist()
    
    eval_1_idx = np.random.choice(study1_idx, size=min(EVAL_SAMPLES_PER_STUDY, len(study1_idx)), replace=False).tolist()
    eval_1b_idx = np.random.choice(study1b_idx, size=min(EVAL_SAMPLES_PER_STUDY, len(study1b_idx)), replace=False).tolist()
    
    train_idx = [i for i in study1_idx if i not in eval_1_idx] + [i for i in study1b_idx if i not in eval_1b_idx]
    eval_idx = eval_1_idx + eval_1b_idx
    test_idx = study3_idx
    
    train_texts = [texts[i] for i in train_idx]
    train_labels = soft_labels[train_idx]
    eval_texts = [texts[i] for i in eval_idx]
    eval_labels = soft_labels[eval_idx]
    test_texts = [texts[i] for i in test_idx]
    test_labels = soft_labels[test_idx]
    
    # Tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Datasets
    train_dataset = EmpathyDataset(train_texts, train_labels, tokenizer)
    eval_dataset = EmpathyDataset(eval_texts, eval_labels, tokenizer)
    test_dataset = EmpathyDataset(test_texts, test_labels, tokenizer)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, eval_loader, test_loader

