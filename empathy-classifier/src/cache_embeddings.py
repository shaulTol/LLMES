import os
import numpy as np
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertModel

RANDOM_SEED = 42
EVAL_SAMPLES_PER_STUDY = 50
MAX_LENGTH = 512
BATCH_SIZE = 32

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

script_dir = os.path.dirname(__file__)
csv_path = os.path.join(script_dir, '..', 'data', 'raw',
                       'Supplementary Data - Responses and Measures - all experiments (1).csv')
out_path = os.path.join(script_dir, '..', 'data', 'processed', 'cls_embeddings_distilbert.npz')

print(f'Device: {device}')
print(f'Loading CSV from {csv_path}')
df = pd.read_csv(csv_path)
df = df[df['StudyNum'].isin(['1', '1b', '3'])].copy().reset_index(drop=True)
N = len(df)
print(f'Filtered rows: {N}')

texts = df['Response'].fillna('').astype(str).tolist()
labels_raw = df[['cognitive', 'affective', 'motivational']].values.astype(np.float32)
label_sums = labels_raw.sum(axis=1, keepdims=True)
label_sums[label_sums == 0] = 1
soft_labels = labels_raw / label_sums
study_num = df['StudyNum'].values.astype(str)

# Reproduce baseline split logic from src/data.py (RANDOM_SEED=42)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
study1_idx = np.where(study_num == '1')[0]
study1b_idx = np.where(study_num == '1b')[0]
study3_idx = np.where(study_num == '3')[0]
eval_1 = np.random.choice(study1_idx, size=min(EVAL_SAMPLES_PER_STUDY, len(study1_idx)), replace=False)
eval_1b = np.random.choice(study1b_idx, size=min(EVAL_SAMPLES_PER_STUDY, len(study1b_idx)), replace=False)
eval_idx = np.sort(np.concatenate([eval_1, eval_1b]))
train_idx = np.sort(np.array(
    [i for i in np.concatenate([study1_idx, study1b_idx]) if i not in set(eval_idx.tolist())]
))
test_idx = np.sort(study3_idx)
print(f'Splits: train={len(train_idx)}, eval={len(eval_idx)}, test={len(test_idx)}')

print('Loading tokenizer + frozen DistilBERT')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
model.eval()
for p in model.parameters():
    p.requires_grad = False

embeddings = np.zeros((N, 768), dtype=np.float32)
print(f'Computing [CLS] embeddings for {N} rows, batch={BATCH_SIZE}')
with torch.no_grad():
    for start in range(0, N, BATCH_SIZE):
        end = min(start + BATCH_SIZE, N)
        batch_texts = texts[start:end]
        enc = tokenizer(batch_texts, truncation=True, padding='max_length',
                       max_length=MAX_LENGTH, return_tensors='pt')
        input_ids = enc['input_ids'].to(device)
        attn = enc['attention_mask'].to(device)
        out = model(input_ids=input_ids, attention_mask=attn)
        cls = out.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings[start:end] = cls
        if start % (BATCH_SIZE * 8) == 0:
            print(f'  {end}/{N}')

os.makedirs(os.path.dirname(out_path), exist_ok=True)
np.savez(out_path,
         embeddings=embeddings,
         soft_labels=soft_labels,
         labels_raw=labels_raw,
         study_num=study_num,
         train_idx=train_idx,
         eval_idx=eval_idx,
         test_idx=test_idx)
print(f'Saved cache to {out_path}')
print(f'  embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}')
print(f'  cache size on disk: {os.path.getsize(out_path)/1024/1024:.2f} MB')
