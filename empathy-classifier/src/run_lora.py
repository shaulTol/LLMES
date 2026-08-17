"""LoRA fine-tune DistilBERT for empathy-type classification.

One-config CLI runner. Outputs JSON with acc/F1 distribution over seeds.

LoRA design axes covered:
  --rank             4 / 8 / 16 / 32 ...
  --target           'qv' (q_lin, v_lin) | 'qkvo' | 'qkvo_ffn'
  --layer_scope      'top2' | 'top3' | 'all6'   (DistilBERT has 6 layers)
  --alpha            LoRA alpha; effective scale = alpha / rank
  --bias             'none' | 'lora_only' | 'all'

Training:
  --lr               encoder + head lr (one optimizer)
  --head             'linear' | 'mlp256'
  --balanced_samp    use WeightedRandomSampler over train classes
  --max_epochs, --patience, --batch_size, --seeds
"""
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import DistilBertModel, DistilBertTokenizer
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(SCRIPT_DIR, '..', 'data', 'raw',
                        'Supplementary Data - Responses and Measures - all experiments (1).csv')
OUT_DIR = os.path.join(SCRIPT_DIR, '..', 'outputs')
MAX_LEN = 256
NUM_LAYERS = 6  # DistilBERT-base has 6 transformer layers

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_target_modules(target, layer_scope):
    if target == 'qv':
        mods = ['q_lin', 'v_lin']
    elif target == 'qkvo':
        mods = ['q_lin', 'k_lin', 'v_lin', 'out_lin']
    elif target == 'qkvo_ffn':
        mods = ['q_lin', 'k_lin', 'v_lin', 'out_lin', 'lin1', 'lin2']
    else:
        raise ValueError(f'Unknown target {target}')

    if layer_scope == 'top2':
        layers = [NUM_LAYERS - 2, NUM_LAYERS - 1]
    elif layer_scope == 'top3':
        layers = [NUM_LAYERS - 3, NUM_LAYERS - 2, NUM_LAYERS - 1]
    elif layer_scope == 'all6':
        layers = list(range(NUM_LAYERS))
    else:
        raise ValueError(f'Unknown layer_scope {layer_scope}')

    # DistilBertModel module paths
    targets = []
    for i in layers:
        for m in mods:
            sub = 'attention' if m in ('q_lin', 'k_lin', 'v_lin', 'out_lin') else 'ffn'
            targets.append(f'transformer.layer.{i}.{sub}.{m}')
    return targets


class TextDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.iids = input_ids
        self.am = attention_mask
        self.y = labels
    def __len__(self):
        return len(self.iids)
    def __getitem__(self, idx):
        return {'input_ids': self.iids[idx], 'attention_mask': self.am[idx],
                'labels': self.y[idx]}


class LoRABertClassifier(nn.Module):
    def __init__(self, lora_cfg, head_kind='mlp256', dropout=0.3, latent_sigma=0.0):
        super().__init__()
        base = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.bert = get_peft_model(base, lora_cfg)
        if head_kind == 'linear':
            self.head = nn.Linear(768, 3)
        elif head_kind == 'mlp256':
            self.head = nn.Sequential(
                nn.Linear(768, 256), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(256, 3),
            )
        else:
            raise ValueError(head_kind)
        self.latent_sigma = latent_sigma  # 0 = no noise; >0 = inject N(0, sigma*per_dim_std)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        if self.training and self.latent_sigma > 0:
            # per-batch per-dim std (detached so it doesn't pull on LoRA grads)
            with torch.no_grad():
                per_dim_std = cls.std(dim=0, keepdim=True) + 1e-6
            noise = torch.randn_like(cls) * self.latent_sigma * per_dim_std
            cls = cls + noise
        return self.head(cls)


def soft_ce(logits, target):
    return -(target * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()


def _word_mask_aug(text, mask_pct, mask_token, rng):
    words = text.split()
    if len(words) < 2:
        return text
    n_mask = max(1, int(round(len(words) * mask_pct)))
    idx = rng.choice(len(words), size=n_mask, replace=False)
    out = list(words)
    for i in idx:
        out[i] = mask_token
    return ' '.join(out)


def _apply_text_aug(texts, labels, aug_mode, target, mask_pct=0.15, seed=2026):
    """Augment training set by word-masking. Returns (texts, labels).

    aug_mode:
      'none' / 'balanced_samp'        — no text-level aug, return as-is.
      'text_min_to_max'               — bring minority classes up to majority count.
      'text_all_to_target'            — bring all classes up to `target` per class.
    """
    if aug_mode in ('none', 'balanced_samp'):
        return list(texts), list(labels)

    rng = np.random.default_rng(seed)
    labels_arr = np.array(labels)
    arg = labels_arr.argmax(axis=1)
    counts = np.bincount(arg, minlength=3)
    if aug_mode == 'text_min_to_max':
        per_class_target = int(counts.max())
    elif aug_mode == 'text_all_to_target':
        per_class_target = int(target)
    else:
        raise ValueError(f'Unknown aug_mode {aug_mode}')

    new_texts = list(texts)
    new_labels = [labels[i] for i in range(len(labels))]
    for k in range(3):
        cur = counts[k]
        if cur >= per_class_target:
            continue
        idx_k = np.where(arg == k)[0]
        need = per_class_target - cur
        for _ in range(int(need)):
            src = idx_k[rng.integers(len(idx_k))]
            aug = _word_mask_aug(texts[src], mask_pct, '[MASK]', rng)
            new_texts.append(aug)
            new_labels.append(labels[src])
    return new_texts, new_labels


def load_data(tokenizer, rng_seed=42, aug_mode='balanced_samp', aug_target=0):
    df = pd.read_csv(DATA_RAW)
    df = df[df['StudyNum'].isin(['1', '1b', '3'])].copy().reset_index(drop=True)
    texts = df['Response'].fillna('').astype(str).tolist()
    raw = df[['cognitive', 'affective', 'motivational']].values.astype(float)
    s = raw.sum(axis=1, keepdims=True); s[s == 0] = 1
    soft = raw / s
    studyn = df['StudyNum'].values

    # Match the canonical split used elsewhere (data.py logic)
    s1 = np.where(studyn == '1')[0]
    s1b = np.where(studyn == '1b')[0]
    s3 = np.where(studyn == '3')[0]
    np.random.seed(rng_seed)
    eval_1 = np.random.choice(s1, size=50, replace=False)
    eval_1b = np.random.choice(s1b, size=50, replace=False)
    train_idx = np.concatenate([np.setdiff1d(s1, eval_1), np.setdiff1d(s1b, eval_1b)])
    eval_idx = np.concatenate([eval_1, eval_1b])
    test_idx = s3

    # Apply text-level augmentation to TRAIN only (if requested)
    train_texts = [texts[i] for i in train_idx]
    train_labels = soft[train_idx]
    train_texts_aug, train_labels_aug = _apply_text_aug(
        train_texts, train_labels.tolist(), aug_mode, aug_target,
        mask_pct=0.15, seed=rng_seed,
    )
    eval_texts = [texts[i] for i in eval_idx]
    test_texts = [texts[i] for i in test_idx]

    enc_tr = tokenizer(train_texts_aug, truncation=True, padding='max_length',
                       max_length=MAX_LEN, return_tensors='pt')
    enc_ev = tokenizer(eval_texts,      truncation=True, padding='max_length',
                       max_length=MAX_LEN, return_tensors='pt')
    enc_te = tokenizer(test_texts,      truncation=True, padding='max_length',
                       max_length=MAX_LEN, return_tensors='pt')

    return {
        'train_iids': enc_tr['input_ids'], 'train_mask': enc_tr['attention_mask'],
        'train_y': torch.tensor(np.array(train_labels_aug), dtype=torch.float32),
        'eval_iids': enc_ev['input_ids'],  'eval_mask': enc_ev['attention_mask'],
        'eval_y': torch.tensor(soft[eval_idx], dtype=torch.float32),
        'test_iids': enc_te['input_ids'],  'test_mask': enc_te['attention_mask'],
        'test_y': torch.tensor(soft[test_idx], dtype=torch.float32),
    }


def train_one_seed(args, data, seed):
    torch.manual_seed(seed); np.random.seed(seed)

    target_modules = build_target_modules(args.target, args.layer_scope)
    lora_cfg = LoraConfig(
        r=args.rank, lora_alpha=args.alpha if args.alpha > 0 else args.rank,
        target_modules=target_modules, lora_dropout=args.lora_dropout, bias=args.bias,
        task_type='FEATURE_EXTRACTION',
    )
    model = LoRABertClassifier(lora_cfg, head_kind=args.head, dropout=args.head_dropout,
                                latent_sigma=args.latent_sigma).to(DEVICE)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.wd,
    )

    # Sampler: balanced_samp uses weights so each class is equally likely per batch.
    # (Compatible with any aug mode — operates on the *post-aug* train labels.)
    train_y = data['train_y']
    if args.aug_mode == 'balanced_samp':
        argmax = train_y.argmax(dim=1).numpy()
        counts = np.array([(argmax == k).sum() for k in range(3)], dtype=float)
        w = 1.0 / counts[argmax]
        sampler = WeightedRandomSampler(weights=torch.tensor(w), num_samples=len(w), replacement=True)
        shuffle = False
    else:
        sampler = None; shuffle = True

    train_ds = TextDataset(data['train_iids'], data['train_mask'], train_y)
    eval_ds  = TextDataset(data['eval_iids'],  data['eval_mask'],  data['eval_y'])
    test_ds  = TextDataset(data['test_iids'],  data['test_mask'],  data['test_y'])
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler)
    eval_ld  = DataLoader(eval_ds, batch_size=args.batch_size)
    test_ld  = DataLoader(test_ds, batch_size=args.batch_size)

    best = float('inf'); bad = 0; best_state = None
    for epoch in range(args.max_epochs):
        model.train()
        for b in train_ld:
            iid = b['input_ids'].to(DEVICE)
            mask = b['attention_mask'].to(DEVICE)
            y = b['labels'].to(DEVICE)
            opt.zero_grad()
            loss = soft_ce(model(iid, mask), y)
            loss.backward(); opt.step()
        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for b in eval_ld:
                iid = b['input_ids'].to(DEVICE)
                mask = b['attention_mask'].to(DEVICE)
                y = b['labels'].to(DEVICE)
                eval_loss += soft_ce(model(iid, mask), y).item()
        eval_loss /= len(eval_ld)
        if eval_loss < best:
            best = eval_loss; bad = 0
            best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience: break

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for b in test_ld:
            iid = b['input_ids'].to(DEVICE)
            mask = b['attention_mask'].to(DEVICE)
            y = b['labels']
            logits = model(iid, mask)
            preds.append(logits.argmax(dim=1).cpu().numpy())
            trues.append(y.argmax(dim=1).cpu().numpy())
    preds = np.concatenate(preds); trues = np.concatenate(trues)
    return {
        'acc': float(accuracy_score(trues, preds)),
        'f1':  float(f1_score(trues, preds, labels=[0, 1, 2], average='macro', zero_division=0)),
        'epochs': epoch + 1,
        'preds': preds,
        'trues': trues,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rank', type=int, required=True)
    p.add_argument('--target', type=str, required=True, choices=['qv', 'qkvo', 'qkvo_ffn'])
    p.add_argument('--layer_scope', type=str, required=True, choices=['top2', 'top3', 'all6'])
    p.add_argument('--alpha', type=int, default=-1, help='-1 means alpha=rank')
    p.add_argument('--bias', type=str, default='none', choices=['none', 'lora_only', 'all'])
    p.add_argument('--head', type=str, default='mlp256', choices=['linear', 'mlp256'])
    p.add_argument('--head_dropout', type=float, default=0.3)
    p.add_argument('--lora_dropout', type=float, default=0.0)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--wd', type=float, default=0.01)
    p.add_argument('--latent_sigma', type=float, default=0.0,
                   help='Gaussian noise added to LoRA-adapted [CLS] during training; 0 = off')
    p.add_argument('--aug_mode', type=str, default='balanced_samp',
                   choices=['none', 'balanced_samp', 'text_min_to_max', 'text_all_to_target'])
    p.add_argument('--aug_target', type=int, default=0,
                   help='for text_all_to_target: target count per class')
    p.add_argument('--max_epochs', type=int, default=60)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--seeds', type=int, default=10)
    p.add_argument('--tag', type=str, default='', help='extra suffix for output filename')
    args = p.parse_args()

    print(f'Device: {DEVICE}')
    print(f'Config: {vars(args)}')

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    data = load_data(tokenizer, aug_mode=args.aug_mode, aug_target=args.aug_target)
    print(f'sizes: train={len(data["train_iids"])}  eval={len(data["eval_iids"])}'
          f'  test={len(data["test_iids"])}')

    results = []
    for s in range(args.seeds):
        r = train_one_seed(args, data, seed=s)
        results.append(r)
        print(f'  seed {s}: acc {r["acc"]:.4f}  F1 {r["f1"]:.4f}  ep {r["epochs"]}')

    accs = np.array([r['acc'] for r in results])
    f1s  = np.array([r['f1']  for r in results])
    eps  = np.array([r['epochs'] for r in results])
    summary = {
        'args': vars(args),
        'acc_mean': float(accs.mean()), 'acc_std': float(accs.std()),
        'f1_mean':  float(f1s.mean()),  'f1_std':  float(f1s.std()),
        'ep_mean':  float(eps.mean()),
        'seed_results': results,
    }
    print(f'\nSummary  acc {accs.mean():.4f}±{accs.std():.4f}   '
          f'F1 {f1s.mean():.4f}±{f1s.std():.4f}   ep {eps.mean():.1f}')

    os.makedirs(OUT_DIR, exist_ok=True)
    alpha_tag = (args.alpha if args.alpha > 0 else args.rank)
    aug_tag = args.aug_mode + (f'{args.aug_target}' if args.aug_target > 0 else '')
    tag_extra = f'_{args.tag}' if args.tag else ''
    tag = (f'lora_r{args.rank}_a{alpha_tag}_{args.target}_{args.layer_scope}'
           f'_bias-{args.bias}_head-{args.head}_lr{args.lr:.0e}_wd{args.wd:.0e}'
           f'_aug-{aug_tag}_ep{args.max_epochs}_pat{args.patience}'
           f'_n{args.seeds}{tag_extra}')
    out_path = os.path.join(OUT_DIR, f'{tag}.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
