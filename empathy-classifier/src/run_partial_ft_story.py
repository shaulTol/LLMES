"""Partial encoder fine-tune on Story+Response architecture.

Mirrors run_lora_story.py but replaces LoRA with partial unfreeze: freeze
embeddings + bottom transformer layers, unfreeze the top `--unfreeze_top` layers.
Same MLP head, same balanced_samp, same eval-loss early stop. Decoupled lrs:
--lr for head, --encoder_lr for the unfrozen layers.

Goal: test whether allowing the top layers to reshape (rather than being
constrained to LoRA's rank-r updates on q/v) breaks the F1 ~0.40 ceiling we
hit with frozen-then-LoRA DistilBERT.

DistilBERT has 6 transformer layers: bert.transformer.layer[0..5].
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
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_lora import MAX_LEN, DATA_RAW

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_LAYERS = 6  # DistilBERT


class StoryRespDataset(Dataset):
    def __init__(self, s_iids, s_mask, r_iids, r_mask, y):
        self.s_iids = s_iids; self.s_mask = s_mask
        self.r_iids = r_iids; self.r_mask = r_mask
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return {'s_iids': self.s_iids[idx], 's_mask': self.s_mask[idx],
                'r_iids': self.r_iids[idx], 'r_mask': self.r_mask[idx],
                'labels': self.y[idx]}


class PartialFTStoryResp(nn.Module):
    def __init__(self, unfreeze_top, mlp_hidden=256, dropout=0.5, skip_conn=False):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        # Freeze everything, then selectively unfreeze top N transformer layers.
        for p in self.bert.parameters():
            p.requires_grad = False
        assert 0 <= unfreeze_top <= NUM_LAYERS, f'unfreeze_top must be in [0, {NUM_LAYERS}]'
        if unfreeze_top > 0:
            start = NUM_LAYERS - unfreeze_top
            for i in range(start, NUM_LAYERS):
                for p in self.bert.transformer.layer[i].parameters():
                    p.requires_grad = True
        self.skip_conn = skip_conn
        if skip_conn:
            # Second BERT instance, fully frozen — provides the unadapted CLS branch.
            self.bert_frozen = DistilBertModel.from_pretrained('distilbert-base-uncased')
            for p in self.bert_frozen.parameters():
                p.requires_grad = False
            self.bert_frozen.eval()
        # Head: concat [story_CLS, response_CLS] -> MLP; doubled if skip_conn.
        per_text_dim = 768 * (2 if skip_conn else 1)
        head_in = per_text_dim * 2  # story + response
        self.head = nn.Sequential(
            nn.Linear(head_in, mlp_hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(mlp_hidden, 3),
        )
    def _enc(self, iids, mask):
        unfrozen = self.bert(input_ids=iids, attention_mask=mask).last_hidden_state[:, 0, :]
        if not self.skip_conn:
            return unfrozen
        with torch.no_grad():
            frozen = self.bert_frozen(input_ids=iids, attention_mask=mask).last_hidden_state[:, 0, :]
        return torch.cat([frozen, unfrozen], dim=1)
    def forward(self, s_iids, s_mask, r_iids, r_mask):
        s = self._enc(s_iids, s_mask)
        r = self._enc(r_iids, r_mask)
        return self.head(torch.cat([s, r], dim=1))


def soft_ce(logits, target):
    return -(target * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()


def load_data(tokenizer, rng_seed=42):
    df = pd.read_csv(DATA_RAW)
    df = df[df['StudyNum'].isin(['1', '1b', '3'])].copy().reset_index(drop=True)
    responses = df['Response'].fillna('').astype(str).tolist()
    stories   = df['Story'].fillna('').astype(str).tolist()
    raw = df[['cognitive', 'affective', 'motivational']].values.astype(float)
    s = raw.sum(axis=1, keepdims=True); s[s == 0] = 1
    soft = raw / s
    studyn = df['StudyNum'].values
    s1  = np.where(studyn == '1')[0]
    s1b = np.where(studyn == '1b')[0]
    s3  = np.where(studyn == '3')[0]
    np.random.seed(rng_seed)
    eval_1  = np.random.choice(s1,  size=50, replace=False)
    eval_1b = np.random.choice(s1b, size=50, replace=False)
    train_idx = np.concatenate([np.setdiff1d(s1, eval_1), np.setdiff1d(s1b, eval_1b)])
    eval_idx  = np.concatenate([eval_1, eval_1b])
    test_idx  = s3

    def tok(lst):
        e = tokenizer(lst, truncation=True, padding='max_length',
                       max_length=MAX_LEN, return_tensors='pt')
        return e['input_ids'], e['attention_mask']

    tr_s_iids, tr_s_mask = tok([stories[i]   for i in train_idx])
    tr_r_iids, tr_r_mask = tok([responses[i] for i in train_idx])
    ev_s_iids, ev_s_mask = tok([stories[i]   for i in eval_idx])
    ev_r_iids, ev_r_mask = tok([responses[i] for i in eval_idx])
    te_s_iids, te_s_mask = tok([stories[i]   for i in test_idx])
    te_r_iids, te_r_mask = tok([responses[i] for i in test_idx])
    return {
        'tr_s_iids': tr_s_iids, 'tr_s_mask': tr_s_mask,
        'tr_r_iids': tr_r_iids, 'tr_r_mask': tr_r_mask,
        'tr_y': torch.tensor(soft[train_idx], dtype=torch.float32),
        'ev_s_iids': ev_s_iids, 'ev_s_mask': ev_s_mask,
        'ev_r_iids': ev_r_iids, 'ev_r_mask': ev_r_mask,
        'ev_y': torch.tensor(soft[eval_idx], dtype=torch.float32),
        'te_s_iids': te_s_iids, 'te_s_mask': te_s_mask,
        'te_r_iids': te_r_iids, 'te_r_mask': te_r_mask,
        'te_y': torch.tensor(soft[test_idx], dtype=torch.float32),
    }


def train_one_seed(args, data, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = PartialFTStoryResp(unfreeze_top=args.unfreeze_top,
                                 mlp_hidden=256, dropout=args.head_dropout,
                                 skip_conn=args.skip_conn).to(DEVICE)

    # Param groups: head, plus one group per unfrozen transformer layer.
    # When LLRD is on, deeper layers get lr scaled by args.llrd ** (depth_from_top).
    # depth_from_top=0 -> top layer keeps args.encoder_lr; depth=1 -> args.encoder_lr * llrd; etc.
    head_params = []
    layer_param_groups = []   # list of (params, lr)
    if args.unfreeze_top > 0:
        start = NUM_LAYERS - args.unfreeze_top
        for layer_idx in range(start, NUM_LAYERS):
            depth_from_top = NUM_LAYERS - 1 - layer_idx
            lr_scale = (args.llrd ** depth_from_top) if args.llrd > 0 else 1.0
            lr = args.encoder_lr * lr_scale
            params = [p for p in model.bert.transformer.layer[layer_idx].parameters() if p.requires_grad]
            layer_param_groups.append((params, lr, layer_idx))
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if not name.startswith('bert.'):
            head_params.append(p)
    groups = [{'params': head_params, 'lr': args.lr}]
    for params, lr, _ in layer_param_groups:
        groups.append({'params': params, 'lr': lr})
    opt = torch.optim.AdamW(groups, weight_decay=args.wd)
    if seed == 0:
        enc_total = sum(sum(p.numel() for p in g) for g, _, _ in layer_param_groups)
        skip_str = ' +skip_conn' if args.skip_conn else ''
        llrd_str = f'  LLRD={args.llrd}' if args.llrd > 0 else ''
        per_layer_lrs = '  '.join(f'L{li}:{lr:.0e}' for _, lr, li in layer_param_groups)
        print(f'  unfreeze_top={args.unfreeze_top}{skip_str}{llrd_str}  head_lr={args.lr}  '
              f'(head_params={sum(p.numel() for p in head_params)}, encoder_params={enc_total})')
        if per_layer_lrs:
            print(f'  per-layer lrs: {per_layer_lrs}')

    ty = data['tr_y']
    arg = ty.argmax(dim=1).numpy()
    counts = np.array([(arg == k).sum() for k in range(3)], dtype=float)
    w = 1.0 / counts[arg]
    sampler = WeightedRandomSampler(weights=torch.tensor(w), num_samples=len(w), replacement=True)

    tr_ds = StoryRespDataset(data['tr_s_iids'], data['tr_s_mask'],
                              data['tr_r_iids'], data['tr_r_mask'], ty)
    ev_ds = StoryRespDataset(data['ev_s_iids'], data['ev_s_mask'],
                              data['ev_r_iids'], data['ev_r_mask'], data['ev_y'])
    te_ds = StoryRespDataset(data['te_s_iids'], data['te_s_mask'],
                              data['te_r_iids'], data['te_r_mask'], data['te_y'])
    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, sampler=sampler)
    ev_ld = DataLoader(ev_ds, batch_size=args.batch_size)
    te_ld = DataLoader(te_ds, batch_size=args.batch_size)

    best = float('inf'); bad = 0; best_state = None
    for epoch in range(args.max_epochs):
        model.train()
        for b in tr_ld:
            si, sm = b['s_iids'].to(DEVICE), b['s_mask'].to(DEVICE)
            ri, rm = b['r_iids'].to(DEVICE), b['r_mask'].to(DEVICE)
            y = b['labels'].to(DEVICE)
            opt.zero_grad()
            loss = soft_ce(model(si, sm, ri, rm), y)
            loss.backward(); opt.step()
        model.eval()
        ev_loss = 0.0
        with torch.no_grad():
            for b in ev_ld:
                si, sm = b['s_iids'].to(DEVICE), b['s_mask'].to(DEVICE)
                ri, rm = b['r_iids'].to(DEVICE), b['r_mask'].to(DEVICE)
                y = b['labels'].to(DEVICE)
                ev_loss += soft_ce(model(si, sm, ri, rm), y).item()
        ev_loss /= len(ev_ld)
        if ev_loss < best:
            best = ev_loss; bad = 0
            best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience: break

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for b in te_ld:
            si, sm = b['s_iids'].to(DEVICE), b['s_mask'].to(DEVICE)
            ri, rm = b['r_iids'].to(DEVICE), b['r_mask'].to(DEVICE)
            preds.append(model(si, sm, ri, rm).argmax(dim=1).cpu().numpy())
            trues.append(b['labels'].argmax(dim=1).cpu().numpy())
    preds = np.concatenate(preds); trues = np.concatenate(trues)
    return {
        'acc': float(accuracy_score(trues, preds)),
        'f1':  float(f1_score(trues, preds, labels=[0, 1, 2], average='macro', zero_division=0)),
        'epochs': epoch + 1,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--unfreeze_top', type=int, required=True,
                   help='Number of top DistilBERT transformer layers to unfreeze (0-6).')
    p.add_argument('--encoder_lr', type=float, default=3e-5)
    p.add_argument('--lr',         type=float, default=3e-4, help='Head learning rate.')
    p.add_argument('--wd',         type=float, default=0.01)
    p.add_argument('--head_dropout', type=float, default=0.5)
    p.add_argument('--skip_conn', action='store_true',
                   help='Head sees both frozen-BERT and partial-FT BERT [CLS] per text.')
    p.add_argument('--llrd', type=float, default=0.0,
                   help='Layer-wise lr decay multiplier; deeper unfrozen layers get '
                        'encoder_lr * llrd^(depth_from_top). 0 = off (uniform).')
    p.add_argument('--max_epochs', type=int, default=100)
    p.add_argument('--patience',   type=int, default=15)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--seeds',      type=int, default=10)
    p.add_argument('--seed_offset', type=int, default=0)
    p.add_argument('--tag',        type=str, default='')
    args = p.parse_args()

    print(f'Device: {DEVICE}\nConfig: {vars(args)}')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    data = load_data(tokenizer)
    print(f'sizes: tr={len(data["tr_y"])} ev={len(data["ev_y"])} te={len(data["te_y"])}')

    results = []
    for i in range(args.seeds):
        s = args.seed_offset + i
        r = train_one_seed(args, data, seed=s)
        r['seed'] = s
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
    tag_extra = f'_{args.tag}' if args.tag else ''
    tag = (f'partialft_story_top{args.unfreeze_top}'
           f'_enclr{args.encoder_lr:.0e}_headlr{args.lr:.0e}_wd{args.wd:.0e}'
           f'_ep{args.max_epochs}_pat{args.patience}_n{args.seeds}{tag_extra}')
    with open(os.path.join(OUT_DIR, f'{tag}.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved outputs/{tag}.json')


if __name__ == '__main__':
    main()
