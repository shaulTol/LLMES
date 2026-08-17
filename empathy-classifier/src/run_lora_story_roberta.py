"""RoBERTa-base LoRA Story+Response.

Mirrors run_lora_story.py but uses roberta-base as the base model instead of
distilbert-base-uncased. Same architecture: Story+Response, two forwards through
shared LoRA-adapted encoder, concat CLS, MLP head with optional skip_conn
(frozen ⊕ LoRA-adapted per text), opener swap, decoupled lrs, etc.

RoBERTa-base: 12 transformer layers, 768 hidden, ~125M params. Trained with
dynamic masking on 160GB text — different pretraining objective and 10x more
data than the BERT teacher of DistilBERT. PEFT LoRA target modules are
'query' and 'value' (RoBERTa attention naming).

(Originally tried DeBERTa-v3-base — known SentencePiece/tiktoken incompat
with transformers 5.x; pivoted to RoBERTa to ship the experiment.)

This is the Phase 28 test: switch base model and see if F1 breaks 0.40.
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
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_lora import DATA_RAW  # CSV path constant

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE_MODEL = 'roberta-base'
NUM_LAYERS = 12
MAX_LEN = 256   # RoBERTa-base supports up to 512; bump from DistilBERT's 128 default


def build_target_modules(target, layer_scope):
    """RoBERTa attention naming (HuggingFace): query, key, value, attention.output.dense.
    layer path: encoder.layer.{i}.attention.self.{query,key,value}.

    For now we support 'qv' (query+value) and 'qkv' (q+k+v).
    layer_scope: 'top6' (layers 6..11), 'top4' (8..11), 'all12' (0..11).
    """
    if layer_scope == 'all12':
        layer_idxs = list(range(NUM_LAYERS))
    elif layer_scope == 'top6':
        layer_idxs = list(range(NUM_LAYERS - 6, NUM_LAYERS))
    elif layer_scope == 'top4':
        layer_idxs = list(range(NUM_LAYERS - 4, NUM_LAYERS))
    else:
        raise ValueError(f'unknown layer_scope: {layer_scope}')

    if target == 'qv':
        suffixes = ['query', 'value']
    elif target == 'qkv':
        suffixes = ['query', 'key', 'value']
    else:
        raise ValueError(f'unknown target: {target}')
    return [f'encoder.layer.{i}.attention.self.{s}' for i in layer_idxs for s in suffixes]


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


class StoryRespDatasetOpenerSwap(Dataset):
    """Per-sample retokenization with optional opener swap. Same logic as in
    run_lora_story.py — kept self-contained here to avoid cross-imports."""
    def __init__(self, stories_text, responses_text, y, tokenizer, max_len,
                  swap_p=0.0, swap_n_words=10, opener_bank=None,
                  bank_by_class=False, strip=False):
        self.stories_text = stories_text
        self.responses_text = responses_text
        self.y = y
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.swap_p = float(swap_p)
        self.swap_n_words = int(swap_n_words)
        self.opener_bank = opener_bank
        self.bank_by_class = bank_by_class
        self.strip = strip
    def __len__(self):
        return len(self.responses_text)
    def __getitem__(self, idx):
        resp = self.responses_text[idx]
        words = resp.split()
        if self.strip:
            resp = ' '.join(words[self.swap_n_words:])
        elif self.swap_p > 0 and np.random.rand() < self.swap_p:
            if self.bank_by_class:
                cls = int(np.argmax(np.asarray(self.y[idx])))
                bank = self.opener_bank[cls]
            else:
                bank = self.opener_bank
            opener = bank[np.random.randint(len(bank))]
            resp = opener + ' ' + ' '.join(words[self.swap_n_words:])
        enc_s = self.tokenizer(self.stories_text[idx], truncation=True,
                                padding='max_length', max_length=self.max_len,
                                return_tensors='pt')
        enc_r = self.tokenizer(resp, truncation=True, padding='max_length',
                                max_length=self.max_len, return_tensors='pt')
        return {
            's_iids': enc_s['input_ids'][0],
            's_mask': enc_s['attention_mask'][0],
            'r_iids': enc_r['input_ids'][0],
            'r_mask': enc_r['attention_mask'][0],
            'labels': torch.as_tensor(self.y[idx], dtype=torch.float32),
        }


class LoRARobertaStoryResp(nn.Module):
    def __init__(self, lora_cfg, mlp_hidden=256, dropout=0.5, skip_conn=False):
        super().__init__()
        base = AutoModel.from_pretrained(BASE_MODEL)
        self.bert = get_peft_model(base, lora_cfg)
        self.skip_conn = skip_conn
        head_in = 768 * 2 * (2 if skip_conn else 1)
        self.head = nn.Sequential(
            nn.Linear(head_in, mlp_hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(mlp_hidden, 3),
        )
    def _enc(self, iids, mask):
        lora = self.bert(input_ids=iids, attention_mask=mask).last_hidden_state[:, 0, :]
        if not self.skip_conn:
            return lora
        with torch.no_grad():
            with self.bert.disable_adapter():
                frozen = self.bert(input_ids=iids, attention_mask=mask).last_hidden_state[:, 0, :]
        return torch.cat([frozen, lora], dim=1)
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

    train_resp_text = [responses[i] for i in train_idx]
    train_stor_text = [stories[i]   for i in train_idx]
    train_y_list    = soft[train_idx].tolist()

    tr_s_iids, tr_s_mask = tok(train_stor_text)
    tr_r_iids, tr_r_mask = tok(train_resp_text)
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
        'tr_resp_text': train_resp_text, 'tr_stor_text': train_stor_text,
        'tr_y_list':    train_y_list,
        'ev_resp_text': [responses[i] for i in eval_idx], 'ev_stor_text': [stories[i] for i in eval_idx],
        'te_resp_text': [responses[i] for i in test_idx], 'te_stor_text': [stories[i] for i in test_idx],
    }


def build_opener_bank(train_resp_text, train_y, n_words=10, by_class=False):
    openers = [' '.join(t.split()[:n_words]) for t in train_resp_text]
    if not by_class:
        return [o for o in openers if o]
    y = np.asarray(train_y)
    arg = y.argmax(axis=1)
    bank = {k: [] for k in range(3)}
    for o, k in zip(openers, arg):
        if o:
            bank[int(k)].append(o)
    return bank


def train_one_seed(args, data, tokenizer, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    target_modules = build_target_modules(args.target, args.layer_scope)
    lora_cfg = LoraConfig(
        r=args.rank, lora_alpha=args.alpha if args.alpha > 0 else args.rank,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout, bias=args.bias,
        task_type='FEATURE_EXTRACTION',
    )
    model = LoRARobertaStoryResp(lora_cfg, mlp_hidden=256, dropout=args.head_dropout,
                                    skip_conn=args.skip_conn).to(DEVICE)

    if args.lora_lr is not None and args.lora_lr > 0:
        lora_params, head_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if 'lora_' in name:
                lora_params.append(p)
            else:
                head_params.append(p)
        opt = torch.optim.AdamW(
            [{'params': lora_params, 'lr': args.lora_lr},
             {'params': head_params, 'lr': args.lr}],
            weight_decay=args.wd,
        )
        if seed == 0:
            print(f'  decoupled lrs: lora_lr={args.lora_lr}  head_lr={args.lr}  '
                  f'(lora_params={sum(p.numel() for p in lora_params)}, '
                  f'head_params={sum(p.numel() for p in head_params)})')
            print(f'  target_modules ({len(target_modules)}): {target_modules[:3]} ...')
    else:
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=args.wd,
        )

    ty = data['tr_y']
    arg = ty.argmax(dim=1).numpy()
    counts = np.array([(arg == k).sum() for k in range(3)], dtype=float)
    w = 1.0 / counts[arg]
    sampler = WeightedRandomSampler(weights=torch.tensor(w), num_samples=len(w), replacement=True)

    if args.opener_swap_p > 0:
        bank_by_class = (args.opener_bank == 'same_class')
        opener_bank = build_opener_bank(data['tr_resp_text'], data['tr_y_list'],
                                          n_words=args.opener_swap_n_words,
                                          by_class=bank_by_class)
        tr_ds = StoryRespDatasetOpenerSwap(
            data['tr_stor_text'], data['tr_resp_text'], data['tr_y_list'],
            tokenizer, MAX_LEN,
            swap_p=args.opener_swap_p, swap_n_words=args.opener_swap_n_words,
            opener_bank=opener_bank, bank_by_class=bank_by_class, strip=False)
        ev_ds = StoryRespDataset(data['ev_s_iids'], data['ev_s_mask'],
                                   data['ev_r_iids'], data['ev_r_mask'], data['ev_y'])
        te_ds = StoryRespDataset(data['te_s_iids'], data['te_s_mask'],
                                   data['te_r_iids'], data['te_r_mask'], data['te_y'])
        if seed == 0:
            n_bank = (sum(len(v) for v in opener_bank.values()) if bank_by_class
                      else len(opener_bank))
            print(f'  opener_swap_p={args.opener_swap_p} bank={args.opener_bank} '
                  f'(n_openers={n_bank}, swap_n_words={args.opener_swap_n_words})')
    else:
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
    # Optionally dump model state for downstream interpretability work.
    if getattr(args, 'save_state', None):
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
        os.makedirs(models_dir, exist_ok=True)
        if isinstance(args.save_state, str) and args.save_state:
            fname = args.save_state if args.save_state.endswith('.pt') else f'{args.save_state}.pt'
        else:
            fname = f'{getattr(args, "tag", "run") or "run"}_seed{seed}.pt'
        out_path = os.path.join(models_dir, fname)
        torch.save({
            'state_dict': model.state_dict(),
            'args': vars(args),
            'base_model': BASE_MODEL,
            'seed': seed,
            'test_f1': float(f1_score(trues, preds, labels=[0, 1, 2], average='macro', zero_division=0)),
            'test_acc': float(accuracy_score(trues, preds)),
            'epochs': epoch + 1,
        }, out_path)
        print(f'  [seed {seed}] saved model state to {out_path}')
    return {
        'acc': float(accuracy_score(trues, preds)),
        'f1':  float(f1_score(trues, preds, labels=[0, 1, 2], average='macro', zero_division=0)),
        'epochs': epoch + 1,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rank', type=int, required=True)
    p.add_argument('--target', type=str, default='qv', choices=['qv', 'qkv'])
    p.add_argument('--layer_scope', type=str, default='top6', choices=['top4', 'top6', 'all12'])
    p.add_argument('--alpha', type=int, default=-1)
    p.add_argument('--bias', type=str, default='none')
    p.add_argument('--head_dropout', type=float, default=0.5)
    p.add_argument('--lora_dropout', type=float, default=0.0)
    p.add_argument('--lr',      type=float, default=3e-4, help='Head learning rate.')
    p.add_argument('--lora_lr', type=float, default=3e-5)
    p.add_argument('--wd',      type=float, default=0.01)
    p.add_argument('--skip_conn', action='store_true')
    p.add_argument('--opener_swap_p', type=float, default=0.0)
    p.add_argument('--opener_swap_n_words', type=int, default=10)
    p.add_argument('--opener_bank', type=str, default='same_class',
                   choices=['cross_class', 'same_class'])
    p.add_argument('--max_epochs', type=int, default=60)
    p.add_argument('--patience',   type=int, default=10)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--seeds',      type=int, default=5)
    p.add_argument('--seed_offset', type=int, default=0)
    p.add_argument('--tag', type=str, default='')
    p.add_argument('--save_state', type=str, default=None, nargs='?', const='',
                   help='Save model state_dict to models/{filename}.pt for interpretability work.')
    args = p.parse_args()

    print(f'Device: {DEVICE}\nBase: {BASE_MODEL}\nConfig: {vars(args)}')
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    data = load_data(tokenizer)
    print(f'sizes: tr={len(data["tr_y"])} ev={len(data["ev_y"])} te={len(data["te_y"])}')

    results = []
    for i in range(args.seeds):
        s = args.seed_offset + i
        r = train_one_seed(args, data, tokenizer, seed=s)
        r['seed'] = s
        results.append(r)
        print(f'  seed {s}: acc {r["acc"]:.4f}  F1 {r["f1"]:.4f}  ep {r["epochs"]}')

    accs = np.array([r['acc'] for r in results])
    f1s  = np.array([r['f1']  for r in results])
    eps  = np.array([r['epochs'] for r in results])
    summary = {
        'args': vars(args), 'base_model': BASE_MODEL,
        'acc_mean': float(accs.mean()), 'acc_std': float(accs.std()),
        'f1_mean':  float(f1s.mean()),  'f1_std':  float(f1s.std()),
        'ep_mean':  float(eps.mean()),
        'seed_results': results,
    }
    print(f'\nSummary  acc {accs.mean():.4f}±{accs.std():.4f}   '
          f'F1 {f1s.mean():.4f}±{f1s.std():.4f}   ep {eps.mean():.1f}')

    os.makedirs(OUT_DIR, exist_ok=True)
    alpha_tag = (args.alpha if args.alpha > 0 else args.rank)
    tag_extra = f'_{args.tag}' if args.tag else ''
    tag = (f'roberta_story_r{args.rank}_a{alpha_tag}_{args.target}_{args.layer_scope}'
           f'_lr{args.lr:.0e}_lora_lr{args.lora_lr:.0e}_wd{args.wd:.0e}'
           f'_ep{args.max_epochs}_pat{args.patience}_n{args.seeds}{tag_extra}')
    with open(os.path.join(OUT_DIR, f'{tag}.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved outputs/{tag}.json')


if __name__ == '__main__':
    main()
