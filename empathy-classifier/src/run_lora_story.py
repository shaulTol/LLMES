"""Story + Response LoRA fine-tune.

Architecture: one LoRA-adapted DistilBERT, applied separately to (story, response).
Concat the two [CLS] vectors -> MLP head.

The 1536-d head matches the frozen `story_plus_response_1536` setup which gave
F1 = 0.378 at 100 seeds — the strongest signal we have.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_lora import build_target_modules, _apply_text_aug, _word_mask_aug, NUM_LAYERS, MAX_LEN, DATA_RAW

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    """Per-sample, per-call retokenization with optional opener swap / strip.

    At each __getitem__, with prob `swap_p` replace the first `swap_n_words` of the
    response with one sampled uniformly from `opener_bank` (a list of strings, or
    a dict {class_idx: [strings]} if `bank_by_class=True`). If `strip` is True,
    just drop the first `swap_n_words` (no swap). Story is never modified.

    Slower than pre-tokenized fast path (re-tokenizes each access) but required
    because the swap is stochastic per epoch.
    """
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


def pool_tokens(h, mask, pool, attn_module=None):
    """h: [B, T, D]; mask: [B, T] (1 for real tokens). Returns [B, D] (or [B, 3D] for cls_mean_max)."""
    if pool == 'cls':
        return h[:, 0, :]
    m = mask.unsqueeze(-1).float()  # [B, T, 1]
    if pool == 'mean':
        return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
    if pool == 'attn':
        scores = attn_module(h).squeeze(-1)  # [B, T]
        scores = scores.masked_fill(mask == 0, -1e9)
        w = scores.softmax(dim=-1).unsqueeze(-1)  # [B, T, 1]
        return (h * w).sum(dim=1)
    if pool == 'cls_mean_max':
        cls = h[:, 0, :]
        mean = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        masked_h = h.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
        mx = masked_h.max(dim=1).values
        return torch.cat([cls, mean, mx], dim=1)  # [B, 3D]
    raise ValueError(f'unknown pool: {pool}')


class LoRABertStoryResp(nn.Module):
    def __init__(self, lora_cfg, mlp_hidden=256, mlp_hidden2=0, dropout=0.3,
                  latent_sigma=0.0, frozen_latent_sigma=0.0,
                  skip_conn=False, mixup_alpha=0.0,
                  pool='cls'):
        super().__init__()
        base = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.bert = get_peft_model(base, lora_cfg)
        self.skip_conn = skip_conn
        self.pool = pool
        # Per-text pooled width: 768 for cls/mean/attn, 3*768 = 2304 for cls_mean_max.
        per_text_dim = 768 * (3 if pool == 'cls_mean_max' else 1)
        # Skip: 2 branches (frozen + lora) per text; else 1 branch per text. 2 texts (story, resp).
        head_in = per_text_dim * 2 * (2 if skip_conn else 1)
        # Attention pool queries (only used when pool == 'attn'). Separate per branch
        # so frozen and lora signals can attend differently.
        if pool == 'attn':
            self.attn_q_lora = nn.Linear(768, 1, bias=False)
            if skip_conn:
                self.attn_q_frozen = nn.Linear(768, 1, bias=False)
            else:
                self.attn_q_frozen = None
        else:
            self.attn_q_lora = None
            self.attn_q_frozen = None
        # MLP head: 1 hidden layer if mlp_hidden2 == 0, else 2 hidden layers.
        head_layers = [nn.Linear(head_in, mlp_hidden), nn.GELU(), nn.Dropout(dropout)]
        if mlp_hidden2 > 0:
            head_layers += [nn.Linear(mlp_hidden, mlp_hidden2), nn.GELU(), nn.Dropout(dropout),
                             nn.Linear(mlp_hidden2, 3)]
        else:
            head_layers += [nn.Linear(mlp_hidden, 3)]
        self.head = nn.Sequential(*head_layers)
        self.latent_sigma = latent_sigma
        self.frozen_latent_sigma = frozen_latent_sigma
        self.mixup_alpha = mixup_alpha
    def _enc(self, iids, mask):
        # Full token states (B, T, 768); LoRA-adapted (gradients flow through LoRA).
        h_lora = self.bert(input_ids=iids, attention_mask=mask).last_hidden_state
        lora = pool_tokens(h_lora, mask, self.pool, attn_module=self.attn_q_lora)
        if self.training and self.latent_sigma > 0:
            with torch.no_grad():
                s = lora.std(dim=0, keepdim=True) + 1e-6
            lora = lora + torch.randn_like(lora) * self.latent_sigma * s
        if not self.skip_conn:
            return lora
        # Frozen branch: same model with adapters disabled, no grad needed.
        with torch.no_grad():
            with self.bert.disable_adapter():
                h_frozen = self.bert(input_ids=iids, attention_mask=mask).last_hidden_state
        frozen = pool_tokens(h_frozen, mask, self.pool, attn_module=self.attn_q_frozen)
        if self.training and self.frozen_latent_sigma > 0:
            with torch.no_grad():
                s = frozen.std(dim=0, keepdim=True) + 1e-6
            frozen = frozen + torch.randn_like(frozen) * self.frozen_latent_sigma * s
        return torch.cat([frozen, lora], dim=1)
    def forward(self, s_iids, s_mask, r_iids, r_mask, labels=None):
        s = self._enc(s_iids, s_mask)
        r = self._enc(r_iids, r_mask)
        feat = torch.cat([s, r], dim=1)
        # Mixup at the concatenated CLS level (training only). Returns mixed labels.
        if self.training and self.mixup_alpha > 0 and labels is not None:
            B = feat.size(0)
            lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
            perm = torch.randperm(B, device=feat.device)
            feat = lam * feat + (1 - lam) * feat[perm]
            labels_mix = lam * labels + (1 - lam) * labels[perm]
            return self.head(feat), labels_mix
        return self.head(feat)


def soft_ce(logits, target):
    return -(target * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()


def soft_ce_smooth(logits, target, eps=0.1):
    """Soft CE with label smoothing applied to the soft target distribution."""
    K = target.size(1)
    target = target * (1 - eps) + eps / K
    return -(target * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()


def soft_focal(logits, target, gamma=2.0):
    """Soft focal loss: focal_weight × soft CE, where focal_weight is computed per-sample
    against the predicted probability of the argmax-true class. Down-weights easy examples
    (those the model already gets right with high confidence)."""
    log_probs = torch.log_softmax(logits, dim=1)
    probs = log_probs.exp()
    # Per-sample focal weight: (1 - p_true)^gamma where p_true is prob mass on argmax(target).
    arg_true = target.argmax(dim=1)
    p_true = probs.gather(1, arg_true.unsqueeze(1)).squeeze(1).clamp(min=1e-8, max=1 - 1e-8)
    focal_w = (1.0 - p_true).pow(gamma)
    losses = -(target * log_probs).sum(dim=1)
    return (focal_w * losses).mean()


def make_loss(name, focal_gamma=2.0, label_smoothing=0.1):
    if name == 'soft_ce':
        return soft_ce
    if name == 'soft_ce_ls':
        return lambda lg, t: soft_ce_smooth(lg, t, eps=label_smoothing)
    if name == 'focal':
        return lambda lg, t: soft_focal(lg, t, gamma=focal_gamma)
    raise ValueError(f'unknown loss: {name}')


def load_data_story_resp(tokenizer, rng_seed=42, aug_mode='balanced_samp', aug_target=0):
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

    # Text aug applied to both story AND response (same masking pattern not required;
    # independent random masking is fine since they're separate sentence inputs).
    train_resp_in = [responses[i] for i in train_idx]
    train_stor_in = [stories[i]   for i in train_idx]
    train_y_in    = soft[train_idx].tolist()

    if aug_mode in ('text_min_to_max', 'text_all_to_target'):
        # joint indexing: pick rows to augment, mask both story and response
        labels_arr = np.array(train_y_in); arg = labels_arr.argmax(axis=1)
        counts = np.bincount(arg, minlength=3)
        per_class_target = int(counts.max()) if aug_mode == 'text_min_to_max' else int(aug_target)
        rng = np.random.default_rng(rng_seed)
        new_resp = list(train_resp_in); new_stor = list(train_stor_in); new_y = list(train_y_in)
        for k in range(3):
            if counts[k] >= per_class_target: continue
            idx_k = np.where(arg == k)[0]
            need = per_class_target - counts[k]
            for _ in range(int(need)):
                src = idx_k[rng.integers(len(idx_k))]
                new_resp.append(_word_mask_aug(train_resp_in[src], 0.15, '[MASK]', rng))
                new_stor.append(_word_mask_aug(train_stor_in[src], 0.15, '[MASK]', rng))
                new_y.append(train_y_in[src])
        train_resp_in, train_stor_in, train_y_in = new_resp, new_stor, new_y
    elif aug_mode == 'text_bertfill_all_to_target':
        # Load pre-computed BERT-filled variants from JSON cache.
        cache_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed',
                                   f'bertfill_aug_tgt{int(aug_target)}.json')
        import json as _json
        with open(cache_path) as f:
            cache = _json.load(f)
        labels_arr = np.array(train_y_in); arg = labels_arr.argmax(axis=1)
        rng = np.random.default_rng(rng_seed)
        new_resp = list(train_resp_in); new_stor = list(train_stor_in); new_y = list(train_y_in)
        # Each train row -> its BERT-filled variants
        for ti, src in enumerate(train_idx):
            key = str(int(src))
            if key not in cache:
                continue
            variants = cache[key]
            for s_text, r_text in zip(variants['stories'], variants['responses']):
                new_stor.append(s_text); new_resp.append(r_text); new_y.append(train_y_in[ti])
        train_resp_in, train_stor_in, train_y_in = new_resp, new_stor, new_y

    enc_tr_s = tokenizer(train_stor_in, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt')
    enc_tr_r = tokenizer(train_resp_in, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt')
    enc_ev_s = tokenizer([stories[i]   for i in eval_idx], truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt')
    enc_ev_r = tokenizer([responses[i] for i in eval_idx], truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt')
    enc_te_s = tokenizer([stories[i]   for i in test_idx], truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt')
    enc_te_r = tokenizer([responses[i] for i in test_idx], truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt')

    return {
        'tr_s_iids': enc_tr_s['input_ids'], 'tr_s_mask': enc_tr_s['attention_mask'],
        'tr_r_iids': enc_tr_r['input_ids'], 'tr_r_mask': enc_tr_r['attention_mask'],
        'tr_y': torch.tensor(np.array(train_y_in), dtype=torch.float32),
        'ev_s_iids': enc_ev_s['input_ids'], 'ev_s_mask': enc_ev_s['attention_mask'],
        'ev_r_iids': enc_ev_r['input_ids'], 'ev_r_mask': enc_ev_r['attention_mask'],
        'ev_y': torch.tensor(soft[eval_idx], dtype=torch.float32),
        'te_s_iids': enc_te_s['input_ids'], 'te_s_mask': enc_te_s['attention_mask'],
        'te_r_iids': enc_te_r['input_ids'], 'te_r_mask': enc_te_r['attention_mask'],
        'te_y': torch.tensor(soft[test_idx], dtype=torch.float32),
        # Raw text + labels, kept for opener-swap / opener-strip path which retokenizes per __getitem__.
        'tr_resp_text': train_resp_in,
        'tr_stor_text': train_stor_in,
        'tr_y_list':    train_y_in,
        'ev_resp_text': [responses[i] for i in eval_idx],
        'ev_stor_text': [stories[i]   for i in eval_idx],
        'te_resp_text': [responses[i] for i in test_idx],
        'te_stor_text': [stories[i]   for i in test_idx],
    }


def build_opener_bank(train_resp_text, train_y, n_words=10, by_class=False):
    """Bank of first-`n_words` substrings from training responses.

    by_class=False -> single flat list (cross-class).
    by_class=True  -> dict {class_idx: [openers]} stratified by argmax-class.
    """
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


def train_one_seed(args, data, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    lora_cfg = LoraConfig(
        r=args.rank, lora_alpha=args.alpha if args.alpha > 0 else args.rank,
        target_modules=build_target_modules(args.target, args.layer_scope),
        lora_dropout=args.lora_dropout, bias=args.bias,
        task_type='FEATURE_EXTRACTION',
    )
    model = LoRABertStoryResp(lora_cfg,
                                mlp_hidden=args.mlp_hidden,
                                mlp_hidden2=args.mlp_hidden2,
                                dropout=args.head_dropout,
                                latent_sigma=args.latent_sigma,
                                frozen_latent_sigma=args.frozen_latent_sigma,
                                skip_conn=args.skip_conn,
                                mixup_alpha=args.mixup_alpha,
                                pool=args.pool).to(DEVICE)
    # Loss function (per --loss flag).
    loss_fn = make_loss(args.loss, focal_gamma=args.focal_gamma,
                         label_smoothing=args.label_smoothing)

    # Decoupled lrs: `lora_lr` for LoRA adapters, `lr` for the head (and any other trainable).
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
    else:
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=args.wd,
        )

    # Optional LR scheduler. 'cosine' = linear warmup over args.warmup_epochs, then
    # cosine decay to 0 over the remaining max_epochs. Applies the same multiplier to
    # all param groups (so decoupled lrs decay proportionally).
    scheduler = None
    if args.lr_schedule == 'cosine':
        warmup = max(0, int(args.warmup_epochs))
        total  = max(1, int(args.max_epochs))
        def _lr_lambda(epoch):
            if epoch < warmup:
                return (epoch + 1) / max(1, warmup)
            progress = (epoch - warmup) / max(1, total - warmup)
            return 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)
        if seed == 0:
            print(f'  LR schedule: cosine, warmup={warmup} epochs, total={total} epochs')

    ty = data['tr_y']
    if args.hard_labels:
        # Convert soft labels to one-hot of argmax — train as if labels were
        # categorical decisions rather than aggregated continuous ratings.
        arg = ty.argmax(dim=1)
        ty_hard = torch.zeros_like(ty)
        ty_hard.scatter_(1, arg.unsqueeze(1), 1.0)
        ty = ty_hard
        data['tr_y'] = ty
        if seed == 0:
            print(f'  hard labels: converted {ty.shape[0]} soft labels to one-hot argmax')
    if args.aug_mode == 'balanced_samp':
        arg = ty.argmax(dim=1).numpy()
        counts = np.array([(arg == k).sum() for k in range(3)], dtype=float)
        w = 1.0 / counts[arg]
        sampler = WeightedRandomSampler(weights=torch.tensor(w), num_samples=len(w), replacement=True)
        shuffle = False
    else:
        sampler = None; shuffle = True

    use_swap_ds = (args.opener_swap_p > 0 or args.opener_strip_n > 0)
    if use_swap_ds:
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        bank_by_class = (args.opener_bank == 'same_class')
        opener_bank = build_opener_bank(data['tr_resp_text'], data['tr_y_list'],
                                         n_words=args.opener_swap_n_words,
                                         by_class=bank_by_class)
        if args.opener_strip_n > 0:
            # Strip at TRAIN and TEST (and eval) — the whole point is "can the body alone classify?".
            tr_ds = StoryRespDatasetOpenerSwap(
                data['tr_stor_text'], data['tr_resp_text'], data['tr_y_list'],
                tokenizer, MAX_LEN, swap_p=0.0, swap_n_words=args.opener_strip_n,
                opener_bank=None, bank_by_class=False, strip=True)
            ev_ds = StoryRespDatasetOpenerSwap(
                data['ev_stor_text'], data['ev_resp_text'], data['ev_y'].tolist(),
                tokenizer, MAX_LEN, swap_p=0.0, swap_n_words=args.opener_strip_n,
                opener_bank=None, bank_by_class=False, strip=True)
            te_ds = StoryRespDatasetOpenerSwap(
                data['te_stor_text'], data['te_resp_text'], data['te_y'].tolist(),
                tokenizer, MAX_LEN, swap_p=0.0, swap_n_words=args.opener_strip_n,
                opener_bank=None, bank_by_class=False, strip=True)
            if seed == 0:
                print(f'  opener_strip_n={args.opener_strip_n} applied at train+eval+test')
        else:
            # Swap: train-time stochastic opener replacement. Eval/test stay clean (pre-tokenized).
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
        # WeightedRandomSampler still works with the swap dataset (drives the index stream).
    else:
        tr_ds = StoryRespDataset(data['tr_s_iids'], data['tr_s_mask'], data['tr_r_iids'], data['tr_r_mask'], ty)
        ev_ds = StoryRespDataset(data['ev_s_iids'], data['ev_s_mask'], data['ev_r_iids'], data['ev_r_mask'], data['ev_y'])
        te_ds = StoryRespDataset(data['te_s_iids'], data['te_s_mask'], data['te_r_iids'], data['te_r_mask'], data['te_y'])
    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler)
    ev_ld = DataLoader(ev_ds, batch_size=args.batch_size)
    te_ld = DataLoader(te_ds, batch_size=args.batch_size)

    def _f1_over_loader(ld):
        preds, trues = [], []
        with torch.no_grad():
            for b in ld:
                si, sm = b['s_iids'].to(DEVICE), b['s_mask'].to(DEVICE)
                ri, rm = b['r_iids'].to(DEVICE), b['r_mask'].to(DEVICE)
                preds.append(model(si, sm, ri, rm).argmax(dim=1).cpu().numpy())
                trues.append(b['labels'].argmax(dim=1).cpu().numpy())
        preds = np.concatenate(preds); trues = np.concatenate(trues)
        return f1_score(trues, preds, labels=[0, 1, 2], average='macro', zero_division=0)

    # Early-stop tracking. f1 → maximize, loss → minimize.
    es_metric = getattr(args, 'early_stop_metric', 'loss')
    if es_metric == 'f1':
        best = -float('inf'); cmp = (lambda new, old: new > old)
    else:
        best = float('inf'); cmp = (lambda new, old: new < old)
    bad = 0; best_state = None
    per_epoch = []
    trainable = [p for p in model.parameters() if p.requires_grad]
    for epoch in range(args.max_epochs):
        model.train()
        train_loss_sum, train_n = 0.0, 0
        for b in tr_ld:
            si, sm = b['s_iids'].to(DEVICE), b['s_mask'].to(DEVICE)
            ri, rm = b['r_iids'].to(DEVICE), b['r_mask'].to(DEVICE)
            y = b['labels'].to(DEVICE)
            opt.zero_grad()
            if args.mixup_alpha > 0:
                logits, y_mix = model(si, sm, ri, rm, labels=y)
                loss = loss_fn(logits, y_mix)
            else:
                loss = loss_fn(model(si, sm, ri, rm), y)
            if args.l1 > 0:
                loss = loss + args.l1 * sum(p.abs().sum() for p in trainable)
            loss.backward(); opt.step()
            train_loss_sum += loss.item(); train_n += 1
        model.eval()
        ev_loss = 0.0
        with torch.no_grad():
            for b in ev_ld:
                si, sm = b['s_iids'].to(DEVICE), b['s_mask'].to(DEVICE)
                ri, rm = b['r_iids'].to(DEVICE), b['r_mask'].to(DEVICE)
                y = b['labels'].to(DEVICE)
                ev_loss += loss_fn(model(si, sm, ri, rm), y).item()
        ev_loss /= len(ev_ld)
        tr_loss = train_loss_sum / max(1, train_n)
        # Per-epoch train/eval F1 (only seed 0 to keep logs manageable; cheap test-set F1 too)
        if seed == 0 and args.log_curve:
            tr_f1 = _f1_over_loader(tr_ld)
            ev_f1 = _f1_over_loader(ev_ld)
            te_f1 = _f1_over_loader(te_ld)
            per_epoch.append({'ep': epoch + 1, 'tr_loss': tr_loss, 'ev_loss': ev_loss,
                              'tr_f1': float(tr_f1), 'ev_f1': float(ev_f1), 'te_f1': float(te_f1)})
            print(f'    epoch {epoch+1:3d}  tr_loss {tr_loss:.4f}  ev_loss {ev_loss:.4f}  '
                  f'tr_F1 {tr_f1:.4f}  ev_F1 {ev_f1:.4f}  te_F1 {te_f1:.4f}')
        # Compute eval F1 only if we need it for early stopping (or curve logging).
        if es_metric == 'f1':
            ev_f1 = _f1_over_loader(ev_ld)
            score = ev_f1
        else:
            score = ev_loss
        if cmp(score, best):
            best = score; bad = 0
            best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience: break
        if scheduler is not None:
            scheduler.step()

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    model.eval()
    preds, trues, probs = [], [], []
    with torch.no_grad():
        for b in te_ld:
            si, sm = b['s_iids'].to(DEVICE), b['s_mask'].to(DEVICE)
            ri, rm = b['r_iids'].to(DEVICE), b['r_mask'].to(DEVICE)
            logits = model(si, sm, ri, rm)
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            preds.append(logits.argmax(dim=1).cpu().numpy())
            trues.append(b['labels'].argmax(dim=1).cpu().numpy())
    preds = np.concatenate(preds); trues = np.concatenate(trues)
    probs = np.concatenate(probs)
    # Optionally save predictions for failure-mode mining (A2-style).
    if getattr(args, 'save_predictions', False):
        soft_true = data['te_y'].numpy()
        out_path = os.path.join(OUT_DIR, f'preds_seed{seed}_{getattr(args, "tag", "run")}.npz')
        np.savez(out_path, probs=probs, preds=preds, trues=trues, soft_labels=soft_true)
        print(f'  [seed {seed}] saved predictions to {out_path}')
    # Optionally dump the full model state for later interpretability work.
    if getattr(args, 'save_state', None):
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
        os.makedirs(models_dir, exist_ok=True)
        # If --save_state is a non-empty string, use it as the filename; otherwise default by tag/seed.
        if isinstance(args.save_state, str) and args.save_state:
            fname = args.save_state if args.save_state.endswith('.pt') else f'{args.save_state}.pt'
        else:
            fname = f'{getattr(args, "tag", "run") or "run"}_seed{seed}.pt'
        out_path = os.path.join(models_dir, fname)
        torch.save({
            'state_dict': model.state_dict(),
            'args': vars(args),
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
        'per_epoch': per_epoch,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rank', type=int, required=True)
    p.add_argument('--target', type=str, default='qv', choices=['qv', 'qkvo', 'qkvo_ffn'])
    p.add_argument('--layer_scope', type=str, default='all6', choices=['top2', 'top3', 'all6'])
    p.add_argument('--alpha', type=int, default=-1)
    p.add_argument('--bias', type=str, default='none')
    p.add_argument('--head_dropout', type=float, default=0.3)
    p.add_argument('--lora_dropout', type=float, default=0.0)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--lora_lr', type=float, default=None,
                   help='If set, use separate lr for LoRA params; --lr applies to head only.')
    p.add_argument('--wd', type=float, default=0.01)
    p.add_argument('--l1', type=float, default=0.0,
                   help='L1 coefficient on all trainable params (LoRA + head). 0 = off.')
    p.add_argument('--early_stop_metric', type=str, default='loss',
                   choices=['loss', 'f1'],
                   help='loss → minimize eval CE; f1 → maximize eval macro F1.')
    p.add_argument('--log_curve', action='store_true',
                   help='Log per-epoch train/eval/test F1 for seed 0 (training curve diagnostic).')
    p.add_argument('--skip_conn', action='store_true',
                   help='Skip connection: head sees both frozen-BERT and LoRA-adapted [CLS].')
    p.add_argument('--save_predictions', action='store_true',
                   help='Save per-example test probs/preds to outputs/preds_seed{S}_{tag}.npz.')
    p.add_argument('--save_state', type=str, default=None, nargs='?', const='',
                   help='Save model state_dict to models/{filename}.pt (default: {tag}_seed{seed}.pt).')
    p.add_argument('--mixup_alpha', type=float, default=0.0,
                   help='Mixup interpolation at concatenated CLS during training; 0 = off.')
    p.add_argument('--latent_sigma', type=float, default=0.0,
                   help='Gaussian noise sigma applied to LoRA-adapted [CLS] (trainable branch).')
    p.add_argument('--frozen_latent_sigma', type=float, default=0.0,
                   help='Gaussian noise sigma applied to FROZEN [CLS] (skip-connection branch only). '
                        'Regularizes the "shortcut" path without harming LoRA gradient flow.')
    p.add_argument('--hard_labels', action='store_true',
                   help='Convert training soft labels to one-hot of argmax. Tests whether '
                        'training on categorical labels (vs continuous-aggregated soft labels) '
                        'gives sharper gradient signal for minority classes.')
    p.add_argument('--aug_mode', type=str, default='balanced_samp',
                   choices=['none', 'balanced_samp', 'text_min_to_max', 'text_all_to_target',
                            'text_bertfill_all_to_target'])
    p.add_argument('--aug_target', type=int, default=0)
    p.add_argument('--max_epochs', type=int, default=60)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--seeds', type=int, default=10)
    p.add_argument('--seed_offset', type=int, default=0,
                   help='Use seeds [offset, offset+seeds). For 100-seed runs across array tasks.')
    # Pooling sweep (phase 22). Default 'cls' = current behavior.
    p.add_argument('--pool', type=str, default='cls',
                   choices=['cls', 'mean', 'attn', 'cls_mean_max'],
                   help='How to collapse BERT token states per text. cls is current behavior.')
    # Opener interventions (phase 23). All default to off.
    p.add_argument('--opener_swap_p', type=float, default=0.0,
                   help='Prob of replacing first opener_swap_n_words of response with one sampled '
                        'uniformly from the training opener bank. 0 = off.')
    p.add_argument('--opener_swap_n_words', type=int, default=10,
                   help='How many leading words the swap replaces.')
    p.add_argument('--opener_bank', type=str, default='cross_class',
                   choices=['cross_class', 'same_class'],
                   help='Opener bank construction. cross_class = flat (any opener, any label). '
                        'same_class = stratified (within true class).')
    p.add_argument('--opener_strip_n', type=int, default=0,
                   help='If > 0, drop the first N words of every response at train, eval, AND test. '
                        'Hard test: can the body alone classify?')
    # Loss variants (phase 27).
    p.add_argument('--loss', type=str, default='soft_ce',
                   choices=['soft_ce', 'soft_ce_ls', 'focal'],
                   help='soft_ce (default), soft_ce_ls (label smoothing), focal (focal loss).')
    p.add_argument('--focal_gamma', type=float, default=2.0)
    p.add_argument('--label_smoothing', type=float, default=0.1)
    # Head config (phase 27).
    p.add_argument('--mlp_hidden',  type=int, default=256)
    p.add_argument('--mlp_hidden2', type=int, default=0,
                   help='If > 0, head has 2 hidden layers (mlp_hidden, mlp_hidden2).')
    # LR schedule (phase 27).
    p.add_argument('--lr_schedule', type=str, default='flat',
                   choices=['flat', 'cosine'],
                   help='Flat (default) or cosine with linear warmup.')
    p.add_argument('--warmup_epochs', type=int, default=0)
    p.add_argument('--tag', type=str, default='')
    args = p.parse_args()

    print(f'Device: {DEVICE}\nConfig: {vars(args)}')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    data = load_data_story_resp(tokenizer, aug_mode=args.aug_mode, aug_target=args.aug_target)
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
    alpha_tag = (args.alpha if args.alpha > 0 else args.rank)
    aug_tag = args.aug_mode + (f'{args.aug_target}' if args.aug_target > 0 else '')
    tag_extra = f'_{args.tag}' if args.tag else ''
    tag = (f'lorastory_r{args.rank}_a{alpha_tag}_{args.target}_{args.layer_scope}'
           f'_lr{args.lr:.0e}_wd{args.wd:.0e}_aug-{aug_tag}_ep{args.max_epochs}'
           f'_pat{args.patience}_n{args.seeds}{tag_extra}')
    with open(os.path.join(OUT_DIR, f'{tag}.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved outputs/{tag}.json')


if __name__ == '__main__':
    main()
