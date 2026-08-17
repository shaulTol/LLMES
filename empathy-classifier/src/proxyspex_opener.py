"""ProxySPEX — Sample-Efficient Feature-Interaction Interpretability.

Implementation of the method presented in LLMES_presentation-shaul-guy.pdf.
Method: (1) sample random token-masks S_i; (2) query model for f(S_i); (3) fit
a Gradient Boosted Tree proxy on (mask, f) pairs; (4) extract top-k Walsh-
Hadamard Fourier coefficients via Monte Carlo on the proxy. Each coefficient
F(T) is the joint importance of token-subset T (interactions, not just single
features).

Phase 29 scope: run on the OPENER (first 10 words of the response) for all 3
reference models. Holds story constant, masks only opener tokens.

Outputs per (model, example):
  - opener words, true class, predicted class, top-k F(T) interactions
  - distribution of |T| (interaction order) across top-k
  - position-distribution of tokens in top-k

Outputs aggregated:
  - per-model top interactions per true class
  - cross-model overlap of top opener tokens
"""
import argparse
import json
import os
import sys
from itertools import combinations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import f1_score, accuracy_score
from transformers import DistilBertModel, DistilBertTokenizer, AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_lora import build_target_modules as build_distilbert_tm, MAX_LEN, DATA_RAW
import run_lora_story
import run_lora_story_roberta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, '..')
OUT_DIR = os.path.join(ROOT, 'outputs')
MODELS_DIR = os.path.join(ROOT, 'models')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else
                       'mps' if torch.backends.mps.is_available() else 'cpu')
CLASSES = ['Cognitive', 'Affective', 'Motivational']


# -------------------------------------------------- Model wrappers
class BaselineModel(nn.Module):
    """Replicates src/model.py.BaselineModel — frozen DistilBERT + linear head."""
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        for p in self.bert.parameters(): p.requires_grad = False
        self.classifier = nn.Linear(768, 3)
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return torch.softmax(self.classifier(cls), dim=1)


class ModelWrapper:
    """Unified interface: predict_batch(stories, responses) -> (B, 3) numpy probs."""
    def predict_batch(self, stories, responses):
        raise NotImplementedError


class BaselineWrapper(ModelWrapper):
    def __init__(self, ckpt_path):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = BaselineModel().to(DEVICE).eval()
        sd = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        if 'state_dict' in sd: sd = sd['state_dict']
        self.model.load_state_dict(sd, strict=True)
        self.name = 'linear_baseline'
        self.uses_story = False
        self.max_len = 512
    @torch.no_grad()
    def predict_batch(self, stories, responses):
        # Baseline is response-only.
        enc = self.tokenizer(responses, truncation=True, padding='max_length',
                              max_length=self.max_len, return_tensors='pt')
        ids = enc['input_ids'].to(DEVICE); am = enc['attention_mask'].to(DEVICE)
        return self.model(ids, am).cpu().numpy()


class LoRAWinnerWrapper(ModelWrapper):
    def __init__(self, ckpt_path):
        ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        a = ck['args']
        from peft import LoraConfig
        lora_cfg = LoraConfig(
            r=a['rank'], lora_alpha=a['rank'],
            target_modules=build_distilbert_tm(a['target'], a['layer_scope']),
            lora_dropout=a['lora_dropout'], bias=a['bias'],
            task_type='FEATURE_EXTRACTION',
        )
        m = run_lora_story.LoRABertStoryResp(
            lora_cfg,
            mlp_hidden=a.get('mlp_hidden', 256),
            mlp_hidden2=a.get('mlp_hidden2', 0),
            dropout=a['head_dropout'],
            latent_sigma=a.get('latent_sigma', 0.0),
            skip_conn=a['skip_conn'],
            mixup_alpha=a.get('mixup_alpha', 0.0),
            pool=a.get('pool', 'cls'),
        ).to(DEVICE).eval()
        m.load_state_dict(ck['state_dict'], strict=True)
        self.model = m
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.name = 'lora_winner'
        self.uses_story = True
        self.max_len = MAX_LEN
    @torch.no_grad()
    def predict_batch(self, stories, responses):
        s_enc = self.tokenizer(stories, truncation=True, padding='max_length',
                                max_length=self.max_len, return_tensors='pt')
        r_enc = self.tokenizer(responses, truncation=True, padding='max_length',
                                max_length=self.max_len, return_tensors='pt')
        logits = self.model(s_enc['input_ids'].to(DEVICE), s_enc['attention_mask'].to(DEVICE),
                             r_enc['input_ids'].to(DEVICE), r_enc['attention_mask'].to(DEVICE))
        return torch.softmax(logits, dim=1).cpu().numpy()


class RobertaWinnerWrapper(ModelWrapper):
    def __init__(self, ckpt_path):
        ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        a = ck['args']
        from peft import LoraConfig
        lora_cfg = LoraConfig(
            r=a['rank'], lora_alpha=a['rank'],
            target_modules=run_lora_story_roberta.build_target_modules(a['target'], a['layer_scope']),
            lora_dropout=a['lora_dropout'], bias=a['bias'],
            task_type='FEATURE_EXTRACTION',
        )
        m = run_lora_story_roberta.LoRARobertaStoryResp(
            lora_cfg, mlp_hidden=256, dropout=a['head_dropout'], skip_conn=a['skip_conn'],
        ).to(DEVICE).eval()
        m.load_state_dict(ck['state_dict'], strict=True)
        self.model = m
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        self.name = 'roberta_winner'
        self.uses_story = True
        self.max_len = 256
    @torch.no_grad()
    def predict_batch(self, stories, responses):
        s_enc = self.tokenizer(stories, truncation=True, padding='max_length',
                                max_length=self.max_len, return_tensors='pt')
        r_enc = self.tokenizer(responses, truncation=True, padding='max_length',
                                max_length=self.max_len, return_tensors='pt')
        logits = self.model(s_enc['input_ids'].to(DEVICE), s_enc['attention_mask'].to(DEVICE),
                             r_enc['input_ids'].to(DEVICE), r_enc['attention_mask'].to(DEVICE))
        return torch.softmax(logits, dim=1).cpu().numpy()


# -------------------------------------------------- ProxySPEX
def proxyspex_example(wrapper, story, response, target_class,
                       n_opener_words=10, n_masks=200, k_order_max=3,
                       n_eval=4096, batch_size=32, seed=0):
    """Run ProxySPEX on the opener of one (story, response).

    Returns top interactions (Fourier coefficients sorted by |F|) and aggregates.
    Holds story constant; masks tokens of the first `n_opener_words` words of response.
    Mask=0 -> replace word with `[MASK]`; mask=1 -> keep original word.
    """
    rng = np.random.default_rng(seed)
    words = response.split()
    opener = words[:n_opener_words]
    body = ' '.join(words[n_opener_words:])
    n = len(opener)
    if n < 2:
        return None  # too few opener tokens

    masks = (rng.random((n_masks, n)) < 0.5).astype(np.int8)
    # Build masked responses
    masked_responses = []
    for m in masks:
        mo = ' '.join(w if m[i] else '[MASK]' for i, w in enumerate(opener))
        masked_responses.append(mo + ((' ' + body) if body else ''))

    # Batch-query the model
    probs = np.zeros((n_masks, 3), dtype=np.float32)
    stor_list = [story] * n_masks
    for s in range(0, n_masks, batch_size):
        e = min(s + batch_size, n_masks)
        probs[s:e] = wrapper.predict_batch(stor_list[s:e], masked_responses[s:e])
    f_values = probs[:, target_class]

    # Fit GBT proxy
    gbt = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                       random_state=seed)
    gbt.fit(masks.astype(np.float32), f_values)
    train_r2 = float(gbt.score(masks.astype(np.float32), f_values))

    # MC Fourier extraction on the proxy: F(T) = E_S[gbt(S) * chi_T(S)]
    # where chi_T(S) = prod_{i in T} (1 - 2*S_i) = (-1)^{|T cap S|}
    eval_masks = (rng.random((n_eval, n)) < 0.5).astype(np.int8)
    eval_fs = gbt.predict(eval_masks.astype(np.float32))
    F0 = float(eval_fs.mean())
    chi_pm = 1 - 2 * eval_masks  # (n_eval, n)
    results = []
    for order in range(1, k_order_max + 1):
        for T in combinations(range(n), order):
            chi = np.prod(chi_pm[:, list(T)], axis=1)
            F_T = float((eval_fs * chi).mean())
            results.append({'T': list(T), 'order': order, 'F': F_T,
                              'tokens': [opener[i] for i in T]})
    results.sort(key=lambda x: -abs(x['F']))

    return {
        'opener_words': opener,
        'n_opener_tokens': n,
        'target_class': int(target_class),
        'target_class_name': CLASSES[target_class],
        'F0': F0,
        'proxy_train_r2': train_r2,
        'top_interactions': results[:30],
    }


# -------------------------------------------------- Example selection
def select_examples(probs, trues, k_per_cell=10):
    """For each true class, pick k confident-correct (highest prob on true class)
    and k confident-wrong (highest prob on wrong class). Returns dict
    {class_name: {'correct': [idxs], 'wrong': [idxs]}}."""
    preds = probs.argmax(axis=1)
    out = {}
    for c in range(3):
        m = trues == c
        idxs = np.where(m)[0]
        correct = sorted([i for i in idxs if preds[i] == c],
                          key=lambda i: -probs[i, c])[:k_per_cell]
        wrong   = sorted([i for i in idxs if preds[i] != c],
                          key=lambda i: -probs[i, preds[i]])[:k_per_cell]
        out[CLASSES[c]] = {'correct': correct, 'wrong': wrong}
    return out


# -------------------------------------------------- Driver
def load_study3_data():
    df = pd.read_csv(DATA_RAW)
    df = df[df['StudyNum'] == '3'].copy().reset_index(drop=True)
    return (df['Response'].fillna('').astype(str).tolist(),
            df['Story'].fillna('').astype(str).tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, required=True,
                   choices=['baseline', 'lora_winner', 'roberta_winner'])
    p.add_argument('--ckpt', type=str, default=None,
                   help='Override default checkpoint path.')
    p.add_argument('--k_per_cell', type=int, default=10,
                   help='Confident-correct + confident-wrong per true class.')
    p.add_argument('--n_opener_words', type=int, default=10)
    p.add_argument('--n_masks', type=int, default=200,
                   help='ProxySPEX inference budget per example (LLM calls).')
    p.add_argument('--k_order_max', type=int, default=3,
                   help='Maximum interaction order |T| extracted.')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--n_eval_proxy', type=int, default=4096,
                   help='MC samples for Fourier estimation on the GBT proxy.')
    p.add_argument('--preds_npz', type=str, default='outputs/preds_seed9_best_seed9.npz',
                   help='File providing test predictions used for example selection. '
                        'Examples are chosen by LoRA-winner confidence so the SAME '
                        'examples are analyzed across all 3 models.')
    p.add_argument('--out_tag', type=str, default='')
    args = p.parse_args()

    print(f'Device: {DEVICE}')
    print(f'Model: {args.model}')

    # Default checkpoint paths
    if args.ckpt is None:
        defaults = {
            'baseline':       os.path.join(MODELS_DIR, 'baseline_v1.pt'),
            'lora_winner':    os.path.join(MODELS_DIR, 'lora_winner_seed9.pt'),
            'roberta_winner': os.path.join(MODELS_DIR, 'roberta_winner_seed0.pt'),
        }
        args.ckpt = defaults[args.model]
    print(f'Checkpoint: {args.ckpt}')

    # Load wrapper
    wrapper_cls = {'baseline': BaselineWrapper,
                    'lora_winner': LoRAWinnerWrapper,
                    'roberta_winner': RobertaWinnerWrapper}[args.model]
    wrapper = wrapper_cls(args.ckpt)
    print(f'Wrapper: {wrapper.name}  (uses_story={wrapper.uses_story})')

    # Load data + LoRA-winner predictions for example selection
    responses, stories = load_study3_data()
    n_test = len(responses)
    npz = np.load(args.preds_npz, allow_pickle=True)
    probs, trues = npz['probs'], npz['trues']
    assert probs.shape[0] == n_test, f'preds {probs.shape} vs test {n_test}'
    selected = select_examples(probs, trues, k_per_cell=args.k_per_cell)
    print('Selected examples per class:')
    for c, d in selected.items():
        print(f'  {c}: {len(d["correct"])} correct, {len(d["wrong"])} wrong')

    # Run ProxySPEX
    all_results = {'model': wrapper.name, 'ckpt': args.ckpt,
                   'config': {k: v for k, v in vars(args).items()
                              if k not in ('ckpt', 'preds_npz')},
                   'examples': {}}
    total = sum(len(d['correct']) + len(d['wrong']) for d in selected.values())
    done = 0
    for c, d in selected.items():
        for cell, idxs in d.items():
            for idx in idxs:
                true_c = int(trues[idx])
                story = stories[idx]
                response = responses[idx]
                # Target class = true class (explain "p(true class)")
                r = proxyspex_example(wrapper, story, response, target_class=true_c,
                                       n_opener_words=args.n_opener_words,
                                       n_masks=args.n_masks,
                                       k_order_max=args.k_order_max,
                                       n_eval=args.n_eval_proxy,
                                       batch_size=args.batch_size,
                                       seed=idx)
                if r is None:
                    done += 1; continue
                r.update({'idx': int(idx), 'true_class': true_c,
                           'true_class_name': CLASSES[true_c],
                           'pred_class': int(probs[idx].argmax()),
                           'pred_probs_lora_winner': probs[idx].tolist(),
                           'cell': cell, 'class_block': c})
                all_results['examples'][int(idx)] = r
                done += 1
                if done % 5 == 0:
                    top = r['top_interactions'][0]
                    print(f'  [{done}/{total}] idx={idx:4d} {c[:3]}/{cell[:5]} '
                          f'r2={r["proxy_train_r2"]:.3f}  '
                          f'top: {top["tokens"]} F={top["F"]:+.3f}')

    # Aggregate stats
    agg = aggregate(all_results['examples'])
    all_results['aggregate'] = agg

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    tag = args.out_tag or wrapper.name
    out_path = os.path.join(OUT_DIR, f'proxyspex_opener_{tag}.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nSaved: {out_path}')

    # Also write a markdown summary
    write_markdown_summary(all_results, os.path.join(OUT_DIR, f'proxyspex_opener_{tag}.md'))


def aggregate(examples):
    """Aggregate per-class top tokens and interaction-order distribution."""
    by_class = {c: {'top_tokens': {}, 'order_hist': {1: 0, 2: 0, 3: 0},
                     'top_pos': {}, 'n_examples': 0}
                for c in CLASSES}
    for ex in examples.values():
        cn = ex['true_class_name']
        by_class[cn]['n_examples'] += 1
        # Top 10 interactions for this example
        for it in ex['top_interactions'][:10]:
            by_class[cn]['order_hist'][it['order']] += 1
            # Single-token attribution: count each token in the interaction
            for tok, pos in zip(it['tokens'], it['T']):
                by_class[cn]['top_tokens'][tok] = (
                    by_class[cn]['top_tokens'].get(tok, 0.0) + abs(it['F']))
                by_class[cn]['top_pos'][pos] = by_class[cn]['top_pos'].get(pos, 0) + 1
    # Sort
    for c, d in by_class.items():
        d['top_tokens'] = sorted(d['top_tokens'].items(), key=lambda x: -x[1])[:15]
        d['top_pos']    = sorted(d['top_pos'].items(),    key=lambda x: -x[1])
    return by_class


def write_markdown_summary(results, out_path):
    lines = [f'# ProxySPEX on opener — {results["model"]}\n',
             f'Checkpoint: `{results["ckpt"]}`\n',
             f'Config: {results["config"]}\n']
    agg = results['aggregate']
    lines.append('\n## Top tokens by aggregated |F| (interaction strength) per true class\n')
    for c in CLASSES:
        d = agg[c]
        lines.append(f'\n### True = {c}  ({d["n_examples"]} examples)\n')
        lines.append(f'Interaction-order distribution (top-10 per example): {d["order_hist"]}')
        toks = ', '.join(f'`{t}` ({w:.2f})' for t, w in d['top_tokens'])
        lines.append(f'Top tokens: {toks}')
        pos = ', '.join(f'pos{p}:{c}' for p, c in d['top_pos'])
        lines.append(f'Top positions: {pos}')
    lines.append('\n## Per-example top-3 interactions (sample)\n')
    for idx, ex in list(results['examples'].items())[:30]:
        lines.append(f'\n### idx={idx} [{ex["class_block"][:3]}/{ex["cell"][:5]}] true={ex["true_class_name"]}')
        lines.append(f'opener: `{" ".join(ex["opener_words"])}`')
        for it in ex['top_interactions'][:3]:
            lines.append(f'  - F={it["F"]:+.3f}  |T|={it["order"]}  '
                          f'tokens={it["tokens"]}  (positions {it["T"]})')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
