"""Train a linear head on cached frozen-BERT embeddings.

Equivalent to the baseline (frozen DistilBERT + 768->3 linear + softmax + soft CE)
but operates on precomputed [CLS] vectors — every call is ~1 sec instead of ~5 min.
Used by A1 (label permutation null) and any other linear-head experiment.
"""
import os
import numpy as np
import torch
import torch.nn as nn

CACHE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed',
                          'cls_embeddings_distilbert.npz')


def load_cache(path=CACHE_PATH):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def soft_cross_entropy(pred, target):
    return -torch.sum(target * torch.log(pred + 1e-8), dim=1).mean()


def train_head(embeddings, soft_labels, train_idx, eval_idx, test_idx,
               lr=1e-3, max_epochs=50, patience=5, batch_size=32,
               seed=42, verbose=False):
    """Train a 768->3 linear+softmax head; return dict with metrics + trained weights."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    X = torch.from_numpy(embeddings).float()
    Y = torch.from_numpy(soft_labels).float()

    Xtr, Ytr = X[train_idx], Y[train_idx]
    Xev, Yev = X[eval_idx], Y[eval_idx]
    Xte, Yte = X[test_idx], Y[test_idx]

    n_in = X.shape[1]
    head = nn.Sequential(nn.Linear(n_in, 3), nn.Softmax(dim=1))
    opt = torch.optim.Adam(head.parameters(), lr=lr)

    best_eval = float('inf')
    best_state = None
    bad_epochs = 0
    epochs_run = 0

    for epoch in range(max_epochs):
        epochs_run = epoch + 1
        head.train()
        perm = torch.randperm(len(Xtr))
        train_loss = 0.0
        n_batches = 0
        for start in range(0, len(Xtr), batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad()
            probs = head(Xtr[idx])
            loss = soft_cross_entropy(probs, Ytr[idx])
            loss.backward()
            opt.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= max(1, n_batches)

        head.eval()
        with torch.no_grad():
            eval_loss = soft_cross_entropy(head(Xev), Yev).item()

        if verbose:
            print(f'  epoch {epoch+1}: train {train_loss:.4f}  eval {eval_loss:.4f}')

        if eval_loss < best_eval:
            best_eval = eval_loss
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()

    with torch.no_grad():
        probs_test = head(Xte).numpy()
        probs_train = head(Xtr).numpy()
        probs_eval = head(Xev).numpy()

    pred_test = probs_test.argmax(axis=1)
    true_test = Yte.numpy().argmax(axis=1)
    test_acc = (pred_test == true_test).mean()

    per_class = {}
    for k, name in enumerate(['Cognitive', 'Affective', 'Motivational']):
        m = true_test == k
        per_class[name] = float((pred_test[m] == true_test[m]).mean()) if m.any() else float('nan')
        per_class[name + '_n'] = int(m.sum())

    return {
        'best_eval_loss': best_eval,
        'epochs_run': epochs_run,
        'test_acc': float(test_acc),
        'per_class_acc': per_class,
        'probs_test': probs_test,
        'probs_train': probs_train,
        'probs_eval': probs_eval,
        'head_state': best_state,
    }


if __name__ == '__main__':
    cache = load_cache()
    print(f"Train/Eval/Test sizes: {len(cache['train_idx'])}/{len(cache['eval_idx'])}/{len(cache['test_idx'])}")
    out = train_head(
        cache['embeddings'], cache['soft_labels'],
        cache['train_idx'], cache['eval_idx'], cache['test_idx'],
        lr=1e-3, max_epochs=50, patience=5, batch_size=32, seed=42, verbose=True,
    )
    print(f"Best eval loss: {out['best_eval_loss']:.4f}  Epochs run: {out['epochs_run']}")
    print(f"Test accuracy:  {out['test_acc']:.4f}")
    for name in ['Cognitive', 'Affective', 'Motivational']:
        print(f"  {name}: {out['per_class_acc'][name]:.4f}  (n={out['per_class_acc'][name+'_n']})")
