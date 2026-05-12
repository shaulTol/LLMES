"""Build caches that concatenate first-N-words [CLS] with the full-response [CLS].

Unlike the earlier `first_N_rest` caches (which split the text into chunks),
these keep the entire-response embedding intact AND add the opener as an extra
feature view. So the head can attend to both the global summary and the
opener-specific signal.

Output shapes: 1536-d (first_N + full) and 2304-d (first_N + rest + full).
Re-uses existing precomputed CLS vectors — no DistilBERT re-run needed.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from head_trainer import load_cache

SCRIPT_DIR = os.path.dirname(__file__)
DATA_PROC = os.path.join(SCRIPT_DIR, '..', 'data', 'processed')


def main():
    full = load_cache()  # cls_embeddings_distilbert.npz
    X_full = full['embeddings']
    Y = full['soft_labels']
    tr, ev, te = full['train_idx'], full['eval_idx'], full['test_idx']
    print(f'Full CLS shape: {X_full.shape}')

    for N in [5, 10]:
        chunked_path = os.path.join(DATA_PROC, f'cls_embeddings_first{N}_rest.npz')
        d = np.load(chunked_path, allow_pickle=True)
        chunked = d['embeddings']  # shape (N_rows, 1536) = [first_N; rest]
        first_N = chunked[:, :768]
        rest = chunked[:, 768:]
        print(f'first{N} shape: {first_N.shape}, rest shape: {rest.shape}')

        # 1536-d: first_N + full
        feats_2 = np.concatenate([first_N, X_full], axis=1).astype(np.float32)
        path_2 = os.path.join(DATA_PROC, f'cls_embeddings_first{N}_plus_full.npz')
        np.savez(path_2, embeddings=feats_2, soft_labels=Y,
                 train_idx=tr, eval_idx=ev, test_idx=te)
        print(f'  saved {path_2}  shape {feats_2.shape}')

        # 2304-d: first_N + rest + full (gives model all three views)
        feats_3 = np.concatenate([first_N, rest, X_full], axis=1).astype(np.float32)
        path_3 = os.path.join(DATA_PROC, f'cls_embeddings_first{N}_rest_full.npz')
        np.savez(path_3, embeddings=feats_3, soft_labels=Y,
                 train_idx=tr, eval_idx=ev, test_idx=te)
        print(f'  saved {path_3}  shape {feats_3.shape}')

    print('Done.')


if __name__ == '__main__':
    main()
