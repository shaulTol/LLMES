"""Step 4 of the architecture search: class-balanced sampling on MLP + soft CE."""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from arch_search import HeadConfig, compare

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

cfg_current = HeadConfig(head_type='mlp', hidden_dim=256, dropout=0.3,
                          loss='soft_ce', balanced_sampling=False)
cfg_proposed = HeadConfig(head_type='mlp', hidden_dim=256, dropout=0.3,
                           loss='soft_ce', balanced_sampling=True)

compare('mlp_256_softCE', cfg_current, 'mlp_256_softCE_balsamp', cfg_proposed,
        n_seeds=30,
        save_path=os.path.join(OUT_DIR, 'arch_step4_balsamp.json'))
