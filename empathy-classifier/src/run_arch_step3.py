"""Step 3 of the architecture search: sharpen soft labels (Y^alpha / sum)."""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from arch_search import HeadConfig, compare

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

cfg_current = HeadConfig(head_type='mlp', hidden_dim=256, dropout=0.3,
                          loss='soft_ce', label_sharpen_alpha=1.0)
cfg_proposed = HeadConfig(head_type='mlp', hidden_dim=256, dropout=0.3,
                           loss='soft_ce', label_sharpen_alpha=3.0)

compare('mlp_256_softCE', cfg_current, 'mlp_256_softCE_sharpen3', cfg_proposed,
        n_seeds=30,
        save_path=os.path.join(OUT_DIR, 'arch_step3_sharpen.json'))
