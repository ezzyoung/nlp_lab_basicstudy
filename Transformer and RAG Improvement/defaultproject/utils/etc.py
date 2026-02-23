import json
import random
import numpy as np
import torch
from pyserini.index.lucene import Document


def print_model_statistics(model):
    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    n_params_trainable = sum({p.data_ptr(): p.numel() for p in model.parameters() if p.requires_grad}.values())
    print(f"Training Total size={n_params / 2**20:.2f}M params. Trainable ratio={n_params_trainable / n_params * 100:.2f}%")


def set_seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def hit2docdict(hit) -> dict:
    return json.loads(Document(hit.lucene_document).raw())
