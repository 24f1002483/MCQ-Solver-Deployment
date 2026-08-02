import torch

class Config:
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_len = 384
    n_folds = 5
    model_name_map = {
        "deberta": "microsoft/deberta-v3-base",
        "roberta": "roberta-base",
        "scratch": "bert-base-uncased"
    }

def seed_everything(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
