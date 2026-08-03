import torch


class Config:
    """Central configuration for the MCQ Solver."""

    # Model name mapping for each supported architecture
    model_name_map = {
        "deberta": "microsoft/deberta-v3-base",
        "roberta": "roberta-base",
        "scratch": "bert-base-uncased",  # scratch model uses bert-base-uncased tokenizer
    }

    # Maximum token length for input sequences
    max_len = 256

    # Number of cross-validation folds (matching the 5 weight files per arch)
    n_folds = 5

    # Automatically select GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")