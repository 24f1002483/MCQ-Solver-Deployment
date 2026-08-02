import os
import gc
import numpy as np
import torch
from transformers import AutoTokenizer
from src.config import Config
from src.models import ModelArchitectureFactory

# Tokenizer Cache
_tokenizer_cache = {}

def _get_tokenizer(architecture):
    if architecture not in _tokenizer_cache:
        model_name = Config.model_name_map[architecture]
        _tokenizer_cache[architecture] = AutoTokenizer.from_pretrained(model_name)
    return _tokenizer_cache[architecture]

def _load_checkpoint_weights(model, weights_path):
    state_dict = torch.load(weights_path, map_location=Config.device, weights_only=True)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=True)
    return model

def _load_model(architecture, tokenizer, weights_path):
    model_name = Config.model_name_map[architecture]
    model = ModelArchitectureFactory.build_for_inference(architecture, model_name)
    model = _load_checkpoint_weights(model, weights_path)
    model.to(Config.device)
    model.eval()
    return model

def predict_single(question, choices, weights_dir, architecture):
    labels = ["A", "B", "C", "D", "E"]
    tokenizer = _get_tokenizer(architecture)
    
    input_ids_list = []
    attention_mask_list = []
    for choice in choices:
        encoded = tokenizer(question, choice, truncation=True, max_length=Config.max_len, padding="max_length")
        input_ids_list.append(encoded["input_ids"])
        attention_mask_list.append(encoded["attention_mask"])

    input_ids = torch.tensor([input_ids_list], dtype=torch.long).to(Config.device)
    attention_mask = torch.tensor([attention_mask_list], dtype=torch.long).to(Config.device)

    fold_logits = []
    for fold in range(Config.n_folds):
        weight_path = os.path.join(weights_dir, f"{architecture}_best_fold_{fold}.pt")
        if not os.path.exists(weight_path): continue

        model = _load_model(architecture, tokenizer, weight_path)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            fold_logits.append(outputs.logits.cpu().numpy())
        del model
        gc.collect()

    avg_logits = np.mean(fold_logits, axis=0)
    exp_logits = np.exp(avg_logits - np.max(avg_logits))
    probs = (exp_logits / exp_logits.sum())[0]
    ranked_indices = np.argsort(probs)[::-1]
    return [(labels[i], float(probs[i])) for i in ranked_indices]
