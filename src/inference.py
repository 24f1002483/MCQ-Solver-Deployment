import os
import numpy as np
import torch
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from src.config import Config
from src.models import ModelArchitectureFactory

# In-memory caches for fast UI response
_tokenizer_cache = {}
_model_cache = {}


def _get_tokenizer(architecture):
    if architecture not in _tokenizer_cache:
        model_name = Config.model_name_map[architecture]
        _tokenizer_cache[architecture] = AutoTokenizer.from_pretrained(model_name)
    return _tokenizer_cache[architecture]


def _load_checkpoint_weights(target, weights_path):
    """Load checkpoint into `target` (raw model or HF inner model)."""
    state_dict = torch.load(weights_path, map_location=Config.device, weights_only=True)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    target.load_state_dict(cleaned, strict=True)
    return target


def _load_model(architecture, weights_path):
    cache_key = (architecture, weights_path)
    if cache_key not in _model_cache:
        model_name = Config.model_name_map[architecture]
        model = ModelArchitectureFactory.build_for_inference(architecture, model_name)

        if architecture == "scratch":
            _load_checkpoint_weights(model, weights_path)
        else:
            _load_checkpoint_weights(model.model, weights_path)

        model.to(Config.device)
        model.eval()
        _model_cache[cache_key] = model
    return _model_cache[cache_key]


def predict_single(question, choices, weights_dir, architecture, temperature=0.15):
    """Predict using weights for the given architecture with temperature scaling."""
    labels = ["A", "B", "C", "D", "E"]
    tokenizer = _get_tokenizer(architecture)
    
    # Prepare Inputs with explicit prompt formatting
    input_ids_list, attention_mask_list = [], []
    for choice in choices:
        encoded = tokenizer(
            f"Question: {question}", 
            f"Option: {choice}", 
            truncation=True, 
            max_length=Config.max_len, 
            padding="max_length"
        )
        input_ids_list.append(encoded["input_ids"])
        attention_mask_list.append(encoded["attention_mask"])

    input_ids = torch.tensor([input_ids_list], dtype=torch.long).to(Config.device)
    attention_mask = torch.tensor([attention_mask_list], dtype=torch.long).to(Config.device)

    # Determine Weight File
    weight_path = os.path.join(weights_dir, f"{architecture}_avg.pt")
    if not os.path.exists(weight_path):
        fold_0_path = os.path.join(weights_dir, f"{architecture}_best_fold_0.pt")
        if os.path.exists(fold_0_path):
            weight_path = fold_0_path
        else:
            os.makedirs(weights_dir, exist_ok=True)
            weight_path = hf_hub_download(
                repo_id="NidhiHe/mcq-solver",
                filename=f"weights/{architecture}_best_fold_0.pt",
                local_dir="."
            )

    model = _load_model(architecture, weight_path)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits / temperature
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # Rank results
    ranked_indices = np.argsort(probs)[::-1]
    return [(labels[i], float(probs[i])) for i in ranked_indices]
