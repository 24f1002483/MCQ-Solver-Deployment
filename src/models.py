import torch
import torch.nn as nn
from transformers import AutoModelForMultipleChoice


class ScratchMCQModel(nn.Module):
    """Custom lightweight MCQ model: Embedding → BiLSTM → Attention → Classifier.

    Architecture reverse-engineered from checkpoint keys/shapes:
      embedding(30522, 256) → BiLSTM(2 layers, hidden=256) → Attention(512) → Linear(512→1)
    """

    def __init__(self, vocab_size=30522, embed_dim=256, hidden_dim=256, num_layers=2, num_choices=5):
        super().__init__()
        self.num_choices = num_choices
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.GRU(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
        )
        # Bidirectional → 2 * hidden_dim = 512
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim * 2, num_heads=1, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim * 2))
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.classifier = nn.Linear(hidden_dim * 2, 1)

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.size(0)
        # (batch, choices, seq) → (batch*choices, seq)
        flat_ids = input_ids.view(-1, input_ids.size(-1))
        flat_mask = attention_mask.view(-1, attention_mask.size(-1))

        x = self.embedding(flat_ids)
        x, _ = self.encoder(x)

        # Attention pooling
        q = self.query.expand(x.size(0), -1, -1)
        key_padding_mask = flat_mask == 0
        attn_out, _ = self.attention(q, x, x, key_padding_mask=key_padding_mask)
        attn_out = self.layer_norm(attn_out.squeeze(1))

        logits = self.classifier(attn_out)
        logits = logits.view(batch_size, self.num_choices)
        return _InferenceOutput(logits=logits)


class _InferenceOutput:
    """Simple container so that `outputs.logits` works uniformly."""

    def __init__(self, logits):
        self.logits = logits


class _TransformerMCQWrapper(nn.Module):
    """Thin wrapper around HuggingFace MultipleChoice models to return
    our _InferenceOutput instead of the transformers output object."""

    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return _InferenceOutput(logits=outputs.logits)


class ModelArchitectureFactory:
    """Factory that builds the correct model variant for a given architecture name."""

    @staticmethod
    def build_for_inference(architecture, model_name):
        if architecture == "scratch":
            return ScratchMCQModel()
        # For deberta / roberta — use the HF MultipleChoice head directly.
        # This gives us matching state_dict keys (e.g. "roberta.encoder...", "classifier.weight").
        hf_model = AutoModelForMultipleChoice.from_pretrained(model_name)
        return _TransformerMCQWrapper(hf_model)
