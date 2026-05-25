"""
Shared building blocks for the NLA: pooling, FVE metric, text encoder, and the
reconstructor head. Kept in one place so every script uses identical definitions.
"""

import torch
import torch.nn as nn


def masked_mean_pool(hidden, attention_mask):
    """Average token vectors, ignoring padding. hidden: (B, T, D), mask: (B, T)."""
    mask = attention_mask.unsqueeze(-1).type_as(hidden)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


def compute_fve(preds, targets):
    """
    Fraction of Variance Explained.

        FVE = 1 - mean||target - pred||^2 / mean||target - mean(target)||^2

    1.0 = perfect reconstruction; 0.0 = no better than predicting the mean;
    negative = worse than the mean. preds/targets: (N, D) tensors.
    """
    mse = ((preds - targets) ** 2).sum(dim=1).mean()
    mean_t = targets.mean(dim=0, keepdim=True)
    var = ((targets - mean_t) ** 2).sum(dim=1).mean()
    return (1.0 - mse / var).item()


@torch.no_grad()
def encode_texts(base, tokenizer, texts, device, batch_size=32, max_length=256):
    """
    Turn a list of strings into fixed-size vectors using the FROZEN base model:
    run each text through the model and mean-pool its final-layer hidden states.
    Returns a CPU float tensor of shape (len(texts), d_model).
    """
    base.eval()
    feats = []
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i:i + batch_size])
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_length).to(device)
        out = base(**enc, output_hidden_states=True)
        h = masked_mean_pool(out.hidden_states[-1], enc["attention_mask"])
        feats.append(h.float().cpu())
    return torch.cat(feats, dim=0)


class ReconstructorHead(nn.Module):
    """Small MLP mapping a text feature vector -> a predicted activation vector."""

    def __init__(self, d_in, d_out, hidden=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)
