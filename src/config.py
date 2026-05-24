"""
Central configuration for the Natural Language Autoencoder (NLA) reimplementation.

Keeping every important choice in one place makes the experiment reproducible and
makes it easy to explain *why* we chose what we chose in the README.
"""

from dataclasses import dataclass


@dataclass
class Config:
    # ----- Target model (the model whose activations we interpret) -----
    # Qwen2.5-0.5B-Instruct: recent (late 2024), capable for its size, fits a free T4.
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # Which residual-stream layer to read. The paper reads ~2/3 of the way through.
    # Qwen2.5-0.5B has 24 layers -> 2/3 * 24 = 16. We read hidden_states[layer].
    # (hidden_states[0] is the embedding output; hidden_states[i] is the output of layer i.)
    layer: int = 16

    # ----- Activation harvesting -----
    dataset_name: str = "wikitext"          # easy, no auth; swap to code data for the AI4Code angle
    dataset_config: str = "wikitext-2-raw-v1"
    dataset_split: str = "train"

    num_samples: int = 8000                 # how many activations to collect
    min_tokens: int = 16                    # random truncation lower bound
    max_tokens: int = 128                   # random truncation upper bound
    seed: int = 0

    # ----- Where to save -----
    # In Colab, point this at your Google Drive so a disconnect never loses work,
    # e.g. "/content/drive/MyDrive/nla/data".
    data_dir: str = "data"


CONFIG = Config()
