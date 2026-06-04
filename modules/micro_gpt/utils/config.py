"""Configuration containers for the micro_gpt scaffold."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MicroGPTConfig:
    """Hyperparameters needed to instantiate a tiny GPT-style model."""

    vocab_size: int
    context_length: int = 128
    embedding_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    dropout: float = 0.1
