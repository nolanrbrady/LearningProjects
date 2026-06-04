"""Configuration containers for the micro_text_diffusion scaffold."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TextDiffusionConfig:
    """Hyperparameters for a toy text diffusion model."""

    vocab_size: int
    sequence_length: int = 64
    num_timesteps: int = 100
    embedding_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    mask_token_id: int = 0
