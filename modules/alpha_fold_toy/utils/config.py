"""Configuration containers for alpha_fold_toy."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AlphaFoldToyConfig:
    """Hyperparameters for an AlphaFold-inspired toy model."""

    max_length: int = 128
    residue_vocab_size: int = 21
    sequence_dim: int = 128
    pair_dim: int = 64
    num_distance_bins: int = 32
    num_evoformer_blocks: int = 2
