"""Configuration containers for the alpha_zero scaffold."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AlphaZeroConfig:
    """Hyperparameters for AlphaZero-style learning."""

    env_name: str = "tictactoe"
    num_simulations: int = 64
    c_puct: float = 1.5
    replay_capacity: int = 10_000
    batch_size: int = 64
    learning_rate: float = 1e-3
