"""External environment adapters for alpha_zero.

These helpers are scaffold infrastructure, not the learning target. The learner
should implement AlphaZero search and learning against these maintained
PettingZoo environments rather than writing game rules from scratch.
"""

SUPPORTED_CLASSIC_ENVS = ("tictactoe", "connect_four")


def make_pettingzoo_env(env_name: str = "tictactoe", render_mode: str | None = None):
    """Create a supported PettingZoo classic environment.

    Args:
        env_name: Supported environment name: `"tictactoe"` or `"connect_four"`.
        render_mode: Optional PettingZoo render mode.

    Returns:
        A PettingZoo AEC environment instance.

    This adapter keeps the module focused on AlphaZero mechanics by delegating
    game rules, legal moves, and terminal-state handling to PettingZoo.
    """
    if env_name == "tictactoe":
        from pettingzoo.classic import tictactoe_v3

        return tictactoe_v3.env(render_mode=render_mode)
    if env_name == "connect_four":
        from pettingzoo.classic import connect_four_v3

        return connect_four_v3.env(render_mode=render_mode)
    raise ValueError(f"Unsupported PettingZoo classic environment: {env_name!r}")


def reset_env(env, seed: int | None = 0):
    """Reset an external PettingZoo environment.

    Args:
        env: PettingZoo AEC environment created by `make_pettingzoo_env`.
        seed: Optional deterministic seed for the environment reset.

    Returns:
        The same environment after reset, for convenient chaining.

    Keeping reset behavior in one small adapter makes tests independent from
    learner-facing AlphaZero implementations.
    """
    env.reset(seed=seed)
    return env
