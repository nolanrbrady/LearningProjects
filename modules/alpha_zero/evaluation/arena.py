"""Arena evaluation skeleton for alpha_zero."""


def evaluate_agents(env_factory, candidate_agent, baseline_agent, num_games: int):
    """Compare two agents by playing repeated games.

    Args:
        env_factory: Callable that creates a fresh PettingZoo environment.
        candidate_agent: Agent being evaluated.
        baseline_agent: Reference agent, previous checkpoint, or random policy.
        num_games: Number of games to play.

    Returns:
        Summary of wins, losses, draws, and average returns.

    Arena evaluation checks whether a new policy is actually stronger than the
    prior reference.
    """
    raise NotImplementedError("Implement arena evaluation.")
