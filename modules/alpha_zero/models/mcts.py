"""Monte Carlo Tree Search skeletons for alpha_zero."""


class MCTSNode:
    """One node in the AlphaZero search tree."""

    def __init__(self, prior: float, to_play):
        """Initialize node statistics.

        Args:
            prior: Prior probability assigned by the policy network.
            to_play: Agent/player whose turn it is at this state.

        Returns:
            None.

        A node should track visits, total value, children, and enough metadata
        for selection and backup.
        """
        raise NotImplementedError("Implement MCTS node initialization.")

    def value(self) -> float:
        """Return the mean value estimate for this node.

        Args:
            None.

        Returns:
            Mean backed-up value, usually `value_sum / visit_count`, with a
            sensible convention for unvisited nodes.

        This statistic balances exploitation against exploration in selection.
        """
        raise NotImplementedError("Implement node value calculation.")


class MCTS:
    """AlphaZero-style search over an external environment."""

    def __init__(self, network, num_simulations: int, c_puct: float):
        """Initialize search with a policy/value network and search settings.

        Args:
            network: Policy/value network used at leaf nodes.
            num_simulations: Number of simulations per root search.
            c_puct: Exploration constant for PUCT selection.

        Returns:
            None.

        Search turns raw network predictions into stronger policy targets for
        self-play.
        """
        raise NotImplementedError("Implement MCTS initialization.")

    def run(self, env):
        """Run MCTS from the current environment state.

        Args:
            env: PettingZoo environment positioned at the current turn.

        Returns:
            Visit-count policy over legal actions.

        This method should clone or otherwise reason about future environment
        states without corrupting the real self-play game.
        """
        raise NotImplementedError("Implement MCTS search.")
