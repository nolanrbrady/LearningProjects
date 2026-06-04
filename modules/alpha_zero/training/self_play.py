"""Self-play skeleton for alpha_zero."""


def run_self_play_game(env, network, search):
    """Generate one game of AlphaZero self-play.

    Args:
        env: PettingZoo environment such as `tictactoe`.
        network: Policy/value network used by MCTS.
        search: MCTS instance that returns action visit distributions.

    Returns:
        A sequence of training examples containing state, policy target, and
        final value target.

    Self-play is where AlphaZero turns search into supervised training data.
    """
    raise NotImplementedError("Implement one self-play game.")


def train_step(network, replay_batch, optimizer):
    """Update the policy/value network from replayed self-play examples.

    Args:
        network: Policy/value network to update.
        replay_batch: Batch from `ReplayBuffer.sample`.
        optimizer: Optimizer for network parameters.

    Returns:
        Dictionary of scalar losses, including policy and value terms.

    The training step teaches the network to predict MCTS-improved policy and
    final outcome value.
    """
    raise NotImplementedError("Implement AlphaZero training step.")
