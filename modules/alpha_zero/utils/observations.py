"""Observation-handling skeletons for alpha_zero."""


def encode_observation(raw_observation):
    """Convert a PettingZoo observation object into a network-ready state.

    Args:
        raw_observation: Observation returned by `env.observe(agent)`. PettingZoo
            classic environments commonly return a mapping with an
            `"observation"` array and an `"action_mask"` array.

    Returns:
        Encoded observation/state representation consumed by
        `PolicyValueNetwork.forward`.

    This method isolates environment-specific observation shapes from the
    learner-facing search and network code.
    """
    raise NotImplementedError("Implement AlphaZero observation encoding as a learning milestone.")


def legal_actions_from_observation(raw_observation) -> list[int]:
    """Extract legal discrete action IDs from an observation action mask.

    Args:
        raw_observation: Observation mapping containing an `"action_mask"` where
            truthy entries indicate legal actions.

    Returns:
        Sorted list of legal action IDs.

    AlphaZero must never expand, sample, or train policy mass on illegal
    actions, so this contract is shared by MCTS, self-play, and evaluation.
    """
    raise NotImplementedError("Implement legal-action extraction as a learning milestone.")
