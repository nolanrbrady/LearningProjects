"""Policy/value network skeleton for alpha_zero."""


class PolicyValueNetwork:
    """Network that predicts action priors and state value."""

    def __init__(self, observation_shape, num_actions: int):
        """Initialize the policy/value network.

        Args:
            observation_shape: Shape of encoded environment observations.
            num_actions: Number of discrete actions in the environment.

        Returns:
            None.

        AlphaZero uses this network twice: policy logits guide MCTS expansion,
        and value estimates bootstrap leaf evaluations.
        """
        raise NotImplementedError("Implement policy/value network initialization.")

    def forward(self, observation, action_mask=None):
        """Predict policy logits and scalar value for one or more observations.

        Args:
            observation: Encoded state or batch of states.
            action_mask: Optional legal-action mask where illegal actions should
                not receive probability mass.

        Returns:
            A pair `(policy_logits, value)` where value is from the current
            player's perspective.

        This is the neural contract consumed by MCTS and by the training loss.
        """
        raise NotImplementedError("Implement policy/value network forward pass.")
