"""Distogram prediction skeleton for alpha_fold_toy."""


class DistogramHead:
    """Predict distance-bin logits from pair representations."""

    def __init__(self, pair_dim: int, num_bins: int):
        """Initialize the distogram projection.

        Args:
            pair_dim: Width of pair representations.
            num_bins: Number of distance bins to predict.

        Returns:
            None.

        The head translates pair features into residue-residue distance
        predictions.
        """
        raise NotImplementedError("Implement distogram head initialization.")

    def forward(self, pair_repr):
        """Predict distogram logits for all residue pairs.

        Args:
            pair_repr: Tensor-like object with shape `(batch, length, length,
                pair_dim)`.

        Returns:
            Logits with shape `(batch, length, length, num_bins)`.

        These logits are compared against binned coordinate distances during
        training.
        """
        raise NotImplementedError("Implement distogram prediction.")
