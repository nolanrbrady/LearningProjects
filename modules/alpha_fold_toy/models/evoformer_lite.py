"""Evoformer-lite skeletons for alpha_fold_toy."""


class EvoformerLiteBlock:
    """Toy block that updates sequence/MSA and pair representations."""

    def __init__(self, sequence_dim: int, pair_dim: int):
        """Initialize sequence and pair update layers.

        Args:
            sequence_dim: Width of per-residue representations.
            pair_dim: Width of pairwise residue representations.

        Returns:
            None.

        This block should capture the high-level idea of information exchange
        between residue-level and pair-level features without reproducing full
        AlphaFold.
        """
        raise NotImplementedError("Implement Evoformer-lite initialization.")

    def forward(self, sequence_repr, pair_repr):
        """Update sequence and pair representations.

        Args:
            sequence_repr: Tensor-like object with shape `(batch, length,
                sequence_dim)`.
            pair_repr: Tensor-like object with shape `(batch, length, length,
                pair_dim)`.

        Returns:
            A pair `(updated_sequence_repr, updated_pair_repr)` with the same
            shapes as the inputs.

        The update should let pair features influence residues and residue
        features influence pairs.
        """
        raise NotImplementedError("Implement Evoformer-lite forward pass.")
