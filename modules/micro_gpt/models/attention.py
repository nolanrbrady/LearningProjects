"""Causal self-attention skeleton for micro_gpt."""


class CausalSelfAttention:
    """Multi-head masked self-attention for autoregressive language modeling."""

    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.0):
        """Store attention dimensions and initialize projection layers.

        Args:
            embedding_dim: Width of each token representation.
            num_heads: Number of parallel attention heads.
            dropout: Dropout probability applied to attention or projections.

        Returns:
            None.

        The implementation should eventually validate that `embedding_dim` is
        divisible by `num_heads` and create query, key, value, and output
        projections.
        """
        raise NotImplementedError("Implement attention initialization as a learning milestone.")

    def forward(self, x):
        """Apply causal self-attention to a batch of token embeddings.

        Args:
            x: Tensor-like object with shape `(batch, sequence, embedding_dim)`.

        Returns:
            Tensor-like object with shape `(batch, sequence, embedding_dim)`.

        The attention mask must prevent each position from reading future
        tokens, preserving the autoregressive training objective.
        """
        raise NotImplementedError("Implement causal self-attention as a learning milestone.")
