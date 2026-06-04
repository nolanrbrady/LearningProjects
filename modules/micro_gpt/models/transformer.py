"""Transformer block and GPT model skeletons for micro_gpt."""


class TransformerBlock:
    """One decoder-only transformer block."""

    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.0):
        """Initialize attention, feed-forward, normalization, and dropout layers.

        Args:
            embedding_dim: Width of token representations.
            num_heads: Number of causal attention heads.
            dropout: Dropout probability for residual paths.

        Returns:
            None.

        The block should preserve sequence length and embedding width while
        letting each token incorporate information from previous tokens.
        """
        raise NotImplementedError("Implement transformer block initialization.")

    def forward(self, x):
        """Apply one transformer block to token representations.

        Args:
            x: Tensor-like object with shape `(batch, sequence, embedding_dim)`.

        Returns:
            Tensor-like object with the same shape as `x`.

        This method is where residual connections, normalization, attention,
        and the feed-forward network are composed.
        """
        raise NotImplementedError("Implement the transformer block forward pass.")


class MicroGPT:
    """Tiny decoder-only language model."""

    def __init__(self, config):
        """Initialize token embeddings, position embeddings, blocks, and LM head.

        Args:
            config: `MicroGPTConfig` describing vocabulary size, context length,
                embedding width, number of heads, and number of layers.

        Returns:
            None.

        The model should map integer token IDs to next-token logits for each
        sequence position.
        """
        raise NotImplementedError("Implement MicroGPT initialization.")

    def forward(self, token_ids, targets=None):
        """Compute next-token logits and optionally a training loss.

        Args:
            token_ids: Integer tensor-like object with shape `(batch, sequence)`.
            targets: Optional integer tensor-like object with the same shape,
                containing the next-token target at each position.

        Returns:
            If `targets` is absent, return logits with shape
            `(batch, sequence, vocab_size)`. If `targets` is present, also
            return a scalar cross-entropy loss.

        This is the central model contract used by training and sampling.
        """
        raise NotImplementedError("Implement the MicroGPT forward pass.")
