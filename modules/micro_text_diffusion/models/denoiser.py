"""Denoising model skeletons for micro_text_diffusion."""


class TimestepEmbedding:
    """Embed diffusion timesteps into model-width vectors."""

    def __init__(self, num_timesteps: int, embedding_dim: int):
        """Initialize a learnable or sinusoidal timestep embedding table.

        Args:
            num_timesteps: Number of discrete diffusion steps.
            embedding_dim: Width of the timestep representation.

        Returns:
            None.

        Timestep conditioning tells the denoiser how much corruption to expect.
        """
        raise NotImplementedError("Implement timestep embedding initialization.")

    def forward(self, timesteps):
        """Map timestep IDs to embeddings.

        Args:
            timesteps: Integer tensor-like object with shape `(batch,)`.

        Returns:
            Tensor-like object with shape `(batch, embedding_dim)`.

        The denoising transformer should combine these vectors with token
        representations.
        """
        raise NotImplementedError("Implement timestep embedding lookup.")


class DenoisingTransformer:
    """Tiny transformer that predicts clean tokens from corrupted tokens."""

    def __init__(self, config):
        """Initialize token embeddings, timestep conditioning, blocks, and output head.

        Args:
            config: `TextDiffusionConfig` describing vocabulary, length, model
                width, layers, heads, and timestep count.

        Returns:
            None.

        The model should output token logits for every corrupted sequence
        position.
        """
        raise NotImplementedError("Implement denoising transformer initialization.")

    def forward(self, corrupted_token_ids, timesteps):
        """Predict clean-token logits from corrupted tokens.

        Args:
            corrupted_token_ids: Integer tensor-like object with shape
                `(batch, sequence)`.
            timesteps: Integer tensor-like object with shape `(batch,)`.

        Returns:
            Logits with shape `(batch, sequence, vocab_size)`.

        This forward pass is the reverse-model component of text diffusion.
        """
        raise NotImplementedError("Implement the denoising transformer forward pass.")
