"""Loss skeletons for micro_text_diffusion."""


def diffusion_reconstruction_loss(logits, clean_token_ids, corruption_mask=None):
    """Compute token reconstruction loss for denoising diffusion.

    Args:
        logits: Predicted token logits with shape `(batch, sequence, vocab_size)`.
        clean_token_ids: Original token IDs with shape `(batch, sequence)`.
        corruption_mask: Optional boolean mask identifying corrupted positions.

    Returns:
        Scalar loss. If `corruption_mask` is provided, the loss may focus only
        on positions the forward process corrupted.

    This objective trains the reverse model to reconstruct text from noisy
    inputs.
    """
    raise NotImplementedError("Implement diffusion reconstruction loss.")
