"""Sampling skeleton for micro_text_diffusion."""


def sample_text(model, tokenizer, sequence_length: int, num_timesteps: int):
    """Generate text by iteratively denoising an initially noisy sequence.

    Args:
        model: Denoising model returning token logits.
        tokenizer: Tokenizer used to decode sampled token IDs.
        sequence_length: Number of tokens to generate.
        num_timesteps: Number of reverse denoising steps to run.

    Returns:
        Generated text string.

    Sampling exposes whether the learned reverse process can move from noise
    toward coherent text.
    """
    raise NotImplementedError("Implement iterative text denoising.")
