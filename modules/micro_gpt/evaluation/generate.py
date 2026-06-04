"""Autoregressive sampling skeleton for micro_gpt."""


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float = 1.0) -> str:
    """Generate text from a prompt using autoregressive sampling.

    Args:
        model: MicroGPT-like model that maps token IDs to next-token logits.
        tokenizer: Fitted tokenizer with `encode` and `decode` methods.
        prompt: Initial text used to seed generation.
        max_new_tokens: Number of additional tokens to sample.
        temperature: Logit scaling factor controlling sample sharpness.

    Returns:
        The prompt plus generated text.

    Sampling lets you inspect what the trained model has learned beyond scalar
    loss values.
    """
    raise NotImplementedError("Implement autoregressive generation as a learning milestone.")
