"""Dataset skeletons for micro_text_diffusion."""


def load_ag_news_text(split: str = "train", max_rows: int = 1024) -> list[str]:
    """Load a small text-only slice of AG News.

    Args:
        split: Hugging Face split name, usually `"train"` or `"test"`.
        max_rows: Maximum number of rows to load for a local experiment.

    Returns:
        A list of raw article/title strings.

    This keeps the project grounded in a real dataset while preserving the
    short feedback loop needed for daily learning sessions.
    """
    raise NotImplementedError("Implement row-limited AG News loading as a learning milestone.")


def tokenize_fixed_length(texts: list[str], tokenizer, sequence_length: int):
    """Tokenize and pad/truncate text rows to a fixed length.

    Args:
        texts: Raw AG News strings.
        tokenizer: Tokenizer object chosen by the learner.
        sequence_length: Desired number of tokens per example.

    Returns:
        Tensor-like integer token IDs with shape `(batch, sequence_length)`.

    Fixed shapes make it easier to focus on denoising mechanics before handling
    variable-length batching.
    """
    raise NotImplementedError("Implement fixed-length tokenization as a learning milestone.")
