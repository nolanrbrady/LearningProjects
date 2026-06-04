"""Dataset-loading skeletons for micro_gpt."""


def load_tiny_shakespeare_text(split: str = "train", max_chars: int = 100_000) -> str:
    """Load a bounded text slice for character-level language modeling.

    Args:
        split: Dataset split or subset name to load.
        max_chars: Maximum number of characters to return for a local run.

    Returns:
        Raw text string with length at most `max_chars`.

    This method is the first data-flow step before fitting the character
    tokenizer. It should keep downloads optional, cache-aware, and small enough
    for short learning sessions.
    """
    raise NotImplementedError("Implement row-limited Tiny Shakespeare loading as a learning milestone.")
