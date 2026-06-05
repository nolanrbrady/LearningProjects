import pandas as pd

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
    print(f"Loading dataset text for split '{split}' with max {max_chars} chars...")
    if split == "train":
        df = pd.read_csv("./tiny_stories/train.csv", header=None)
    elif split == "val":
        df = pd.read_csv("./tiny_stories/val.csv", header=None)
    else:
        raise ValueError(f"Unsupported split: {split}")
    
    all_text = ""

    current_length = 0
    for _, row in df.iterrows():
        story = row[0]
        all_text += story + " "  # Add a space between stories
        current_length += len(story) + 1  # Account for the added space
        if current_length >= max_chars:
            break
    
    all_text = all_text[:max_chars]  # Ensure we don't exceed max_chars

    print(f"Loaded {len(all_text)} characters of text.")
    
    return all_text


if __name__ == "__main__":
    load_tiny_shakespeare_text(split="train", max_chars=100_000)

