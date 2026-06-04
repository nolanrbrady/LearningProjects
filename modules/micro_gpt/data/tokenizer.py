"""Tokenizer and language-model batch skeletons for micro_gpt."""


class CharacterTokenizer:
    """Character-level tokenizer for tiny Shakespeare experiments."""

    def fit(self, text: str) -> None:
        """Build vocabulary tables from raw training text.

        Args:
            text: Raw corpus text. Each distinct character should receive one
                stable integer ID.

        Returns:
            None. The method should store the character-to-ID and ID-to-character
            mappings on the tokenizer instance.

        This tokenizer is the first bridge from human-readable text into the
        integer sequence consumed by the GPT model.
        """
        raise NotImplementedError("Implement tokenizer fitting as a learning milestone.")

    def encode(self, text: str) -> list[int]:
        """Convert text into token IDs using the fitted vocabulary.

        Args:
            text: Raw text containing only characters known to the tokenizer.

        Returns:
            A list of integer token IDs with the same length as `text`.

        Encoding creates the discrete input sequence used to build next-token
        training examples.
        """
        raise NotImplementedError("Implement text encoding as a learning milestone.")

    def decode(self, token_ids: list[int]) -> str:
        """Convert token IDs back into text.

        Args:
            token_ids: Integer IDs previously produced by `encode`.

        Returns:
            A string reconstructed from the ID-to-character table.

        Round-trip correctness is a basic check that the model's inputs and
        generated outputs share the same vocabulary contract.
        """
        raise NotImplementedError("Implement text decoding as a learning milestone.")


def build_lm_batches(token_ids: list[int], context_length: int, batch_size: int):
    """Create next-token language-model batches.

    Args:
        token_ids: Flat token stream from the tokenizer.
        context_length: Number of input tokens per training example.
        batch_size: Number of examples per batch.

    Returns:
        A pair `(x, y)` where both tensors or arrays have shape
        `(batch_size, context_length)`. `y` should contain the next token for
        every position in `x`.

    This is the data contract consumed by the GPT training loop.
    """
    raise NotImplementedError("Implement language-model batching as a learning milestone.")
