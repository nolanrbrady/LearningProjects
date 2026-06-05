import random

"""Tokenizer and language-model batch skeletons for micro_gpt."""


class CharacterTokenizer:
    """Character-level tokenizer for tiny Shakespeare experiments."""
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}

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
        self.char_to_id = {}
        self.id_to_char = {}
        for char in text:
            if char not in self.char_to_id:
                self.char_to_id[char] = len(self.char_to_id)
                self.id_to_char[len(self.id_to_char)] = char
        
        print("Character to ID mapping:")
        print(self.char_to_id)
        print("Id to char mapping:")
        print(self.id_to_char)

    def encode(self, text: str) -> list[int]:
        """Convert text into token IDs using the fitted vocabulary.

        Args:
            text: Raw text containing only characters known to the tokenizer.

        Returns:
            A list of integer token IDs with the same length as `text`.

        Encoding creates the discrete input sequence used to build next-token
        training examples.
        """
        output_ids = []
        for char in text:
            if char in self.char_to_id:
                output_ids.append(self.char_to_id[char])
            else:
                raise ValueError(f"Character '{char}' not in tokenizer vocabulary.")
        print(f"Encoded '{text}' to token IDs: {output_ids}")
        return output_ids

    def decode(self, token_ids: list[int]) -> str:
        """Convert token IDs back into text.

        Args:
            token_ids: Integer IDs previously produced by `encode`.

        Returns:
            A string reconstructed from the ID-to-character table.

        Round-trip correctness is a basic check that the model's inputs and
        generated outputs share the same vocabulary contract.
        """
        output_text = ""
        for token_id in token_ids:
            if token_id in self.id_to_char:
                output_text = output_text + self.id_to_char[token_id]
            else:
                ValueError(f"Token ID {token_id} is available in the tokenzer.")
        
        print("Decoded string from token IDs: ")
        print(output_text)
        return output_text


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
    batch_x = []
    batch_y = []
    for _ in range(batch_size):
        start_char = random.randint(0, len(token_ids) - (context_length + 1))
        batch_x.append(token_ids[start_char : start_char + context_length + 1])
        batch_y.append(token_ids[start_char + 1 : start_char + context_length + 2]) 

    return batch_x, batch_y
        


if __name__ == "__main__":
    # Example usage of the tokenizer and batch builder.
    tokenizer_text = "To be, or not to be, that is the question. Going forward, we will test the tokenizer and batch builder with this sample text."
    tokenizer = CharacterTokenizer()
    tokenizer.fit(tokenizer_text)
    text_to_encode = "This is a test string to be encoded."
    token_ids = tokenizer.encode(text_to_encode)
    print(f"Encoded token IDs: {token_ids}")
    decoded_text = tokenizer.decode(token_ids)
    print(f"Decoded text: {decoded_text}")

    context_length = 5
    batch_size = 2
    x, y = build_lm_batches(token_ids, context_length, batch_size)
    print(f"Input batch (x): {x}")
    print(f"Target batch (y): {y}")