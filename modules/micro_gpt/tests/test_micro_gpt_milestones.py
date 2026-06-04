"""Milestone tests for micro_gpt.

These tests document desired behavior. They are marked xfail until the learner
implements each milestone.
"""

import pytest

from modules.micro_gpt.data.tokenizer import CharacterTokenizer, build_lm_batches
from modules.micro_gpt.data.dataset import load_tiny_shakespeare_text
from modules.micro_gpt.models.attention import CausalSelfAttention
from modules.micro_gpt.models.transformer import MicroGPT, TransformerBlock
from modules.micro_gpt.training.trainer import estimate_loss, train_one_epoch
from modules.micro_gpt.evaluation.generate import generate_text
from modules.micro_gpt.utils.config import MicroGPTConfig


def _shape(value):
    if hasattr(value, "shape"):
        return tuple(value.shape)
    shape = []
    current = value
    while isinstance(current, (list, tuple)):
        shape.append(len(current))
        current = current[0] if current else []
    return tuple(shape)


def _row(value, index=0):
    row = value[index]
    if hasattr(row, "tolist"):
        return row.tolist()
    return list(row)


@pytest.mark.micro_gpt_tokenizer
@pytest.mark.xfail(reason="Learner has not implemented Tiny Shakespeare loading yet.", strict=False)
def test_load_tiny_shakespeare_text_returns_bounded_text_slice():
    text = load_tiny_shakespeare_text(split="train", max_chars=128)

    assert isinstance(text, str)
    assert 0 < len(text) <= 128


@pytest.mark.micro_gpt_tokenizer
@pytest.mark.xfail(reason="Learner has not implemented CharacterTokenizer yet.", strict=False)
def test_character_tokenizer_round_trips_text():
    tokenizer = CharacterTokenizer()
    tokenizer.fit("hello")

    encoded = tokenizer.encode("hello")

    assert tokenizer.decode(encoded) == "hello"
    assert len(encoded) == 5


@pytest.mark.micro_gpt_tokenizer
@pytest.mark.xfail(reason="Learner has not implemented CharacterTokenizer yet.", strict=False)
def test_character_tokenizer_builds_stable_sorted_vocabulary():
    tokenizer = CharacterTokenizer()
    tokenizer.fit("banana")

    assert tokenizer.encode("abn") == [0, 1, 2]
    assert tokenizer.decode([2, 1, 0]) == "nba"


@pytest.mark.micro_gpt_tokenizer
@pytest.mark.xfail(reason="Learner has not implemented CharacterTokenizer validation yet.", strict=False)
def test_character_tokenizer_rejects_unknown_characters_and_ids():
    tokenizer = CharacterTokenizer()
    tokenizer.fit("abc")

    with pytest.raises((KeyError, ValueError)):
        tokenizer.encode("ax")

    with pytest.raises((KeyError, ValueError, IndexError)):
        tokenizer.decode([99])


@pytest.mark.micro_gpt_training
@pytest.mark.xfail(reason="Learner has not implemented language-model batching yet.", strict=False)
def test_lm_batches_shift_targets_by_one_token():
    x, y = build_lm_batches([0, 1, 2, 3, 4, 5], context_length=3, batch_size=1)

    assert list(x[0]) == [0, 1, 2]
    assert list(y[0]) == [1, 2, 3]


@pytest.mark.micro_gpt_training
@pytest.mark.xfail(reason="Learner has not implemented language-model batching yet.", strict=False)
def test_lm_batches_returns_requested_batch_shape_and_contiguous_windows():
    x, y = build_lm_batches(list(range(10)), context_length=3, batch_size=2)

    assert _shape(x) == (2, 3)
    assert _shape(y) == (2, 3)
    assert _row(y, 0) == [token + 1 for token in _row(x, 0)]
    assert _row(y, 1) == [token + 1 for token in _row(x, 1)]


@pytest.mark.micro_gpt_training
@pytest.mark.xfail(reason="Learner has not implemented language-model batching validation yet.", strict=False)
def test_lm_batches_rejects_sequences_that_are_too_short():
    with pytest.raises(ValueError):
        build_lm_batches([0, 1, 2], context_length=3, batch_size=1)


@pytest.mark.micro_gpt_model
@pytest.mark.xfail(reason="Learner has not implemented causal self-attention yet.", strict=False)
def test_causal_attention_preserves_batch_sequence_and_width():
    attention = CausalSelfAttention(embedding_dim=8, num_heads=2)
    output = attention.forward([[[0.0] * 8, [1.0] * 8]])

    assert len(output) == 1
    assert len(output[0]) == 2
    assert len(output[0][0]) == 8


@pytest.mark.micro_gpt_model
@pytest.mark.xfail(reason="Learner has not implemented attention validation yet.", strict=False)
def test_causal_attention_rejects_non_divisible_head_dimensions():
    with pytest.raises(ValueError):
        CausalSelfAttention(embedding_dim=10, num_heads=3)


@pytest.mark.micro_gpt_model
@pytest.mark.xfail(reason="Learner has not implemented transformer block yet.", strict=False)
def test_transformer_block_preserves_input_shape():
    block = TransformerBlock(embedding_dim=8, num_heads=2)
    output = block.forward([[[0.0] * 8, [1.0] * 8, [2.0] * 8]])

    assert _shape(output) == (1, 3, 8)


@pytest.mark.micro_gpt_model
@pytest.mark.xfail(reason="Learner has not implemented MicroGPT yet.", strict=False)
def test_micro_gpt_forward_returns_logits_for_each_token_position():
    model = MicroGPT(MicroGPTConfig(vocab_size=7, context_length=4, embedding_dim=8, num_heads=2, num_layers=1))

    logits = model.forward([[0, 1, 2, 3]])

    assert _shape(logits) == (1, 4, 7)


@pytest.mark.micro_gpt_model
@pytest.mark.xfail(reason="Learner has not implemented MicroGPT loss yet.", strict=False)
def test_micro_gpt_forward_returns_scalar_loss_when_targets_are_provided():
    model = MicroGPT(MicroGPTConfig(vocab_size=7, context_length=4, embedding_dim=8, num_heads=2, num_layers=1))

    output = model.forward([[0, 1, 2, 3]], targets=[[1, 2, 3, 4]])

    assert isinstance(output, tuple)
    logits, loss = output
    assert _shape(logits) == (1, 4, 7)
    assert _shape(loss) == ()


@pytest.mark.micro_gpt_training
@pytest.mark.xfail(reason="Learner has not implemented training helpers yet.", strict=False)
def test_training_helpers_average_losses_and_update_only_during_training():
    class DummyOptimizer:
        def __init__(self):
            self.zero_grad_calls = 0
            self.step_calls = 0

        def zero_grad(self):
            self.zero_grad_calls += 1

        def step(self):
            self.step_calls += 1

    class DummyLoss(float):
        def backward(self):
            pass

    class DummyModel:
        def __init__(self):
            self.calls = 0

        def __call__(self, x, targets=None):
            self.calls += 1
            return None, DummyLoss(self.calls)

    model = DummyModel()
    optimizer = DummyOptimizer()
    batches = [([0], [1]), ([1], [2])]

    assert train_one_epoch(model, batches, optimizer) == pytest.approx(1.5)
    assert optimizer.zero_grad_calls == 2
    assert optimizer.step_calls == 2
    assert estimate_loss(model, batches) == pytest.approx(3.5)
    assert optimizer.step_calls == 2


@pytest.mark.micro_gpt_model
@pytest.mark.xfail(reason="Learner has not implemented autoregressive generation yet.", strict=False)
def test_generate_text_preserves_prompt_and_respects_requested_length():
    class EchoTokenizer:
        def encode(self, text):
            return [ord(char) for char in text]

        def decode(self, token_ids):
            return "".join(chr(token_id) for token_id in token_ids)

    class ConstantNextTokenModel:
        def __call__(self, token_ids):
            return [[[0.0] * 123]]

    generated = generate_text(ConstantNextTokenModel(), EchoTokenizer(), "hi", max_new_tokens=3)

    assert generated.startswith("hi")
    assert len(generated) == 5
