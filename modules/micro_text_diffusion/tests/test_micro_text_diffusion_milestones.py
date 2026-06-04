"""Milestone tests for micro_text_diffusion."""

import pytest

from modules.micro_text_diffusion.data.dataset import load_ag_news_text, tokenize_fixed_length
from modules.micro_text_diffusion.models.denoiser import TimestepEmbedding
from modules.micro_text_diffusion.models.denoiser import DenoisingTransformer
from modules.micro_text_diffusion.models.noising import TokenNoisingSchedule
from modules.micro_text_diffusion.evaluation.sample import sample_text
from modules.micro_text_diffusion.training.losses import diffusion_reconstruction_loss
from modules.micro_text_diffusion.training.trainer import estimate_reconstruction_loss, train_one_epoch
from modules.micro_text_diffusion.utils.config import TextDiffusionConfig


def _shape(value):
    if hasattr(value, "shape"):
        return tuple(value.shape)
    shape = []
    current = value
    while isinstance(current, (list, tuple)):
        shape.append(len(current))
        current = current[0] if current else []
    return tuple(shape)


def _to_list(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


@pytest.mark.micro_text_diffusion_schedule
@pytest.mark.xfail(reason="Learner has not implemented TokenNoisingSchedule yet.", strict=False)
def test_noising_probability_increases_with_timestep():
    schedule = TokenNoisingSchedule(num_timesteps=10, min_noise=0.1, max_noise=0.9)

    assert schedule.probability(0) == pytest.approx(0.1)
    assert schedule.probability(9) == pytest.approx(0.9)
    assert schedule.probability(5) > schedule.probability(4)


@pytest.mark.micro_text_diffusion_schedule
@pytest.mark.xfail(reason="Learner has not implemented schedule validation yet.", strict=False)
def test_noising_schedule_rejects_invalid_bounds_and_timesteps():
    with pytest.raises(ValueError):
        TokenNoisingSchedule(num_timesteps=0)

    with pytest.raises(ValueError):
        TokenNoisingSchedule(num_timesteps=10, min_noise=0.8, max_noise=0.2)

    schedule = TokenNoisingSchedule(num_timesteps=3)
    with pytest.raises((IndexError, ValueError)):
        schedule.probability(3)


@pytest.mark.micro_text_diffusion_schedule
@pytest.mark.xfail(reason="Learner has not implemented token corruption yet.", strict=False)
def test_corrupt_preserves_shape_returns_mask_and_does_not_mutate_input():
    original = [[1, 2, 3], [4, 5, 6]]
    schedule = TokenNoisingSchedule(num_timesteps=1, min_noise=1.0, max_noise=1.0)

    corrupted, mask = schedule.corrupt(original, timestep=0, mask_token_id=99)

    assert _shape(corrupted) == (2, 3)
    assert _shape(mask) == (2, 3)
    assert _to_list(corrupted) == [[99, 99, 99], [99, 99, 99]]
    assert _to_list(mask) == [[True, True, True], [True, True, True]]
    assert original == [[1, 2, 3], [4, 5, 6]]


@pytest.mark.micro_text_diffusion_schedule
@pytest.mark.xfail(reason="Learner has not implemented zero-probability corruption yet.", strict=False)
def test_corrupt_with_zero_noise_leaves_tokens_unchanged():
    schedule = TokenNoisingSchedule(num_timesteps=1, min_noise=0.0, max_noise=0.0)

    corrupted, mask = schedule.corrupt([[1, 2, 3]], timestep=0, mask_token_id=99)

    assert _to_list(corrupted) == [[1, 2, 3]]
    assert _to_list(mask) == [[False, False, False]]


@pytest.mark.micro_text_diffusion_model
@pytest.mark.xfail(reason="Learner has not implemented TimestepEmbedding yet.", strict=False)
def test_timestep_embedding_returns_one_vector_per_example():
    embedding = TimestepEmbedding(num_timesteps=100, embedding_dim=16)
    output = embedding.forward([0, 5, 99])

    assert len(output) == 3
    assert len(output[0]) == 16


@pytest.mark.micro_text_diffusion_model
@pytest.mark.xfail(reason="Learner has not implemented timestep validation yet.", strict=False)
def test_timestep_embedding_rejects_out_of_range_timesteps():
    embedding = TimestepEmbedding(num_timesteps=3, embedding_dim=4)

    with pytest.raises((IndexError, ValueError)):
        embedding.forward([3])


@pytest.mark.micro_text_diffusion_model
@pytest.mark.xfail(reason="Learner has not implemented denoising transformer yet.", strict=False)
def test_denoising_transformer_returns_vocab_logits_per_position():
    config = TextDiffusionConfig(vocab_size=11, sequence_length=5, num_timesteps=4, embedding_dim=8, num_heads=2, num_layers=1)
    model = DenoisingTransformer(config)

    logits = model.forward([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], timesteps=[0, 3])

    assert _shape(logits) == (2, 5, 11)


@pytest.mark.micro_text_diffusion_model
@pytest.mark.xfail(reason="Learner has not implemented reconstruction loss yet.", strict=False)
def test_diffusion_reconstruction_loss_is_scalar_and_supports_masking():
    logits = [
        [[10.0, 0.0], [0.0, 10.0]],
        [[10.0, 0.0], [0.0, 10.0]],
    ]
    clean = [[0, 1], [1, 0]]
    masked_loss = diffusion_reconstruction_loss(logits, clean, corruption_mask=[[True, True], [False, False]])
    full_loss = diffusion_reconstruction_loss(logits, clean)

    assert _shape(masked_loss) == ()
    assert _shape(full_loss) == ()
    assert float(masked_loss) < float(full_loss)


@pytest.mark.micro_text_diffusion_schedule
@pytest.mark.xfail(reason="Learner has not implemented fixed-length tokenization yet.", strict=False)
def test_tokenize_fixed_length_pads_and_truncates_rows():
    class SimpleTokenizer:
        pad_token_id = 0

        def encode(self, text, **kwargs):
            return [ord(char) for char in text]

    tokenized = tokenize_fixed_length(["abcdef", "xy"], SimpleTokenizer(), sequence_length=4)

    assert _shape(tokenized) == (2, 4)
    assert _to_list(tokenized)[0] == [97, 98, 99, 100]
    assert _to_list(tokenized)[1] == [120, 121, 0, 0]


@pytest.mark.micro_text_diffusion_schedule
@pytest.mark.xfail(reason="Learner has not implemented AG News loading yet.", strict=False)
def test_load_ag_news_text_respects_max_rows_and_returns_strings():
    rows = load_ag_news_text(split="train", max_rows=3)

    assert len(rows) <= 3
    assert rows
    assert all(isinstance(row, str) and row for row in rows)


@pytest.mark.micro_text_diffusion_model
@pytest.mark.xfail(reason="Learner has not implemented diffusion training helpers yet.", strict=False)
def test_diffusion_training_helpers_average_losses_and_step_optimizer():
    class DummyOptimizer:
        def __init__(self):
            self.zero_grad_calls = 0
            self.step_calls = 0

        def zero_grad(self):
            self.zero_grad_calls += 1

        def step(self):
            self.step_calls += 1

    class DummySchedule:
        def corrupt(self, clean_tokens, timestep, mask_token_id):
            return clean_tokens, [[False for _ in row] for row in clean_tokens]

    class DummyModel:
        def __init__(self):
            self.calls = 0

        def __call__(self, corrupted_tokens, timesteps):
            self.calls += 1
            return [[[0.0, 1.0]]]

    config = TextDiffusionConfig(vocab_size=2, sequence_length=1, num_timesteps=2, mask_token_id=0)
    batches = [[[1]], [[0]]]
    optimizer = DummyOptimizer()

    train_loss = train_one_epoch(DummyModel(), batches, DummySchedule(), optimizer, config)
    eval_loss = estimate_reconstruction_loss(DummyModel(), batches, DummySchedule(), config)

    assert _shape(train_loss) == ()
    assert _shape(eval_loss) == ()
    assert optimizer.zero_grad_calls == 2
    assert optimizer.step_calls == 2


@pytest.mark.micro_text_diffusion_model
@pytest.mark.xfail(reason="Learner has not implemented iterative text sampling yet.", strict=False)
def test_sample_text_returns_decoded_sequence_of_requested_length():
    class SimpleTokenizer:
        mask_token_id = 0

        def decode(self, token_ids):
            return "".join(str(token_id) for token_id in token_ids)

    class DummyModel:
        def __call__(self, corrupted_token_ids, timesteps):
            return [[[0.0, 1.0]] for _ in corrupted_token_ids]

    generated = sample_text(DummyModel(), SimpleTokenizer(), sequence_length=4, num_timesteps=3)

    assert isinstance(generated, str)
    assert len(generated) == 4
