"""Milestone tests for alpha_fold_toy."""

import pytest

from modules.alpha_fold_toy.data.dataset import extract_sequence_and_coordinates, load_nanofold_slice
from modules.alpha_fold_toy.evaluation.metrics import mean_pairwise_distance_error
from modules.alpha_fold_toy.models.distogram import DistogramHead
from modules.alpha_fold_toy.models.evoformer_lite import EvoformerLiteBlock
from modules.alpha_fold_toy.models.model import AlphaFoldToyModel
from modules.alpha_fold_toy.training.losses import distogram_cross_entropy
from modules.alpha_fold_toy.utils.config import AlphaFoldToyConfig
from modules.alpha_fold_toy.models.features import (
    AMINO_ACIDS,
    build_msa_profile,
    build_pair_features,
    coordinates_to_distogram_bins,
    encode_residues,
)


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


@pytest.mark.alpha_fold_features
@pytest.mark.xfail(reason="Learner has not implemented residue encoding yet.", strict=False)
def test_residue_encoding_preserves_sequence_length():
    encoded = encode_residues("ACD")

    assert len(encoded) == 3
    assert all(isinstance(token_id, int) for token_id in encoded)


@pytest.mark.alpha_fold_features
@pytest.mark.xfail(reason="Learner has not implemented residue encoding yet.", strict=False)
def test_residue_encoding_maps_standard_amino_acids_to_stable_ids():
    encoded = encode_residues(AMINO_ACIDS)

    assert encoded == list(range(len(AMINO_ACIDS)))


@pytest.mark.alpha_fold_features
@pytest.mark.xfail(reason="Learner has not implemented residue validation yet.", strict=False)
def test_residue_encoding_rejects_unknown_residues():
    with pytest.raises((KeyError, ValueError)):
        encode_residues("ACX")


@pytest.mark.alpha_fold_features
@pytest.mark.xfail(reason="Learner has not implemented pair features yet.", strict=False)
def test_pair_features_are_square_over_residue_length():
    pair_features = build_pair_features([0, 1, 2, 3])

    assert len(pair_features) == 4
    assert len(pair_features[0]) == 4


@pytest.mark.alpha_fold_features
@pytest.mark.xfail(reason="Learner has not implemented pair-feature construction yet.", strict=False)
def test_pair_features_include_channel_dimension_and_are_symmetric_for_residue_pairs():
    pair_features = build_pair_features([0, 1, 2])

    assert _shape(pair_features)[:2] == (3, 3)
    assert len(_shape(pair_features)) == 3
    as_list = _to_list(pair_features)
    assert as_list[0][1] == as_list[1][0]
    assert as_list[0][2] == as_list[2][0]


@pytest.mark.alpha_fold_features
@pytest.mark.xfail(reason="Learner has not implemented MSA profile construction yet.", strict=False)
def test_msa_profile_is_aligned_to_query_length_and_residue_channels():
    profile = build_msa_profile(msa_sequences=["ACD", "AAD", "ACD"], query_sequence="ACD")

    assert _shape(profile) == (3, len(AMINO_ACIDS))
    as_list = _to_list(profile)
    assert as_list[0][0] == pytest.approx(1.0)
    assert as_list[1][1] > as_list[1][2]


@pytest.mark.alpha_fold_features
@pytest.mark.xfail(reason="Learner has not implemented MSA validation yet.", strict=False)
def test_msa_profile_rejects_sequences_that_do_not_align_to_query():
    with pytest.raises(ValueError):
        build_msa_profile(msa_sequences=["AC", "ACD"], query_sequence="ACD")


@pytest.mark.alpha_fold_model
@pytest.mark.xfail(reason="Learner has not implemented distogram targets yet.", strict=False)
def test_distogram_targets_are_square_distance_bins():
    bins = coordinates_to_distogram_bins(
        coordinates=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        num_bins=4,
        max_distance=4.0,
    )

    assert len(bins) == 2
    assert len(bins[0]) == 2


@pytest.mark.alpha_fold_model
@pytest.mark.xfail(reason="Learner has not implemented distogram target construction yet.", strict=False)
def test_distogram_bins_are_symmetric_diagonal_zero_and_clip_far_distances():
    bins = coordinates_to_distogram_bins(
        coordinates=[(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (10.0, 0.0, 0.0)],
        num_bins=4,
        max_distance=4.0,
    )
    bins = _to_list(bins)

    assert bins[0][0] == 0
    assert bins[1][1] == 0
    assert bins[2][2] == 0
    assert bins[0][1] == bins[1][0]
    assert bins[0][2] == bins[2][0] == 3
    assert all(0 <= bin_id < 4 for row in bins for bin_id in row)


@pytest.mark.alpha_fold_model
@pytest.mark.xfail(reason="Learner has not implemented distogram head yet.", strict=False)
def test_distogram_head_returns_bin_logits_for_each_residue_pair():
    head = DistogramHead(pair_dim=3, num_bins=5)
    pair_repr = [[[[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]], [[1.0, 1.0, 1.0], [3.0, 2.0, 1.0]]]]

    logits = head.forward(pair_repr)

    assert _shape(logits) == (1, 2, 2, 5)


@pytest.mark.alpha_fold_model
@pytest.mark.xfail(reason="Learner has not implemented Evoformer-lite yet.", strict=False)
def test_evoformer_lite_preserves_sequence_and_pair_shapes():
    block = EvoformerLiteBlock(sequence_dim=4, pair_dim=3)
    sequence_repr = [[[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]]]
    pair_repr = [[[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]]]

    updated_sequence, updated_pair = block.forward(sequence_repr, pair_repr)

    assert _shape(updated_sequence) == (1, 2, 4)
    assert _shape(updated_pair) == (1, 2, 2, 3)


@pytest.mark.alpha_fold_model
@pytest.mark.xfail(reason="Learner has not implemented AlphaFoldToyModel yet.", strict=False)
def test_alpha_fold_toy_model_connects_features_to_distogram_logits():
    config = AlphaFoldToyConfig(
        max_length=3,
        residue_vocab_size=len(AMINO_ACIDS),
        sequence_dim=8,
        pair_dim=4,
        num_distance_bins=6,
        num_evoformer_blocks=1,
    )
    model = AlphaFoldToyModel(config)

    logits = model.forward(
        residue_ids=[[0, 1, 2]],
        pair_features=[[[[0.0] * 4 for _ in range(3)] for _ in range(3)]],
        msa_profile=[[[0.0] * len(AMINO_ACIDS) for _ in range(3)]],
    )

    assert _shape(logits) == (1, 3, 3, 6)


@pytest.mark.alpha_fold_features
@pytest.mark.xfail(reason="Learner has not implemented NanoFold loading yet.", strict=False)
def test_load_nanofold_slice_respects_max_rows():
    rows = load_nanofold_slice(split="train", max_rows=2)

    assert len(rows) <= 2


@pytest.mark.alpha_fold_features
@pytest.mark.xfail(reason="Learner has not implemented NanoFold example extraction yet.", strict=False)
def test_extract_sequence_and_coordinates_normalizes_common_example_fields():
    example = {
        "sequence": "ACD",
        "coordinates": [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)],
    }

    sequence, coordinates = extract_sequence_and_coordinates(example)

    assert sequence == "ACD"
    assert _shape(coordinates) == (3, 3)


@pytest.mark.alpha_fold_model
@pytest.mark.xfail(reason="Learner has not implemented distance error metric yet.", strict=False)
def test_mean_pairwise_distance_error_averages_absolute_errors_and_respects_mask():
    predicted = [[[0.0, 2.0], [2.0, 0.0]]]
    target = [[[0.0, 1.0], [1.0, 10.0]]]

    full_error = mean_pairwise_distance_error(predicted, target)
    masked_error = mean_pairwise_distance_error(predicted, target, mask=[[[False, True], [True, False]]])

    assert float(full_error) == pytest.approx(3.0)
    assert float(masked_error) == pytest.approx(1.0)


@pytest.mark.alpha_fold_model
@pytest.mark.xfail(reason="Learner has not implemented distogram cross-entropy yet.", strict=False)
def test_distogram_cross_entropy_is_scalar_and_respects_pair_mask():
    logits = [[[[10.0, 0.0], [0.0, 10.0]], [[10.0, 0.0], [0.0, 10.0]]]]
    target_bins = [[[0, 1], [1, 0]]]

    masked_loss = distogram_cross_entropy(logits, target_bins, mask=[[[True, True], [False, False]]])
    full_loss = distogram_cross_entropy(logits, target_bins)

    assert _shape(masked_loss) == ()
    assert _shape(full_loss) == ()
    assert float(masked_loss) < float(full_loss)
