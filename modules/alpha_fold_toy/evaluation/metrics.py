"""Evaluation skeletons for alpha_fold_toy."""


def mean_pairwise_distance_error(predicted_distances, target_distances, mask=None):
    """Measure average error between predicted and target pairwise distances.

    Args:
        predicted_distances: Tensor-like object with shape `(batch, length,
            length)`.
        target_distances: Tensor-like object with the same shape.
        mask: Optional valid-pair mask.

    Returns:
        Scalar mean absolute distance error over valid pairs.

    This metric gives an interpretable readout of geometry quality in Angstroms
    or the chosen coordinate unit.
    """
    raise NotImplementedError("Implement pairwise distance error metric.")
