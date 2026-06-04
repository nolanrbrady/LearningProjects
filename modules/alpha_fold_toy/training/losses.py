"""Loss skeletons for alpha_fold_toy."""


def distogram_cross_entropy(logits, target_bins, mask=None):
    """Compute cross-entropy over pairwise distance bins.

    Args:
        logits: Distance-bin logits with shape `(batch, length, length,
            num_bins)`.
        target_bins: Integer bin labels with shape `(batch, length, length)`.
        mask: Optional valid-pair mask with shape `(batch, length, length)`.

    Returns:
        Scalar loss over valid residue pairs.

    This is the first toy geometry objective before attempting coordinate-level
    prediction.
    """
    raise NotImplementedError("Implement distogram cross-entropy.")
