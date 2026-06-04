"""Feature-construction skeletons for alpha_fold_toy."""


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def encode_residues(sequence: str) -> list[int]:
    """Encode an amino-acid sequence into residue IDs.

    Args:
        sequence: Protein sequence using one-letter amino-acid codes.

    Returns:
        List of integer residue IDs with length equal to `len(sequence)`.

    Residue IDs are the sequence-level input to the toy folding model.
    """
    raise NotImplementedError("Implement amino-acid residue encoding.")


def build_pair_features(residue_ids: list[int]):
    """Construct pairwise residue features.

    Args:
        residue_ids: Encoded residue IDs with shape conceptually `(length,)`.

    Returns:
        Pair feature tensor-like object with shape `(length, length, channels)`.

    Pair representations let the model reason about relationships between all
    residue positions, a central AlphaFold idea.
    """
    raise NotImplementedError("Implement pair-feature construction.")


def build_msa_profile(msa_sequences: list[str], query_sequence: str):
    """Build a toy MSA profile aligned to a query sequence.

    Args:
        msa_sequences: Aligned amino-acid sequences that include or correspond
            to the query sequence.
        query_sequence: Primary protein sequence with length `L`.

    Returns:
        Profile-like tensor or array with shape `(L, residue_channels)` that
        summarizes residue frequencies at each aligned position.

    This keeps the module connected to the MSA idea behind AlphaFold while
    staying small enough for a toy guided project.
    """
    raise NotImplementedError("Implement toy MSA profile construction.")


def coordinates_to_distogram_bins(coordinates, num_bins: int, max_distance: float):
    """Convert residue coordinates into pairwise distance-bin labels.

    Args:
        coordinates: Coordinate tensor-like object with shape `(length, 3)` or
            an equivalent C-alpha coordinate representation.
        num_bins: Number of discrete distance bins.
        max_distance: Maximum distance represented before clipping.

    Returns:
        Integer bin labels with shape `(length, length)`.

    Distogram targets turn continuous 3D geometry into a supervised prediction
    problem suitable for a toy model.
    """
    raise NotImplementedError("Implement distogram target construction.")
