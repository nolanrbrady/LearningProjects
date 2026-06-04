"""Dataset skeletons for alpha_fold_toy."""


def load_nanofold_slice(split: str = "train", max_rows: int = 32):
    """Load a tiny row-limited slice of NanoFold public data.

    Args:
        split: Dataset split, typically `"train"` or validation if available.
        max_rows: Maximum rows to load for a local toy experiment.

    Returns:
        Dataset-like collection of protein-chain examples.

    This function should keep large protein-structure data optional and
    cache-aware while exposing real sequence/structure examples.
    """
    raise NotImplementedError("Implement row-limited NanoFold loading.")


def extract_sequence_and_coordinates(example):
    """Extract sequence and coordinate fields from one dataset example.

    Args:
        example: One row from `ChrisHayduk/nanofold-public`.

    Returns:
        A pair `(sequence, coordinates)` where `sequence` is an amino-acid
        string and `coordinates` contains residue coordinates used for distance
        targets.

    This normalizes the dataset row into the minimal objects needed by the toy
    feature pipeline.
    """
    raise NotImplementedError("Implement NanoFold example extraction.")
