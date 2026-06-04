# alpha_fold_toy

## Overview

Build an AlphaFold-inspired toy protein-structure module. This is not a full AlphaFold reproduction. The goal is to understand selected building blocks: residue encodings, MSA/pair features, Evoformer-like updates, distogram prediction, and simple geometry losses.

## Learning Goals

- Understand how amino-acid sequences become model features.
- Build pairwise residue representations.
- Use MSA-style context without recreating the full AlphaFold data pipeline.
- Implement an Evoformer-lite block at toy scale.
- Predict residue-residue distance bins.
- Compute simple geometric losses and inspect protein-specific metrics.

## Dataset Or Environment

Use the Hugging Face dataset `ChrisHayduk/nanofold-public`. It is a compact NanoFold/OpenProteinSet-derived benchmark with train and validation chains. Default experiments should use tiny row-limited slices and avoid downloading the full dataset unless explicitly requested.

## Information Flow

1. `load_nanofold_slice` returns a tiny protein-structure dataset slice.
2. `extract_sequence_and_coordinates` normalizes one example into a sequence and C-alpha-style coordinates.
3. `encode_residues` converts amino-acid letters into residue IDs.
4. `build_msa_profile` summarizes aligned sequence context at each residue position.
5. `build_pair_features` constructs residue-pair inputs with shape `(length, length, channels)`.
6. `coordinates_to_distogram_bins` converts coordinates into supervised pairwise distance-bin labels.
7. `EvoformerLiteBlock.forward` updates sequence and pair representations while preserving shapes.
8. `AlphaFoldToyModel.forward` connects residue IDs, optional MSA profile, pair features, Evoformer-lite blocks, and the distogram head.
9. `distogram_cross_entropy` trains distance-bin predictions, and `mean_pairwise_distance_error` reports interpretable geometry error.

The tests cover all public contracts in this path, including residue validation, MSA alignment, pair-feature shape and symmetry, distogram clipping, model output shape, masked losses, dataset extraction, and masked distance metrics.

## Milestones

1. Dataset slice: inspect one row and identify sequence, MSA, and coordinate fields.
2. Residue encoding: map amino-acid strings to integer IDs.
3. MSA profile: summarize aligned sequences into per-position residue frequencies.
4. Pair features: create pairwise residue features with shape `(length, length, channels)`.
5. Distance targets: convert coordinates to symmetric, clipped distogram bin labels.
6. Evoformer-lite block: update sequence/MSA and pair representations.
7. Distogram head and toy model: predict distance-bin logits for every residue pair.
8. Geometry loss and metrics: train on a tiny slice and inspect distance errors.

## Suggested 30-60 Minute Sessions

- Session 1: read one NanoFold row and sketch sequence, MSA, coordinate, and pair-feature shapes.
- Session 2: implement residue vocabulary encoding.
- Session 3: implement MSA profile construction and validation.
- Session 4: implement pair-feature construction.
- Session 5: implement pairwise distance and bin targets.
- Session 6: implement one Evoformer-lite update.
- Session 7: implement the distogram head, toy model wrapper, and loss.
- Session 8: overfit a tiny protein slice and visualize predicted distances.

## Tests

```bash
pytest modules/alpha_fold_toy/tests -m alpha_fold_features
pytest modules/alpha_fold_toy/tests -m alpha_fold_model
```

## Resources

- AlphaFold and OpenFold papers for architecture motivation.
- NanoFold/OpenProteinSet references for dataset context.
- Protein structure primers on amino acids, C-alpha coordinates, and distance maps.

## What Agents May Help With

Agents may explain concepts, help interpret tests, improve scaffolding, and add new tests. Agents must not implement the learner-facing solution methods in this module.
