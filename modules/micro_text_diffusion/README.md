# micro_text_diffusion

## Overview

Build a toy text diffusion model. Instead of predicting the next token from left to right, the model learns to denoise corrupted token sequences across discrete timesteps.

## Learning Goals

- Understand discrete token corruption and denoising objectives.
- Build timestep embeddings and condition a model on noise level.
- Implement a small denoising transformer.
- Train a reconstruction objective over corrupted text.
- Sample by iteratively denoising from noisy tokens.

## Dataset Or Environment

Use the Hugging Face dataset `fancyzhx/ag_news`. The rows are short enough for fast tokenization and fixed-length text-denoising experiments.

## Information Flow

1. `load_ag_news_text` returns a row-limited text slice.
2. `tokenize_fixed_length` converts rows into padded or truncated clean token batches.
3. `TokenNoisingSchedule.probability` maps timesteps to corruption probabilities.
4. `TokenNoisingSchedule.corrupt` creates corrupted inputs and corruption masks.
5. `TimestepEmbedding` and `DenoisingTransformer.forward` condition predictions on the selected timestep.
6. `diffusion_reconstruction_loss` compares predicted logits with clean token IDs, optionally focusing on corrupted positions.
7. `train_one_epoch` and `estimate_reconstruction_loss` connect corruption, model prediction, loss, and optimizer behavior.
8. `sample_text` starts from noisy tokens and iteratively applies the reverse process before decoding.

The module tests cover every public method in this path, including schedule bounds, input mutation guarantees, fixed-length tokenization, model shape contracts, masked losses, training updates, and iterative sampling.

## Milestones

1. Dataset slice: load a small row-limited subset of AG News text.
2. Fixed-length tokenization: pad or truncate rows into clean token batches.
3. Token corruption schedule: define valid monotonic noise by timestep.
4. Corruption function: replace or mask tokens according to the schedule without mutating clean inputs.
5. Timestep embeddings: represent the current denoising step and validate timestep ranges.
6. Denoising transformer: predict original-token logits from corrupted tokens.
7. Diffusion loss: compute scalar reconstruction loss with optional corruption masking.
8. Training loop: sample timesteps, corrupt inputs, update parameters, and estimate validation loss.
9. Sampling loop: iteratively denoise a noisy sequence and decode text.

## Suggested 30-60 Minute Sessions

- Session 1: inspect AG News rows and choose a fixed sequence length.
- Session 2: implement fixed-length tokenization.
- Session 3: implement corruption probability by timestep.
- Session 4: implement token corruption and shape checks.
- Session 5: implement timestep embeddings.
- Session 6: connect embeddings and transformer blocks.
- Session 7: implement reconstruction loss and training helpers.
- Session 8: train on a tiny subset and watch reconstruction loss.
- Session 9: implement iterative denoising and inspect samples.

## Tests

```bash
pytest modules/micro_text_diffusion/tests -m micro_text_diffusion_schedule
pytest modules/micro_text_diffusion/tests -m micro_text_diffusion_model
```

## Resources

- Diffusion model tutorials for the forward/reverse process intuition.
- Text diffusion papers for discrete corruption objectives.
- Hugging Face Datasets documentation for row-limited loading.

## What Agents May Help With

Agents may explain concepts, help interpret tests, improve scaffolding, and add new tests. Agents must not implement the learner-facing solution methods in this module.
