# micro_gpt

## Overview

Build a tiny GPT-style language model from scratch. The project moves from text tokenization to causal self-attention, transformer blocks, autoregressive training, and sampling.

## Learning Goals

- Understand how text becomes token IDs and training windows.
- Implement causal masking and scaled dot-product attention.
- Build transformer blocks from attention, MLPs, normalization, and residual paths.
- Train next-token prediction with cross-entropy.
- Generate text autoregressively and inspect model behavior.

## Dataset Or Environment

Use the Hugging Face dataset `karpathy/tiny_shakespeare`. Keep downloads optional and row-limited during experiments. The dataset is small enough for character-level modeling and quick training loops.

## Information Flow

1. [X] `load_tiny_shakespeare_text` returns a bounded raw text slice.
2. [X] `CharacterTokenizer.fit` builds the character vocabulary from training text.
3. [X] `CharacterTokenizer.encode` converts text into a flat token stream.
4. `build_lm_batches` turns the stream into next-token `(x, y)` windows.
5. `MicroGPT.forward` embeds tokens and positions, applies transformer blocks, and produces next-token logits.
6. `train_one_epoch` and `estimate_loss` use the model loss to track learning.
7. `generate_text` repeatedly feeds generated tokens back into the model and decodes the final sequence.

Every public function or method in this flow has a milestone test. The tests are intended to identify the exact missing contract: dataset loading, vocabulary behavior, batching, attention masking, model shapes, loss handling, training updates, or sampling.

## Milestones

1. Dataset loading: fetch a bounded Tiny Shakespeare text slice.
2. Character tokenizer: vocabulary, encode, decode, validation, and round-trip behavior.
3. Language-model batches: convert token streams into `(x, y)` next-token windows.
4. Causal self-attention: validate head dimensions and compute masked attention over a batch of sequences.
5. Transformer block: combine attention, MLP, residual connections, and normalization.
6. GPT model: produce logits with shape `(batch, sequence, vocab_size)` and scalar loss when targets exist.
7. Training loop: estimate loss, update parameters during training, and overfit a tiny batch.
8. Sampling: generate new text from a prompt with temperature-aware autoregressive decoding.

## Suggested 30-60 Minute Sessions

- Session 1: implement `load_tiny_shakespeare_text` and inspect a short corpus slice.
- Session 2: implement `CharacterTokenizer.fit`, `encode`, and `decode`.
- Session 3: implement fixed-length next-token batches.
- Session 4: implement causal masking and attention shape logic.
- Session 5: implement the transformer block.
- Session 6: connect embeddings, blocks, LM head, and optional target loss.
- Session 7: train on a tiny subset and inspect loss.
- Session 8: implement sampling and compare temperatures.

## Tests

```bash
pytest modules/micro_gpt/tests -m micro_gpt_tokenizer
pytest modules/micro_gpt/tests -m micro_gpt_model
pytest modules/micro_gpt/tests -m micro_gpt_training
```

## Resources

- Andrej Karpathy, "The Unreasonable Effectiveness of Recurrent Neural Networks" for the tiny Shakespeare context.
- "Attention Is All You Need" for the transformer building blocks.
- The nanoGPT repository for architecture naming conventions only; do not copy implementation code into this module.

## What Agents May Help With

Agents may explain concepts, help interpret tests, improve scaffolding, and add new tests. Agents must not implement the learner-facing solution methods in this module.
