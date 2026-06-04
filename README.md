# LearningProjects

This repository is a set of from-scratch deep learning and reinforcement learning learning modules. Each module is an end-to-end toy project with enough scaffold to make progress in focused 30-60 minute sessions, but with the core learner-facing implementation intentionally left blank.

The goal is not to collect polished reference implementations. The goal is to build understanding by implementing the important pieces yourself, checking your work against tests, and using the READMEs as a study path.

## Daily Workflow

1. Pick one module and one milestone from its README.
2. Run the milestone tests listed for that step.
3. Read the docstrings for the failing methods.
4. Implement only the current concept.
5. Re-run the targeted tests and write down what changed.

Useful commands:

```bash
pytest tests
pytest modules/micro_gpt/tests -m micro_gpt_tokenizer
pytest modules/alpha_zero/tests -m alpha_zero_mcts
```

## Current Modules

- `modules/micro_gpt`: build a tiny GPT-style language model with tokenizer, causal attention, transformer blocks, training, and sampling. Dataset: `karpathy/tiny_shakespeare`.
- `modules/micro_text_diffusion`: build a text denoising diffusion model with token corruption schedules, denoising transformer, loss, and sampling. Dataset: `fancyzhx/ag_news`.
- `modules/alpha_zero`: build AlphaZero-style policy/value learning, MCTS, self-play, replay, and arena evaluation using PettingZoo environments instead of custom game engines.
- `modules/alpha_fold_toy`: build an AlphaFold-inspired protein-structure toy model with residue/MSA/pair features, Evoformer-lite blocks, distograms, and geometric losses. Dataset: `ChrisHayduk/nanofold-public`.

## Backlog Ideas

- `dqn_cartpole`: DQN with replay buffers and target networks using Gymnasium `CartPole-v1`.
- `ppo_lunar_or_cartpole`: PPO with advantage estimation and clipping using Gymnasium environments.
- `tabular_q_learning`: dynamic programming and Q-learning with Gymnasium Toy Text tasks.
- `offline_rl_intro`: behavior cloning or conservative Q-learning with Hugging Face-hosted RL datasets.
- `cnn_from_scratch`: convolution, pooling, normalization, and residual blocks on MNIST or CIFAR-10.
- `vae_mnist`: variational autoencoders, ELBO, and latent interpolation.
- `ddpm_images`: image diffusion with a small U-Net on MNIST or CIFAR-10.
- `clip_toy`: contrastive image-text learning on a small captioned dataset.
- `gnn_cora`: message passing and node classification on graph data.

## Agent Policy

Read `AGENTS.md` before asking an agent for help. Agents are allowed and expected to create scaffolds, tests, and new modules. They are not allowed to solve learner-facing implementation tasks for you.
