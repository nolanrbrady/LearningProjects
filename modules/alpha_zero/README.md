# alpha_zero

## Overview

Build the core pieces of AlphaZero: policy/value prediction, Monte Carlo Tree Search, self-play, replay, training, and arena evaluation. The environment is intentionally not part of the learning task.

## Learning Goals

- Understand action masks and turn-based multi-agent environment APIs.
- Implement a policy/value network contract.
- Implement MCTS selection, expansion, evaluation, and backup.
- Generate self-play examples from search-improved policies.
- Train from replayed `(state, policy_target, value_target)` examples.
- Compare agents through arena evaluation.

## Dataset Or Environment

Use PettingZoo classic environments. Start with `tictactoe`, then extend to `connect_four`. PettingZoo provides the game rules, legal action masks, resets, and stepping API, so learning time stays focused on AlphaZero.

## Information Flow

1. `make_pettingzoo_env` and `reset_env` create a fresh external game state.
2. `encode_observation` and `legal_actions_from_observation` separate network inputs from legal-action masks.
3. `PolicyValueNetwork.forward` predicts action logits/priors and a scalar value from the current player's perspective.
4. `MCTSNode` stores prior, visits, value sum, and children for search bookkeeping.
5. `MCTS.run` uses legal actions, network priors, leaf values, and backups to return a normalized visit-count policy without corrupting the real environment.
6. `run_self_play_game` records `(state, policy_target, value_target)` examples from search-guided play.
7. `ReplayBuffer` stores and samples self-play examples for training.
8. `train_step` optimizes policy imitation and value prediction losses.
9. `evaluate_agents` plays repeated games to compare a candidate agent against a baseline.

The tests intentionally cover each public contract in this loop: environment support, action-mask extraction, network shape/masking behavior, MCTS statistics and policies, replay capacity, self-play row structure, optimizer stepping, and arena summaries.

## Milestones

1. Environment adapter: instantiate and reset PettingZoo `tictactoe`.
2. Observation handling: read observations and legal-action masks.
3. Policy/value network: map observations to masked action logits and scalar value.
4. MCTS node: store prior, visit count, value sum, children, and unvisited-node conventions.
5. MCTS search: selection, expansion, evaluation, backup, and normalized legal-action policies.
6. Self-play: collect search policies and terminal outcomes from the current player's perspective.
7. Replay buffer: enforce capacity and sample training columns.
8. Training loop: optimize policy and value losses and return scalar diagnostics.
9. Arena: compare current and previous agents with wins, losses, draws, and average returns.

## Suggested 30-60 Minute Sessions

- Session 1: run the adapter test and inspect PettingZoo observations.
- Session 2: implement action-mask extraction and policy masking.
- Session 3: implement the policy/value network shape contract.
- Session 4: implement MCTS node statistics.
- Session 5: implement one MCTS simulation.
- Session 6: run self-play for a few games and inspect replay rows.
- Session 7: implement losses and a tiny training step.
- Session 8: run arena evaluation against a random policy.

## Tests

```bash
pytest modules/alpha_zero/tests -m alpha_zero_env
pytest modules/alpha_zero/tests -m alpha_zero_mcts
pytest modules/alpha_zero/tests -m alpha_zero_self_play
```

## Resources

- AlphaZero paper for the self-play/search/training loop.
- PettingZoo Classic documentation for `tictactoe` and `connect_four`.
- Gymnasium/PettingZoo API references for reset/step/action-space conventions.

## What Agents May Help With

Agents may explain concepts, help interpret tests, improve scaffolding, and add new tests. Agents must not implement the learner-facing solution methods in this module.
