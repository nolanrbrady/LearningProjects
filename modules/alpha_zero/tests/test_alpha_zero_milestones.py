"""Milestone tests for alpha_zero."""

import importlib.util

import pytest

from modules.alpha_zero.data.replay_buffer import ReplayBuffer
from modules.alpha_zero.evaluation.arena import evaluate_agents
from modules.alpha_zero.models.mcts import MCTSNode
from modules.alpha_zero.models.mcts import MCTS
from modules.alpha_zero.models.network import PolicyValueNetwork
from modules.alpha_zero.training.self_play import run_self_play_game, train_step
from modules.alpha_zero.utils.observations import encode_observation, legal_actions_from_observation
from modules.alpha_zero.utils.envs import make_pettingzoo_env, reset_env


@pytest.mark.alpha_zero_env
def test_pettingzoo_tictactoe_adapter_resets_external_environment():
    if importlib.util.find_spec("pettingzoo") is None:
        pytest.skip("pettingzoo is not installed")

    env = make_pettingzoo_env("tictactoe")
    returned = reset_env(env, seed=0)

    assert returned is env
    assert env.agents
    assert env.agent_selection in env.agents
    env.close()


@pytest.mark.alpha_zero_env
def test_pettingzoo_adapter_rejects_unsupported_environment_names():
    with pytest.raises(ValueError, match="Unsupported"):
        make_pettingzoo_env("not_a_game")


@pytest.mark.alpha_zero_env
@pytest.mark.xfail(reason="Learner has not implemented observation handling yet.", strict=False)
def test_observation_helpers_extract_state_and_legal_actions_from_action_mask():
    raw_observation = {"observation": [[1, 0], [0, 1]], "action_mask": [1, 0, 1, 0]}

    encoded = encode_observation(raw_observation)
    legal_actions = legal_actions_from_observation(raw_observation)

    assert encoded == [[1, 0], [0, 1]]
    assert legal_actions == [0, 2]


@pytest.mark.alpha_zero_env
@pytest.mark.xfail(reason="Learner has not implemented action-mask validation yet.", strict=False)
def test_legal_actions_from_observation_rejects_missing_action_mask():
    with pytest.raises((KeyError, ValueError)):
        legal_actions_from_observation({"observation": [0.0]})


@pytest.mark.alpha_zero_self_play
@pytest.mark.xfail(reason="Learner has not implemented PolicyValueNetwork yet.", strict=False)
def test_policy_value_network_returns_policy_logits_and_bounded_value():
    network = PolicyValueNetwork(observation_shape=(3, 3, 2), num_actions=9)

    policy_logits, value = network.forward(
        observation=[[[0.0, 1.0]] * 3] * 3,
        action_mask=[1, 0, 1, 0, 1, 0, 1, 0, 1],
    )

    assert len(policy_logits) == 9
    assert float(value) >= -1.0
    assert float(value) <= 1.0


@pytest.mark.alpha_zero_self_play
@pytest.mark.xfail(reason="Learner has not implemented policy masking yet.", strict=False)
def test_policy_value_network_masks_illegal_actions():
    network = PolicyValueNetwork(observation_shape=(1,), num_actions=4)

    policy_logits, _ = network.forward(observation=[0.0], action_mask=[1, 0, 1, 0])

    assert policy_logits[1] < policy_logits[0]
    assert policy_logits[3] < policy_logits[2]


@pytest.mark.alpha_zero_mcts
@pytest.mark.xfail(reason="Learner has not implemented MCTSNode yet.", strict=False)
def test_mcts_node_value_uses_backed_up_mean():
    node = MCTSNode(prior=0.25, to_play="player_1")
    node.visit_count = 2
    node.value_sum = 1.0

    assert node.value() == pytest.approx(0.5)


@pytest.mark.alpha_zero_mcts
@pytest.mark.xfail(reason="Learner has not implemented MCTSNode initialization yet.", strict=False)
def test_mcts_node_initializes_search_statistics():
    node = MCTSNode(prior=0.25, to_play="player_1")

    assert node.prior == pytest.approx(0.25)
    assert node.to_play == "player_1"
    assert node.visit_count == 0
    assert node.value_sum == 0.0
    assert node.children == {}
    assert node.value() == 0.0


@pytest.mark.alpha_zero_mcts
@pytest.mark.xfail(reason="Learner has not implemented MCTS search yet.", strict=False)
def test_mcts_run_returns_normalized_policy_over_legal_actions_without_stepping_real_env():
    class DummyEnv:
        agent_selection = "player_1"

        def __init__(self):
            self.step_calls = 0

        def observe(self, agent):
            return {"observation": [0.0], "action_mask": [1, 0, 1]}

        def step(self, action):
            self.step_calls += 1

    class DummyNetwork:
        def predict(self, observation):
            return [0.7, 0.2, 0.1], 0.0

    env = DummyEnv()
    policy = MCTS(DummyNetwork(), num_simulations=4, c_puct=1.5).run(env)

    assert set(policy) == {0, 2}
    assert sum(policy.values()) == pytest.approx(1.0)
    assert all(probability >= 0.0 for probability in policy.values())
    assert env.step_calls == 0


@pytest.mark.alpha_zero_self_play
@pytest.mark.xfail(reason="Learner has not implemented ReplayBuffer yet.", strict=False)
def test_replay_buffer_respects_capacity_and_samples_training_columns():
    buffer = ReplayBuffer(capacity=2)

    buffer.add("state_0", {0: 1.0}, -1.0)
    buffer.add("state_1", {1: 1.0}, 0.0)
    buffer.add("state_2", {2: 1.0}, 1.0)
    states, policies, values = buffer.sample(batch_size=2)

    assert len(states) == 2
    assert len(policies) == 2
    assert len(values) == 2
    assert "state_0" not in states
    assert set(states) == {"state_1", "state_2"}


@pytest.mark.alpha_zero_self_play
@pytest.mark.xfail(reason="Learner has not implemented replay sampling validation yet.", strict=False)
def test_replay_buffer_rejects_oversized_samples():
    buffer = ReplayBuffer(capacity=2)
    buffer.add("state", {0: 1.0}, 1.0)

    with pytest.raises(ValueError):
        buffer.sample(batch_size=2)


@pytest.mark.alpha_zero_self_play
@pytest.mark.xfail(reason="Learner has not implemented self-play yet.", strict=False)
def test_run_self_play_game_returns_policy_and_value_targets_for_each_state():
    class OneMoveEnv:
        agent_selection = "player_1"
        rewards = {"player_1": 1.0, "player_2": -1.0}

        def __init__(self):
            self.agents = ["player_1", "player_2"]
            self.done = False

        def observe(self, agent):
            return {"observation": f"state_for_{agent}", "action_mask": [1, 0]}

        def step(self, action):
            self.done = True
            self.agents = []

    class DummySearch:
        def run(self, env):
            return {0: 1.0}

    examples = run_self_play_game(OneMoveEnv(), network=None, search=DummySearch())

    assert examples
    state, policy, value = examples[0]
    assert state == "state_for_player_1"
    assert policy == {0: 1.0}
    assert value == pytest.approx(1.0)


@pytest.mark.alpha_zero_self_play
@pytest.mark.xfail(reason="Learner has not implemented AlphaZero train_step yet.", strict=False)
def test_train_step_returns_policy_and_value_losses_and_steps_optimizer():
    class DummyOptimizer:
        def __init__(self):
            self.zero_grad_calls = 0
            self.step_calls = 0

        def zero_grad(self):
            self.zero_grad_calls += 1

        def step(self):
            self.step_calls += 1

    class DummyNetwork:
        def __call__(self, states):
            return [[0.6, 0.4]], [0.25]

    optimizer = DummyOptimizer()
    losses = train_step(DummyNetwork(), replay_batch=(["state"], [[1.0, 0.0]], [1.0]), optimizer=optimizer)

    assert {"policy_loss", "value_loss", "total_loss"} <= set(losses)
    assert all(float(value) >= 0.0 for value in losses.values())
    assert optimizer.zero_grad_calls == 1
    assert optimizer.step_calls == 1


@pytest.mark.alpha_zero_self_play
@pytest.mark.xfail(reason="Learner has not implemented arena evaluation yet.", strict=False)
def test_evaluate_agents_returns_game_outcome_summary():
    class CandidateAlwaysWinsEnv:
        def __init__(self):
            self.agents = ["candidate", "baseline"]
            self.agent_selection = "candidate"
            self.rewards = {"candidate": 1.0, "baseline": -1.0}

        def reset(self, seed=None):
            self.agents = ["candidate", "baseline"]

        def observe(self, agent):
            return {"action_mask": [1]}

        def step(self, action):
            self.agents = []

        def close(self):
            pass

    class FirstActionAgent:
        def act(self, observation):
            return 0

    summary = evaluate_agents(CandidateAlwaysWinsEnv, FirstActionAgent(), FirstActionAgent(), num_games=2)

    assert summary["wins"] == 2
    assert summary["losses"] == 0
    assert summary["draws"] == 0
    assert summary["num_games"] == 2
