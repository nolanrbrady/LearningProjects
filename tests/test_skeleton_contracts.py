"""Static tests for learner-facing skeleton contracts."""

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
LEARNER_FILES = (
    "modules/micro_gpt/data/tokenizer.py",
    "modules/micro_gpt/data/dataset.py",
    "modules/micro_gpt/models/attention.py",
    "modules/micro_gpt/models/transformer.py",
    "modules/micro_gpt/training/trainer.py",
    "modules/micro_gpt/evaluation/generate.py",
    "modules/micro_text_diffusion/data/dataset.py",
    "modules/micro_text_diffusion/models/noising.py",
    "modules/micro_text_diffusion/models/denoiser.py",
    "modules/micro_text_diffusion/training/losses.py",
    "modules/micro_text_diffusion/training/trainer.py",
    "modules/micro_text_diffusion/evaluation/sample.py",
    "modules/alpha_zero/data/replay_buffer.py",
    "modules/alpha_zero/models/network.py",
    "modules/alpha_zero/models/mcts.py",
    "modules/alpha_zero/training/self_play.py",
    "modules/alpha_zero/evaluation/arena.py",
    "modules/alpha_zero/utils/observations.py",
    "modules/alpha_fold_toy/data/dataset.py",
    "modules/alpha_fold_toy/models/features.py",
    "modules/alpha_fold_toy/models/evoformer_lite.py",
    "modules/alpha_fold_toy/models/distogram.py",
    "modules/alpha_fold_toy/models/model.py",
    "modules/alpha_fold_toy/training/losses.py",
    "modules/alpha_fold_toy/evaluation/metrics.py",
)


def _public_functions_and_methods(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                yield node


@pytest.mark.scaffold
@pytest.mark.parametrize("relative_path", LEARNER_FILES)
def test_learner_files_raise_not_implemented(relative_path):
    text = (ROOT / relative_path).read_text()

    assert "NotImplementedError" in text, relative_path


@pytest.mark.scaffold
@pytest.mark.parametrize("relative_path", LEARNER_FILES)
def test_public_skeleton_functions_and_methods_have_docstrings(relative_path):
    tree = ast.parse((ROOT / relative_path).read_text())

    public_callables = list(_public_functions_and_methods(tree))
    assert public_callables, relative_path
    for node in public_callables:
        assert ast.get_docstring(node), f"{relative_path}:{node.name} missing docstring"


@pytest.mark.scaffold
def test_alpha_zero_env_adapter_is_scaffold_only_not_a_custom_environment():
    text = (ROOT / "modules" / "alpha_zero" / "utils" / "envs.py").read_text()

    assert "pettingzoo.classic" in text
    assert "class " not in text
    assert "SUPPORTED_CLASSIC_ENVS" in text
