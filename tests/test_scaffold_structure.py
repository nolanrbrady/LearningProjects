"""Repository-level scaffold integrity tests."""

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
INITIAL_MODULES = ("micro_gpt", "micro_text_diffusion", "alpha_zero", "alpha_fold_toy")
REQUIRED_DIRS = ("data", "models", "training", "evaluation", "utils", "tests")
REQUIRED_README_SECTIONS = (
    "## Overview",
    "## Learning Goals",
    "## Dataset Or Environment",
    "## Information Flow",
    "## Milestones",
    "## Suggested 30-60 Minute Sessions",
    "## Tests",
    "## Resources",
    "## What Agents May Help With",
)


@pytest.mark.scaffold
def test_root_scaffold_files_exist():
    for relative_path in ("AGENTS.md", "README.md", "MODULE_TEMPLATE.md", "pyproject.toml"):
        assert (ROOT / relative_path).exists(), relative_path


@pytest.mark.scaffold
def test_agents_policy_allows_scaffolding_but_forbids_solutions():
    text = (ROOT / "AGENTS.md").read_text()

    assert "Creating modules is allowed and expected" in text
    assert "Do not solve learner-facing implementation tasks" in text
    assert "Agents may" in text
    assert "Agents must not" in text
    assert "RL time should be spent on algorithms and models" in text


@pytest.mark.scaffold
@pytest.mark.parametrize("module_name", INITIAL_MODULES)
def test_initial_modules_have_required_directory_shape(module_name):
    module_root = ROOT / "modules" / module_name

    assert module_root.exists()
    assert (module_root / "README.md").exists()
    for directory in REQUIRED_DIRS:
        assert (module_root / directory).is_dir(), f"{module_name}/{directory}"
        assert (module_root / directory / "__init__.py").exists(), f"{module_name}/{directory}/__init__.py"


@pytest.mark.scaffold
@pytest.mark.parametrize("module_name", INITIAL_MODULES)
def test_initial_module_readmes_have_required_sections(module_name):
    readme = (ROOT / "modules" / module_name / "README.md").read_text()

    for section in REQUIRED_README_SECTIONS:
        assert section in readme, f"{module_name} missing {section}"
    assert "Agents must not implement the learner-facing solution methods" in readme


@pytest.mark.scaffold
def test_template_documents_future_module_creation_method():
    template = (ROOT / "MODULE_TEMPLATE.md").read_text()
    template_module = ROOT / "modules" / "_template"

    assert "Required Directory Shape" in template
    assert "Skeleton Rules" in template
    assert template_module.exists()
    for directory in REQUIRED_DIRS:
        assert (template_module / directory).is_dir()
