# LearningProjects Agent Rules

This repository is a learning workspace. Its purpose is to help the human build deep learning and reinforcement learning systems from first principles while receiving structure, feedback, and conceptual guidance.

## Core Rule

Do not solve learner-facing implementation tasks.

Agents must not fill in TODOs, complete milestone methods, write answer code for the central model or algorithm being studied, or otherwise remove the productive struggle from the module. When the human asks for help inside a module, guide them toward understanding through questions, explanations, failing-test interpretation, debugging strategy, and pointers to the relevant README/resource section.

## Scaffolding Is Expected

The no-solution rule does not forbid creating new modules. Creating modules is allowed and expected.

When the human asks for a new learning module, agents may and should create the complete scaffold: folders, README, milestone plan, docstring-heavy skeletons, tests, fixtures, configs, dataset adapters, environment adapters, and helper scripts that define the exercise. These scaffolds should make the learning path clear without implementing the learner-facing solution.

## Allowed Work

Agents may:

- Create new modules under `modules/`.
- Create or update `README.md`, `MODULE_TEMPLATE.md`, tests, fixtures, configs, and documentation.
- Create empty classes/functions/methods with detailed docstrings and `NotImplementedError` placeholders.
- Create tests that describe expected behavior, including milestone tests that fail or are marked `xfail` until the learner implements the relevant step.
- Patch tests and docs to improve clarity, correctness, coverage, or diagnostics.
- Implement lightweight scaffold infrastructure that is not the learning objective, such as package metadata, simple config containers, dataset-name constants, and adapters to external RL environment APIs.
- Run tests, explain failures, and suggest the next smallest experiment.

## Disallowed Work

Agents must not:

- Implement learner-facing algorithms, model internals, training loops, losses, tokenizers, search logic, or evaluation logic.
- Replace a learning milestone with copied code from a tutorial, paper implementation, library, or previous solution.
- Provide full solution snippets that the human can paste directly into a learner-facing TODO.
- Make hidden implementation changes that cause milestone tests to pass without the human doing the work.
- Build custom RL environments when a maintained environment API can provide the task. RL time should be spent on algorithms and models, not environment engineering.

## Help Style

When the human asks for implementation help, prefer:

- Restating the expected input/output contract.
- Explaining the concept and why the shape/math matters.
- Asking one focused question that helps locate the misunderstanding.
- Pointing to the specific failing test and what it is checking.
- Suggesting a tiny experiment or print/debug observation.
- Giving pseudocode only when it does not become a pasteable implementation.

If the human explicitly asks an agent to solve a learner-facing method, refuse that part briefly and offer guided help instead.

## Module Standards

Every module should:

- Live under `modules/<module_name>/`.
- Include `data/`, `models/`, `training/`, `evaluation/`, `utils/`, and `tests/`.
- Include a `README.md` with the task, learning goals, dataset/environment, milestones sized for 30-60 minute sessions, resources, and acceptance tests.
- Prefer real toy datasets from Hugging Face when appropriate.
- Prefer external RL environment APIs, especially Gymnasium for single-agent RL and PettingZoo for multi-agent/board-game RL.
- Keep skeleton docstrings explicit about inputs, outputs, tensor shapes, and how the method fits into the full project.
- Include comprehensive milestone tests for every public learner-facing function, class, and method. These tests should document shapes, deterministic examples, edge cases, expected validation errors, mutation guarantees, masking or batching semantics, and core invariants so test results clearly identify which component is correct or incomplete.
- Keep unfinished learner-facing contracts marked `xfail` rather than omitting them, so future modules expose the full intended behavior before the learner implements it.

## Default Commands

- Run scaffold checks with `pytest tests`.
- Run a module's daily checks with `pytest modules/<module_name>/tests -m <marker>`.
- Keep dataset downloads optional, row-limited, and cache-aware.
