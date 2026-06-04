# Module Creation Template

Use this template whenever a new learning module is requested. The agent should create the scaffold fully, but leave the learning implementation as docstring-heavy skeletons with `NotImplementedError`.

## Required Directory Shape

```text
modules/<module_name>/
  README.md
  data/
  models/
  training/
  evaluation/
  utils/
  tests/
```

Each Python package directory should include `__init__.py`.

## Required README Sections

- Overview
- Learning Goals
- Dataset Or Environment
- Information Flow
- Milestones
- Suggested 30-60 Minute Sessions
- Tests
- Resources
- What Agents May Help With

## Skeleton Rules

- Every learner-facing function or method needs a docstring.
- Docstrings should name inputs, outputs, expected shapes, and why the function matters in the module.
- Learner-facing implementations should raise `NotImplementedError`.
- Scaffold-only helpers may be implemented when they are not the concept being learned.
- RL modules should use Gymnasium, PettingZoo, Minari, or another maintained environment/data API when possible.
- Skeletons should outline the complete guided project path from data/environment intake through features, model or algorithm internals, training/search, evaluation, and sampling or deployment-style inspection when relevant.
- Do not skip major conceptual components by leaving them only in prose. If a step is part of the module's learning flow, represent it with a learner-facing function, class, method, or an explicit scaffold-only helper.

## Information Flow Rules

- Add an `Information Flow` README section that lists the concrete public functions/classes in execution order.
- The flow should show how outputs from one component become inputs to the next component.
- Include the final feedback loop: loss, metric, arena result, generated sample, visualization, or other inspection signal that tells the learner whether the implementation is improving.
- Keep the flow consolidated. A learner should be able to read it once and understand the full guided project without guessing which pieces are missing.

## Test Rules

- Scaffold tests should pass immediately.
- Milestone tests should be targeted and small.
- Tests that describe unfinished learner work should be marked `xfail` until the learner completes that milestone.
- Markers should be module-specific, such as `micro_gpt_tokenizer` or `dqn_cartpole_replay`.
- Every public learner-facing function, class, and method in the module should have at least one milestone test that states its expected contract.
- Tests should make correctness diagnosable, not just confirm that code runs. Cover input/output shapes, deterministic examples, edge cases, validation errors, mutation guarantees, masking or batching semantics, and important invariants for the component.
- Prefer several focused tests over one broad smoke test. A learner should be able to run the module tests and see exactly which component or method still needs work.
- Keep tests implementation-agnostic where possible. It is fine to accept lists, NumPy arrays, or torch tensors when the contract only requires shape and values.
- Do not leave major module components untested because they are in a later milestone. Add `xfail` contract tests for future milestones so the full suite documents the complete intended implementation.

## Module README Agent Boundary

Include this text in each module README:

> Agents may explain concepts, help interpret tests, improve scaffolding, and add new tests. Agents must not implement the learner-facing solution methods in this module.
