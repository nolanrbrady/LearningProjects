# Template Module

## Overview

Copy this scaffold when creating a new learning module. Replace placeholder names with the target method, architecture, or project.

## Learning Goals

- Identify the central concept the learner should implement.
- Separate scaffold-only helpers from learner-facing methods.
- Provide tests that make progress observable without giving away the solution.

## Dataset Or Environment

Name the real toy dataset or maintained environment API. Prefer Hugging Face datasets for data-driven modules, Gymnasium for single-agent RL, and PettingZoo for multi-agent RL.

## Information Flow

1. Load or reset the dataset/environment.
2. Convert raw inputs into the smallest useful learner-facing representation.
3. Pass that representation through the core model or algorithm component.
4. Connect the component to a loss, search objective, or training update.
5. Evaluate, sample, visualize, or otherwise inspect the final behavior.

Replace this placeholder flow with concrete public functions and classes when creating a real module. The flow should make it clear how information moves from one component to the next.

## Milestones

1. Understand the data/environment contract.
2. Implement the smallest useful representation.
3. Implement the core model or algorithm block.
4. Implement the training or search loop.
5. Evaluate and inspect the output.

## Suggested 30-60 Minute Sessions

- Session 1: read the README, run scaffold tests, and inspect docstrings.
- Session 2: implement one data contract and pass its tests.
- Session 3: implement one model/algorithm component and pass its tests.
- Session 4: connect the component to a training or evaluation smoke test.

## Tests

Run targeted tests with:

```bash
pytest modules/<module_name>/tests -m <module_marker>
```

## Resources

- Add one primary paper or tutorial.
- Add one implementation-independent conceptual explanation.
- Add one dataset or environment reference.

## What Agents May Help With

Agents may explain concepts, help interpret tests, improve scaffolding, and add new tests. Agents must not implement the learner-facing solution methods in this module.
