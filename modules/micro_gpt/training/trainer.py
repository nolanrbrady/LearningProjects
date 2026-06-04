"""Training-loop skeletons for micro_gpt."""


def train_one_epoch(model, batches, optimizer):
    """Run one epoch of next-token language-model training.

    Args:
        model: MicroGPT-like model returning logits and loss.
        batches: Iterable of `(x, y)` training batches.
        optimizer: Optimizer used to update model parameters.

    Returns:
        Mean training loss for the epoch.

    This loop connects the dataset contract to parameter updates while keeping
    the objective focused on predicting the next token.
    """
    raise NotImplementedError("Implement one training epoch as a learning milestone.")


def estimate_loss(model, batches):
    """Estimate loss without updating model parameters.

    Args:
        model: MicroGPT-like model returning a scalar loss when targets exist.
        batches: Iterable of validation or training batches.

    Returns:
        Mean loss over the provided batches.

    Evaluation loss is the main feedback signal for early language-model
    experiments before generated samples become meaningful.
    """
    raise NotImplementedError("Implement loss estimation as a learning milestone.")
