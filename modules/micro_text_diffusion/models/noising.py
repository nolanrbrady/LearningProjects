"""Token noising schedule skeletons for micro_text_diffusion."""


class TokenNoisingSchedule:
    """Discrete corruption schedule for text diffusion."""

    def __init__(self, num_timesteps: int, min_noise: float = 0.0, max_noise: float = 1.0):
        """Store the timestep range and noise bounds.

        Args:
            num_timesteps: Number of discrete denoising steps.
            min_noise: Corruption probability at the earliest timestep.
            max_noise: Corruption probability at the final timestep.

        Returns:
            None.

        The schedule controls how difficult the reconstruction problem is at
        each timestep.
        """
        raise NotImplementedError("Implement noising schedule initialization.")

    def probability(self, timestep: int) -> float:
        """Return the corruption probability for one timestep.

        Args:
            timestep: Integer timestep in `[0, num_timesteps)`.

        Returns:
            Corruption probability between 0.0 and 1.0.

        This function is the scalar contract used by the token corruption
        routine and by tests that check monotonic noise.
        """
        raise NotImplementedError("Implement timestep-to-probability mapping.")

    def corrupt(self, token_ids, timestep: int, mask_token_id: int):
        """Corrupt token IDs according to the timestep's probability.

        Args:
            token_ids: Tensor-like integer IDs with shape `(batch, sequence)`.
            timestep: Current diffusion timestep.
            mask_token_id: Token ID used to replace corrupted positions.

        Returns:
            A pair `(corrupted_ids, corruption_mask)` with the same batch and
            sequence dimensions as `token_ids`.

        Corruption creates the supervised denoising problem solved by the model.
        """
        raise NotImplementedError("Implement token corruption as a learning milestone.")
