import torch
from abc import ABC, abstractmethod

class Scheduler(ABC):

    def __call__(self, t: torch.Tensor):
        return self.interpolate(t)

    @abstractmethod
    def interpolate(self, t: torch.tensor):
        """
        Interpolates noise for the given time tensor.
        Args:
            t: tensor of shape (batch_size, 1)
        Returns:
            noise: tensor of shape (batch_size, 1)
        """
        raise NotImplementedError("Subclasses must implement this method.")


class Linear(Scheduler):

    def __init__(self, start, end):
        self.start = start
        self.end = end

    def interpolate(self, t: torch.tensor):
        """
        Linearly interpolates noise from `self.start` to `self.end` over the range of t.

        Args:
            t: time tensor of shape (batch_size, 1)
        Returns:
            noise: tensor of shape (batch_size, 1)
        """
        t_min = t.min()
        t_max = t.max()
        # Normalize t to [0, 1]
        t_norm = (t - t_min) / (t_max - t_min + 1e-8)
        return self.start + (self.end - self.start) * t_norm


class Sigmoid(Scheduler):

    def __init__(self, start, end, steepness=0.6):
        self.start = start
        self.end = end
        self.steepness = steepness

    def interpolate(self, t: torch.tensor):
        """
        Applies a sigmoid decay from 1.0 to 0.01 over the range of t.

        Args:
            t: tensor of shape (batch_size, 1)
        Returns:
            noise: tensor of shape (batch_size, 1)
        """
        t_min = t.min()
        t_max = t.max()
        t_norm = (t - t_min) / (t_max - t_min + 1e-8)  # Normalize to [0,1]

        # Sigmoid decay: steeper in the middle, flatter near ends
        sigmoid_scaled = torch.sigmoid(self.steepness * (1 - 2 * t_norm))
        return self.end + (self.start - self.end) * sigmoid_scaled

