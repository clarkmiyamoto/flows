from abc import ABC, abstractmethod
import torch
from torch.func import vmap, jacrev

from distribution import Sampleable, Density 


class FunctionalSampleable(ABC):

    def __init__(self, dim: int, lattice_points: int):
        """
        Args:
            - dim: Number of space dimensions
            - lattice_points: Number of lattice points in each dimension
        """
        self._lattice_points = lattice_points
        self._dim = dim
    
    @property
    def dim(self) -> int:
        return self._dim * self._lattice_points
    
    @abstractmethod
    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: shape (batch_size, _lattice_points, _lattice_points, ...)
                       num_samples goes on self._dim times.
        """
        pass

class FunctionalDensity(ABC):
    """
    Distribution with tractable density
    """
    @abstractmethod
    def log_density(self, phi: torch.Tensor) -> torch.Tensor:
        """
        Returns the log unnormalized density of the field configuration `phi`.

        Args:
            - phi: shape (batch_size, lattice_points, lattice_points, ...)
        Returns:
            - log_density: shape (batch_size,)
        """
        pass
    
    def score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the score dx log density(x)
        Args:
            - x: (batch_size, dim)
        Returns:
            - score: (batch_size, dim)
        """
        x = x.unsqueeze(1)  # (batch_size, 1, ...)
        score = vmap(jacrev(self.log_density))(x)  # (batch_size, 1, 1, 1, ...)
        return score.squeeze((1, 2, 3))  # (batch_size, ...)
    
class FieldTheory(FunctionalSampleable, FunctionalDensity):
    """
    Field theory with a functional distribution
    """

    def __init__(self, dim: int, lattice_points: int):
        super().__init__(dim, lattice_points)
        self._lattice_points = lattice_points
        self._dim = dim

    @abstractmethod
    def action(self, phi: torch.Tensor) -> torch.Tensor:
        """
        Returns the action of the field configuration `phi`.

        Args:
            - phi: shape (batch_size, num_samples, num_samples, ...)
        Returns:
            - action: shape (batch_size,)
        """
        pass

    def log_density(self, phi: torch.Tensor) -> torch.Tensor:
        return self.action(phi)

    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: shape (batch_size, num_samples, num_samples, ...)
                       num_samples goes on self._dim times.
        """
        # Implement the sampling logic here
        pass