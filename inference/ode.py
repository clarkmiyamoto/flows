import torch
from abc import ABC, abstractmethod
from .path import ConditionalProbabilityPath

class ODE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Returns the drift coefficient of the ODE.
        Args:
            - xt: state at time t, shape (batch_size, dim)
            - t: time, shape (batch_size, 1)
        Returns:
            - drift_coefficient: shape (batch_size, dim)
        """
        pass

# Example ODEs

class ProbabilityFlow(ODE):
    """
    Implements equation (13) in https://openreview.net/pdf?id=PxTIG12RRHS
    
    dx/dt = f(x,t) - 0.5 g(t)^2 \nabla_x \log p_t(x)
    """

    def __init__(self, score):
        self.score = score

    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.score(xt, t)

class ConditionalVectorFieldODE(ODE):
    def __init__(self, path: ConditionalProbabilityPath, z: torch.Tensor):
        """
        Args:
        - path: the ConditionalProbabilityPath object to which this vector field corresponds
        - z: the conditioning variable, (1, dim)
        """
        super().__init__()
        self.path = path
        self.z = z

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Returns the conditional vector field u_t(x|z)
        Args:
            - x: state at time t, shape (bs, dim)
            - t: time, shape (bs,.) or (1,)
        Returns:
            - u_t(x|z): shape (batch_size, dim)
        """
        bs = x.shape[0]
        z = self.z.expand(bs, *self.z.shape[1:])
        # Ensure t has the right shape for broadcasting
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.shape[0] == 1:
            t = t.expand(bs, *t.shape[1:])
        return self.path.conditional_vector_field(x,z,t)

class LearnedVectorFieldODE(ODE):
    def __init__(self, net: torch.nn.Module):
        self.net = net

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: (bs, dim)
            - t: (bs, dim)
        Returns:
            - u_t: (bs, dim)
        """
        return self.net(x, t)