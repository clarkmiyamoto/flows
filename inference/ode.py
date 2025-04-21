from abc import ABC, abstractmethod
import torch
from .simulator import Simulator

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

class EulerSimulator(Simulator):
    def __init__(self, ode: ODE):
        self.ode = ode
        
    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor):
        return xt + self.ode.drift_coefficient(xt,t) * h

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