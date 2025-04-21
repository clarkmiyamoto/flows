from abc import ABC, abstractmethod
import torch
from .simulator import Simulator

class SDE(ABC):
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

    @abstractmethod
    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Returns the diffusion coefficient of the ODE.
        Args:
            - xt: state at time t, shape (batch_size, dim)
            - t: time, shape (batch_size, 1)
        Returns:
            - diffusion_coefficient: shape (batch_size, dim)
        """
        pass

class EulerMaruyamaSimulator(Simulator):
    def __init__(self, sde: SDE):
        self.sde = sde
        
    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor):
        return xt + self.sde.drift_coefficient(xt,t) * h + self.sde.diffusion_coefficient(xt,t) * torch.sqrt(h) * torch.randn_like(xt)

# Example SDEs

class OU(SDE):

    def __init__(self, theta, sigma):
        '''
        '''
        self.theta = theta
        self.sigma = sigma

    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor):
        return -1.0 * xt * self.theta
    
    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor):
        return self.sigma * torch.ones_like(xt)


class LangevinDynamics(SDE):
    ''' 
    Implements

    dX_t = (sigma_t^2 / 2) s(x) dt + sigma_t dW_t
    
    where s(x) = \nabla_x \log p(x)
    '''
    
    def __init__(self, score, noise_scheduler: torch.Tensor):
        '''
        The score is of the form `\nabla_x \log p(x)`

        The limiting distribution of LangevinDynamics is `p(x)`

        Args:
            score (func): Inputs tensor of shape (batch_size, dim), outputs tensor of shape (batch_size, dim).
                Can be `distribution.Density.score`.
            noise_scheduler: Inputs torch tensor (batch_size, 1), outputs tensor of shape (batch_size, 1)
        '''
        self.score = score
        self.noise_scheduler = noise_scheduler

    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor):
        _, dim = xt.shape
        noise = self.noise_scheduler(t).repeat(1, dim)

        return self.score(xt, t)
    
    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor):
        _, dim = xt.shape
        noise = self.noise_scheduler(t).repeat(1, dim)

        return noise