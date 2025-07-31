from abc import ABC, abstractmethod
import torch
from .path import ConditionalProbabilityPath

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

class LangevinFlowSDE(SDE):
    def __init__(self, flow_model: torch.nn.Module, score_model: torch.nn.Module, sigma: float):
        """
        Args:
        - path: the ConditionalProbabilityPath object to which this vector field corresponds
        - z: the conditioning variable, (1, dim)
        """
        super().__init__()
        self.flow_model = flow_model
        self.score_model = score_model
        self.sigma = sigma

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: state at time t, shape (bs, dim)
            - t: time, shape (bs,.)
        Returns:
            - u_t(x|z): shape (batch_size, dim)
        """
        return self.flow_model(x,t) + 0.5 * self.sigma ** 2 * self.score_model(x, t)

    def diffusion_coefficient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: state at time t, shape (bs, dim)
            - t: time, shape (bs,.)
        Returns:
            - u_t(x|z): shape (batch_size, dim)
        """
        return self.sigma * torch.randn_like(x)

class ConditionalVectorFieldSDE(SDE):
    def __init__(self, path: ConditionalProbabilityPath, z: torch.Tensor, sigma: float):
        """
        Args:
        - path: the ConditionalProbabilityPath object to which this vector field corresponds
        - z: the conditioning variable, (1, ...)
        """
        super().__init__()
        self.path = path
        self.z = z
        self.sigma = sigma

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
        return self.path.conditional_vector_field(x,z,t) + 0.5 * self.sigma**2 * self.path.conditional_score(x,z,t)

    def diffusion_coefficient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: state at time t, shape (bs, dim)
            - t: time, shape (bs,.)
        Returns:
            - u_t(x|z): shape (batch_size, dim)
        """
        return self.sigma * torch.randn_like(x)