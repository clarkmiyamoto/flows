import torch
from tqdm import tqdm
from abc import ABC, abstractmethod

from .ode import ODE
from .sde import SDE

class Simulator(ABC):
    @abstractmethod
    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor):
        """
        Takes one simulation step
        Args:
            - xt: state at time t, shape (bs, dim)
            - t: time, shape (bs,1)
            - dt: time, shape (bs,1)
        Returns:
            - nxt: state at time t + dt (bs, dim)
        """
        pass
    
    def _canon_ts(self, ts: torch.Tensor, batch: int) -> torch.Tensor:
        """
        Returns `ts` as shape (batch, T) on the same device/dtype as the state
        """
        # (T,)  →  (1, T) → (batch, T)   (cheap: expand() is a view)
        if ts.dim() == 1:
            ts = ts.unsqueeze(0).expand(batch, -1)

        # (batch, T, 1) → (batch, T)
        elif ts.dim() == 3 and ts.size(-1) == 1:
            ts = ts.squeeze(-1)

        # (batch, T) is already fine
        elif ts.dim() != 2:
            raise ValueError(
                "ts must be (T,), (batch,T) or (batch,T,1); got {}".format(ts.shape)
            )

        if ts.size(0) != batch:
            raise ValueError(
                f"Batch mismatch: x has {batch}, ts has {ts.size(0)}"
            )
        return ts

    @torch.no_grad()
    def simulate(self, x: torch.Tensor, ts: torch.Tensor):
        """
        Simulates using the discretization gives by ts
        Args:
            - x_init: initial state at time ts[0], shape (batch_size, dim)
            - ts: timesteps, shape (bs, num_timesteps,1)
        Returns:
            - x_final: final state at time ts[-1], shape (batch_size, dim)
        """
        for t_idx in range(len(ts) - 1):
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h)
        return x

    @torch.no_grad()
    def simulate_with_trajectory(self, x: torch.Tensor, ts: torch.Tensor):
        """
        x  : (batch, dim)
        ts : (T,), (batch, T) or (batch, T, 1)
        """
        bs, dim = x.shape
        ts = self._canon_ts(ts, bs)       # → (bs, T)
        T = ts.size(1)

        xs = torch.empty(bs, T, dim, device=x.device, dtype=x.dtype)
        xs[:, 0] = x                      # initial state

        for i in range(T - 1):
            t  = ts[:, i : i + 1]         # (bs, 1)  keeps last axis
            dt = ts[:, i + 1 : i + 2] - t # (bs, 1)
            x  = self.step(x, t, dt)      # user-defined integrator
            xs[:, i + 1] = x

        return xs                         # (bs, T, dim)

class EulerSimulator(Simulator):
    def __init__(self, ode: ODE):
        self.ode = ode

    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor):
        return xt + self.ode.drift_coefficient(xt,t) * h

class EulerMaruyamaSimulator(Simulator):
    def __init__(self, sde: SDE):
        self.sde = sde

    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor):
        return xt + self.sde.drift_coefficient(xt,t) * h + self.sde.diffusion_coefficient(xt,t) * torch.sqrt(h) * torch.randn_like(xt)

def record_every(num_timesteps: int, record_every: int) -> torch.Tensor:
    """
    Compute the indices to record in the trajectory given a record_every parameter
    """
    if record_every == 1:
        return torch.arange(num_timesteps)
    return torch.cat(
        [
            torch.arange(0, num_timesteps - 1, record_every),
            torch.tensor([num_timesteps - 1]),
        ]
    )