from abc import ABC, abstractmethod

import torch
import torch.jit as jit
from inference.distribution import Sampleable

class FieldTheory(Sampleable, jit.ScriptModule, ABC):
    
    def __init__(self, 
                 spacetime_dim: int,
                 device = "cuda" if torch.cuda.is_available() else "cpu"):
        '''
        Args:
            lattice_points (int): number of lattice points in theory
            spacetime_dim (int): number of space dimensions
        '''
        self.spacetime_dim = spacetime_dim
        self.device = device
    
    @abstractmethod
    def action(self, phi: torch.Tensor) -> torch.Tensor:
        '''
        Evaluates action of correpsonding field configuration

        Args:
            phi (torch.Tensor): Field configuration. Shape (batch_size, power(lattice_points, spacetime_dim))
        
        Returns:
            (torch.Tensor): Shape (batch_size,)
        '''
        pass

    @property
    def dim(self) -> int:
        pass

    def _hmc_hamiltonian(self, chi: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute the Hamiltonian H(φ, χ) = K(χ) + V(φ), where
        K = ½ ∑ χ_x²    is the kinetic energy,
        V = S[φ]        is the potential energy (the action).

        Args:
            chi (torch.Tensor): Auxiliary momenta χ_x. Shape (batch_size, power(lattice_points, spacetime_dim))
            action (torch.Tensor): Evalulation of the action. Shape (batch_size,)

        Returns:
            The total Hamiltonian H.
        """
        kinetic = 0.5 * (chi**2).view(chi.size(0), -1).sum(dim=1) # Shape (batch_size,)
        return kinetic + action
    
    @abstractmethod
    def _hmc_drift(self, phi: torch.Tensor) -> torch.Tensor:
        """
        Compute the drift (i.e. the force) ∂S/∂φ\_x.

        In HMC, the “drift” is the gradient of the action with respect to φ,
        which plays the role of minus the force in Hamilton’s equations.

        Args:
            phi (torch.Tensor): Field configuration. Shape (batch_size, lattice_points, lattice_points, ....)

        Returns:
            Shape (batch_size, lattice_points, lattice_points, ....) (same shape as argument: `phi`)
        """
        pass
    
    @jit.script_method
    def sample(self,
               phi_0: torch.Tensor, 
               n_steps: int = 100) -> torch.Tensor:
        """
        Perform one Hamiltonian Monte Carlo update on the field φ.

        Steps:
        1. Sample initial momenta χ ~ N(0, I).
        2. Compute initial H₀ = K(χ) + S₀.
        3. Do a “leapfrog” integration of length n_steps with step dt = 1/n_steps:
            a. χ ← χ + (dt/2) * (-∂S/∂φ)        [half-step momentum update]
            b. φ ← φ + dt * χ                   [full-step position update]
            c. χ ← χ + dt * (-∂S/∂φ)            [full-step momentum update]
            … repeat b/c n_steps-1 times …
            d. φ ← φ + dt * χ                   [last full-step position]
            e. χ ← χ + (dt/2) * (-∂S/∂φ)        [final half-step momentum]
        4. Compute new action S and Hamiltonian H.
        5. Metropolis accept/reject: if ΔH = H - H₀ > 0, accept with probability e^(-ΔH).

        Args:
            phi_0 (torch.Tensor): initial field configuration (batch_size, lattice_points, lattice_points, ...).
            n_steps (int): number of leapfrog steps.

        Returns:
            phi_new, 
            S_new, 
            accepted)
        """
        dt = 1 / n_steps
        k = 1
        l = 1

        phi = phi_0 # Initial field configuration
        chi = torch.randn_like(phi, device=self.device) # Momentum field (for HMC)
        S_0 = self.action(phi) # Action of initial field configuration

        H_0 = self._hmc_hamiltonian(chi, S_0) # Initial energy difference (for HMC)

        #3) Leapfrog integrator
        chi += 0.5 * dt * self._hmc_drift(phi)
        for i in range(n_steps-1):
            phi += dt * chi
            chi += dt * self._hmc_drift(phi)
        phi += dt * chi
        chi += 0.5 * dt * self._hmc_drift(phi)

        # 4) Compute New Action & Hamiltonian
        S = self.action(phi)                     # Shape (batch_size,)
        dH = self._hmc_hamiltonian(chi, S) - H_0 # Shape (batch_size,)

        # 5) Metroplis Step
        accept_prob = torch.exp(-dH).clamp(max=1.0) # Shape (batch_size,)

        u = torch.rand_like(accept_prob, device=self.device)
        accept_mask = (u < accept_prob)

        new_phi = torch.where(
            accept_mask.view(-1, *([1] * (phi.dim()-1))),  # broadcast to phi.shape
            phi,
            phi_0
        )
        new_S = torch.where(
            accept_mask,
            S,
            S_0
        )
        return new_phi, new_S, accept_mask