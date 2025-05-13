import torch
import torch.jit as jit
from typing import List, Type


def build_mlp(dims: List[int], activation: Type[torch.nn.Module] = torch.nn.SiLU):
        mlp = []
        for idx in range(len(dims) - 1):
            mlp.append(torch.nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                mlp.append(activation())
        return torch.nn.Sequential(*mlp)

class MLPVectorField(jit.ScriptModule):
    """
    MLP-parameterization of the learned vector field u_t^theta(x)
    """
    def __init__(self, dim: int, hiddens: List[int]):
        super().__init__()
        self.dim = dim
        self.net = build_mlp([dim+1] + hiddens + [dim])

    @jit.script_method
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Args:
        - x: (bs, L, L)
        - t: (bs, 1, 1)
        Returns:
        - u_t^theta(x): (bs, L, L)

        """
        og_shape = x.shape # (bs, L, L)

        x = x.view(x.shape[0], -1)  # shape: (bs, L*L)
        t = t.view(-1, 1)           # shape: (bs, 1)
        xt = torch.cat([x, t], dim=-1)  # shape: (bs, L*L + 1)
        xt = self.net(xt)

        return xt.view(og_shape)        # reshape to (bs, L, L)