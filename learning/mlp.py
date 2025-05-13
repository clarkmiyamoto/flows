import torch
import torch.jit as jit
from typing import List, Type


# ────────────────────────────────────────────────────────────────────────────────
# Helper: Sinusoidal time embedding
# ────────────────────────────────────────────────────────────────────────────────
class SinusoidalTimeEmbedding(jit.ScriptModule):
    """
    Classic transformer‑style positional embedding, but for scalar time.
    """
    def __init__(self, 
                 embed_dim: int = 64, 
                 max_period: torch.Tensor = torch.tensor(10_000.0)):
        super().__init__()
        self.embed_dim = embed_dim

        # Pre‑compute 1 / f where f = periods spaced geometrically
        inv_freq = torch.exp(
            -torch.log(max_period) * torch.arange(0, embed_dim, 2).float() / embed_dim
        )                                           # (embed_dim/2,)
        self.register_buffer("inv_freq", inv_freq)  # non‑trainable

    @jit.script_method
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
          t: (bs,) or (bs,1) – scalar time in ℝ
        Returns:
          (bs, embed_dim) – concatenated sin / cos embeddings
        """
        t = t.view(-1, 1)                           # (bs,1)
        sinusoid = t * self.inv_freq                # broadcast multiply
        return torch.cat([sinusoid.sin(), sinusoid.cos()], dim=-1)


# ────────────────────────────────────────────────────────────────────────────────
# MLP Builder
# ────────────────────────────────────────────────────────────────────────────────
def build_mlp(dims: List[int],
              activation: Type[torch.nn.Module] = torch.nn.SiLU) -> torch.nn.Sequential:
    layers: List[torch.nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(activation())
    return torch.nn.Sequential(*layers)


# ────────────────────────────────────────────────────────────────────────────────
# Vector‑field network with skip connection + time embedding
# ────────────────────────────────────────────────────────────────────────────────
class MLPVectorField(jit.ScriptModule):
    """
    u_θ(t, x) parameterised by an MLP over [flatten(x) ‖ emb(t)],
    with a residual connection:  u = x + f([x,t]).
    """
    def __init__(
        self,
        dim: int,                       # L (field is L×L)
        hiddens: List[int],             # e.g. [512, 512, 512]
        time_embed_dim: int = 64,
        activation: Type[torch.nn.Module] = torch.nn.SiLU,
    ):
        super().__init__()
        self.dim = dim

        # Time embedding module
        self.time_emb = SinusoidalTimeEmbedding(time_embed_dim)

        # Core network: input = L² + time_embed_dim → ... → L²
        in_dim = dim + time_embed_dim
        self.net = build_mlp([in_dim] + hiddens + [dim], activation)

    # --------------------------------------------------------------------------
    @jit.script_method
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        x : (bs, L, L)      – lattice configuration
        t : (bs,) or (bs,1) – scalar time

        Returns
        -------
        u_t^θ(x) : (bs, L, L)
        """
        bs = x.shape[0]
        x_flat = x.view(bs, -1)                     # (bs, L²)
        t_emb = self.time_emb(t)                   # (bs, time_embed_dim)
        h = torch.cat([x_flat, t_emb], dim=-1)     # (bs, L² + time_embed_dim)

        dx = self.net(h)                           # (bs, L²)
        out_flat = x_flat + dx                     # <-- residual / skip
        return out_flat.view_as(x)                 # reshape to (bs, L, L)
