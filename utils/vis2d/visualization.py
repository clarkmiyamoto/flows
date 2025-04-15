from typing import Optional, List, Type, Tuple, Dict

import torch
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes

from inference.distribution import Density, Sampleable



def visualize_samples(samples: torch.Tensor, 
                      x_bounds: Tuple[float, float], 
                      y_bounds: Tuple[float, float], 
                      ax: Optional[Axes] = None, 
                      **kwargs):
    """
    Visualizes 2D samples on a given axis.

    Args:
        samples: Tensor of shape (num_samples, 2)
        x_bounds: Tuple defining x-axis bounds
        y_bounds: Tuple defining y-axis bounds
        ax: Optional matplotlib axis to plot on
        figsize: Used if ax is None
        kwargs: Passed to scatter
    """
    samples_np = samples.detach().cpu().numpy()
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(samples_np[:, 0], samples_np[:, 1], alpha=0.5, s=10, **kwargs)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(x_bounds)
    ax.set_ylim(y_bounds)
    ax.set_aspect("equal")


def visualize_density(density: Density, 
                      x_bounds: Tuple[float, float], 
                      y_bounds: Tuple[float, float], 
                      bins: int, 
                      device: torch.device, 
                      ax: Optional[Axes] = None, 
                      x_offset: float = 0.0, 
                      show_colorbar: bool = False, 
                      **kwargs):
    """
    Visualizes the log-density of a 2D distribution on a grid.

    Args:
        density: Object with method log_density(x), where x has shape (n, 2)
        x_bounds: Tuple (x_min, x_max)
        y_bounds: Tuple (y_min, y_max)
        bins: Number of grid points per axis
        device: Torch device
        ax: Axis to draw on
        x_offset: Constant shift applied to x
        show_colorbar: Whether to add a colorbar
        kwargs: Passed to imshow

    Returns:
        im: AxesImage from imshow
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    x = torch.linspace(x_min, x_max, bins).to(device) + x_offset
    y = torch.linspace(y_min, y_max, bins).to(device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    log_density = density.log_density(xy).reshape(bins, bins).T
    im = ax.imshow(log_density.cpu(), extent=[x_min, x_max, y_min, y_max], origin='lower', **kwargs)
    if show_colorbar:
        plt.colorbar(im, ax=ax)
    return im
