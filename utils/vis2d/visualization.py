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


def visualize_scalarfield(scalarfield,
                          x_bounds: Tuple[float, float], 
                          y_bounds: Tuple[float, float],
                          bins: int, 
                          device: torch.device, 
                          ax: Optional[Axes] = None,
                          x_offset: float = 0.0, 
                          show_colorbar: bool = False, 
                          **kwargs):
    """
    Visualizes a scalar field (f: R^2 -> R) on a grid.

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

    log_density = scalarfield(xy).reshape(bins, bins).T
    im = ax.imshow(log_density.cpu(), origin='lower', extent=[x_min, x_max, y_min, y_max], aspect='equal', **kwargs)
    ax.set_aspect('equal')
    if show_colorbar:
        plt.colorbar(im, ax=ax)
    return im


def visualize_contours_scalarfield(scalarfield,
                                   x_bounds: Tuple[float, float], 
                                   y_bounds: Tuple[float, float],
                                   bins: int, 
                                   device: torch.device, 
                                   ax: Optional[Axes] = None,
                                   x_offset: float = 0.0, 
                                   show_colorbar: bool = False, 
                                   **kwargs):
    """
    Visualizes a scalar field (f: R^2 -> R) on a grid.

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
    X, Y = torch.meshgrid(x, y)
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)

    field = scalarfield(xy).reshape(bins, bins)
    field = field.cpu()

    contour = ax.contourf(X, Y, field, levels=bins, **kwargs)
    if show_colorbar:
        plt.colorbar(contour, ax=ax)
    return contour



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
    im = visualize_scalarfield(density.log_density,
                               x_bounds = x_bounds,
                               y_bounds = y_bounds,
                               bins = bins,
                               device = device,
                               ax = ax,
                               x_offset = x_offset,
                               show_colorbar = show_colorbar,
                               **kwargs)
    return im


def visualize_vectorfield(vectorfield,
                          x_bounds: Tuple[float, float], 
                          y_bounds: Tuple[float, float],
                          num_arrows: int,
                          ax: Optional[Axes] = None,):
    """
    Visualizes 2d Vector Field on grid.

    Args:
        - vectorfield (function): Map from R^2 -> R^2
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Coordinates
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    x = torch.linspace(x_min, x_max, num_arrows)
    y = torch.linspace(y_min, y_max, num_arrows)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    grid_points = torch.stack([X.flatten(), Y.flatten()], dim=-1)
    
    # Vector Field
    with torch.no_grad():
        vectors = vectorfield(grid_points)
    
    # Reshape to match grid
    U = vectors[:, 0].reshape(X.shape).numpy()
    V = vectors[:, 1].reshape(Y.shape).numpy()

    # Plot
    ax.quiver(X.numpy(), Y.numpy(), U, V)


def visualize_score_vectorfield(density: Density,
                                x_bounds: Tuple[float, float], 
                                y_bounds: Tuple[float, float],
                                num_arrows: int,
                                bins: int,
                                device: torch.device,
                                ax: Optional[Axes] = None,
                                **kwargs):
    """
    Visualizes the score vector field of a 2D distribution.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    ### Visualization
    # Score Vector Field
    visualize_vectorfield(vectorfield=density.score, 
                          x_bounds=x_bounds, 
                          y_bounds=y_bounds, 
                          num_arrows=num_arrows,
                          ax=ax)

    # Density Scalar Field
    visualize_density(density=density, 
                      x_bounds=x_bounds, 
                      y_bounds=y_bounds, 
                      bins=bins, 
                      device=device, 
                      ax=ax,
                      **kwargs)

