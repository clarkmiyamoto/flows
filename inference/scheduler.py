import torch

def linear(t: torch.tensor):
    """
    Linearly interpolates noise from 1.0 to 0.01 over the range of t.

    Args:
        t: time tensor of shape (batch_size, 1)
    Returns:
        noise: tensor of shape (batch_size, 1)
    """
    t_min = t.min()
    t_max = t.max()
    # Normalize t to [0, 1]
    t_norm = (t - t_min) / (t_max - t_min + 1e-8)
    return 1.0 - 0.99 * t_norm

def sigmoid(t: torch.tensor):
    """
    Applies a sigmoid decay from 1.0 to 0.01 over the range of t.

    Args:
        t: tensor of shape (batch_size, 1)
    Returns:
        noise: tensor of shape (batch_size, 1)
    """
    t_min = t.min()
    t_max = t.max()
    t_norm = (t - t_min) / (t_max - t_min + 1e-8)  # Normalize to [0,1]

    # Sigmoid decay: steeper in the middle, flatter near ends
    sigmoid_scaled = torch.sigmoid(6 * (1 - 2 * t_norm))  # shifted so sigmoid(0)=1, sigmoid(1)=~0
    return 0.01 + 0.99 * sigmoid_scaled  # scale to [0.01, 1.0]

