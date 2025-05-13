import torch
from typing import Tuple, List, Union
from tabulate import tabulate
import matplotlib.pyplot as plt

def compare_statistics(cfgs1: torch.Tensor, cfgs2: torch.Tensor):
    """
    Compute the sufficient statistics of the field configurations.
    
    Args:
        cfgs (torch.Tensor): Configuration tensor of shape (n_samples, *lattice_shape)
    """
    # Compute magnetization and susceptibility
    exectations(cfgs1, cfgs2)
    # Compute two-point correlation
    twoPointCorrelation(cfgs1, cfgs2)

def exectations(cfgs1: torch.Tensor, cfgs2: torch.Tensor):
    ### Compute Magnetization and Susceptibility
    # Sample stats from generative model
    M, M_err = get_mag(cfgs1)
    M_abs, M_abs_err = get_abs_mag(cfgs1)
    chi2, chi2_err = get_chi2(cfgs1)

    # Sample stats from target distribution
    M_target, M_target_err = get_mag(cfgs2)
    M_abs_target, M_abs_target_err = get_abs_mag(cfgs2)
    chi2_target, chi2_target_err = get_chi2(cfgs2)

    # Tolerance checks
    in_tol_M = torch.abs(M - M_target) < (M_err + M_target_err)
    in_tol_M_abs = torch.abs(M_abs - M_abs_target) < (M_abs_err + M_abs_target_err)
    in_tol_chi2 = torch.abs(chi2 - chi2_target) < (chi2_err + chi2_target_err)

    # Table data
    headers = ["Statistic", "Generative Model", "Target Distribution", "In Tolerance?"]
    table = [
        ["M", f"{M:.4f} ± {M_err:.4f}", f"{M_target:.4f} ± {M_target_err:.4f}", str(in_tol_M.item())],
        ["|M|", f"{M_abs:.4f} ± {M_abs_err:.4f}", f"{M_abs_target:.4f} ± {M_abs_target_err:.4f}", str(in_tol_M_abs.item())],
        ["chi²", f"{chi2:.4f} ± {chi2_err:.4f}", f"{chi2_target:.4f} ± {chi2_target_err:.4f}", str(in_tol_chi2.item())],
    ]

    print(tabulate(table, headers=headers, tablefmt="pretty"))

def twoPointCorrelation(cfgs1: torch.Tensor, cfgs2: torch.Tensor):
    corr_samples = get_corr_func(cfgs1).cpu().detach().numpy()
    corr_target = get_corr_func(cfgs2).cpu().detach().numpy()

    fig, ax = plt.subplots(1,1, figsize=(6,6))
    plt.xticks([i for i in range(1, 32, 4)])
    ax.errorbar(corr_target[:,0], corr_target[:,1], yerr=corr_target[:,2], label='Target Distribution', color='black')
    ax.errorbar(corr_samples[:,0], corr_samples[:,1], yerr=corr_samples[:,2], label='Generative Model', color='red', linestyle=':' )
    plt.title('2-Point Correlator')
    plt.ylabel(r'$\langle  \phi_x \phi_0 \rangle$')
    plt.xlabel('$x$ (distance)')
    plt.legend()
    plt.show()


def jackknife(samples: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the mean and estimated error using the jackknife resampling method.
    
    Args:
        samples (torch.Tensor): Input tensor of shape (n_samples, ...) where n_samples is the number of configurations
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - mean: The mean value across all samples
            - error: The estimated error using jackknife resampling
    """
    means = []
    
    for i in range(samples.shape[0]):
        # Remove one sample at a time and compute mean
        means.append(torch.cat([samples[:i], samples[i+1:]]).mean(dim=0))
    
    means = torch.stack(means)
    mean = means.mean(dim=0)
    error = torch.sqrt((samples.shape[0] - 1) * torch.mean((means - mean)**2, dim=0))
    
    return mean, error

def get_mag(cfgs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the mean and error of magnetization.
    
    Args:
        cfgs (torch.Tensor): Configuration tensor of shape (n_samples, *lattice_shape)
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - mean: The mean magnetization
            - error: The error in magnetization
    """
    axis = tuple(range(1, len(cfgs.shape)))
    return jackknife(cfgs.mean(dim=axis))

def get_abs_mag(cfgs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the mean and error of absolute magnetization.
    
    Args:
        cfgs (torch.Tensor): Configuration tensor of shape (n_samples, *lattice_shape)
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - mean: The mean absolute magnetization
            - error: The error in absolute magnetization
    """
    axis = tuple(range(1, len(cfgs.shape)))
    return jackknife(torch.abs(cfgs.mean(dim=axis)))

def get_chi2(cfgs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the mean and error of susceptibility.
    
    Args:
        cfgs (torch.Tensor): Configuration tensor of shape (n_samples, *lattice_shape)
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - mean: The mean susceptibility
            - error: The error in susceptibility
    """
    V = torch.prod(torch.tensor(cfgs.shape[1:], device=cfgs.device))
    axis = tuple(range(1, len(cfgs.shape)))
    mags = cfgs.mean(dim=axis)
    return jackknife(V * (mags**2 - mags.mean()**2))

def get_corr_func(cfgs: torch.Tensor) -> torch.Tensor:
    """
    Compute the connected two-point correlation function with errors for symmetric lattices.
    
    Args:
        cfgs (torch.Tensor): Configuration tensor of shape (n_samples, *lattice_shape)
        
    Returns:
        torch.Tensor: A tensor of shape (max_distance, 3) containing:
            - Column 0: Distance
            - Column 1: Mean correlation
            - Column 2: Error in correlation
    """
    mag_sq = torch.mean(cfgs)**2
    corr_func = []
    axis = tuple(range(1, len(cfgs.shape)))
    
    for i in range(1, cfgs.shape[1]):
        corrs = []
        
        for mu in range(len(cfgs.shape)-1):
            # Roll the tensor along the specified dimension
            rolled = torch.roll(cfgs, i, dims=mu+1)
            corrs.append(torch.mean(cfgs * rolled, dim=axis))
        
        corrs = torch.stack(corrs).mean(dim=0)
        corr_mean, corr_err = jackknife(corrs - mag_sq)
        corr_func.append([i, corr_mean, corr_err])
    
    return torch.tensor(corr_func, device=cfgs.device)