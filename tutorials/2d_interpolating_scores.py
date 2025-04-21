import os
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)

# Regular Imports
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import matplotlib.pyplot as plt
import imageio
import glob
from tqdm import tqdm

from inference.distribution import Gaussian, GaussianMixture, InterpolatedDensity
import utils.vis2d.visualization


class LinearInterpolatedDensity(InterpolatedDensity):

    def __init__(self, density0, density1):
        super().__init__(density0, density1)
    
    def interpolate(self, log_density0: torch.Tensor, log_density1: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        Constructs interpolation between density0 and density1
        Args:
            - log_density0 (torch.Tensor): Log density of density0 evaluated at x
            - log_density1 (torch.Tensor): Log density of density1 evaluated at x
            - alpha (float): Interpolation parameter (0 <= alpha <= 1)
        """
        return log_density0 + alpha * (log_density1 - log_density0)

if __name__ == "__main__":
    ### Parameters
    # Distribution Setup
    nmodes = 5 # number of gaussians
    std = 1.0
    scale = 20.0
    seed = 42
    

    # Visualization Setup
    save_directory = 'vectorfield_frames'
    x_bounds, y_bounds = (-10, 10), (-10, 150)



    ### Run Code
    # Create Distributions
    dist0 = Gaussian(mean=torch.zeros(2), cov=torch.eye(2)) # Initalize from unit gaussian
    dist1 = GaussianMixture.random_2D(nmodes=nmodes, 
                                    std=std, 
                                    scale=scale, 
                                    seed=seed)
    dist_interpolated = LinearInterpolatedDensity(density0=dist0, density1=dist1) # Interpolated density


    # Create a directory to save frames
    os.makedirs(save_directory, exist_ok=True)

    # Create frames over alpha
    alphas = torch.linspace(0, 1, 60)  # 60 frames
    for i, alpha in tqdm(enumerate(alphas)):

        ### Interpolated
        logprob = lambda x: dist_interpolated.log_density(x=x, alpha=alpha)
        score = lambda x: dist_interpolated.score(x=x, alpha=alpha)

        ### Visualization
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        # Visualize Score vector field
        utils.vis2d.visualization.visualize_vectorfield(
            vectorfield=score,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            num_arrows=30,
            ax=ax
        )

        # Visualize interpolated density
        utils.vis2d.visualization.visualize_scalarfield(
            scalarfield=logprob, 
            x_bounds=x_bounds, 
            y_bounds=y_bounds, 
            bins=200, 
            device=device,
            ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Blues')
        )
        # Save frame
        plt.title(f"Alpha = {alpha:.2f}")
        plt.savefig(f"{save_directory}/frame_{i:03d}.png")
        plt.close(fig)

    # Create GIF
    # Load and sort images
    images = []
    filenames = sorted(glob.glob(f"{save_directory}/frame_*.png"))

    for filename in filenames:
        images.append(imageio.imread(filename))

    # Save as GIF
    imageio.mimsave("interpolated_vectorfield.gif", images, duration=100)
