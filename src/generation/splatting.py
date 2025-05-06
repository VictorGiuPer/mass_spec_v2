import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.ndimage import gaussian_filter


def create_sampling_grid(rt_range=(10, 15), mz_range=(150, 160), rt_points=100, mz_points=1000):
    """
    Create a regular RT × m/z sampling grid.

    Returns:
        rt_axis (np.array), mz_axis (np.array), grid (2D np.array of zeros)
    """
    rt_axis = np.linspace(rt_range[0], rt_range[1], rt_points)
    mz_axis = np.linspace(mz_range[0], mz_range[1], mz_points)
    grid = np.zeros((mz_points, rt_points))  # shape: (m/z, RT)
    return rt_axis, mz_axis, grid


def splat_gaussians_to_grid(gaussians_df, rt_axis, mz_axis, grid):
    """
    Populate the grid with intensities from Gaussian peak definitions.

    Parameters:
        gaussians_df (pd.DataFrame): Must contain 'rt', 'mz', 'intensities'
        rt_axis (np.array): RT sampling axis
        mz_axis (np.array): m/z sampling axis
        grid (np.array): Output grid to populate (modified in-place)
    """
    for _, row in gaussians_df.iterrows():
        rt = row['rt']
        mz_values = row['mz']
        intensities = row['intensities']

        rt_idx = np.abs(rt_axis - rt).argmin()

        for mz, intensity in zip(mz_values, intensities):
            mz_idx = np.abs(mz_axis - mz).argmin()
            grid[mz_idx, rt_idx] += intensity  # hard binning


def plot_grid(grid, rt_axis, mz_axis, title='Grid Visualization'):
    """
    Show the 2D intensity grid.
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(grid, aspect='auto',
               extent=[rt_axis.min(), rt_axis.max(), mz_axis.min(), mz_axis.max()],
               origin='lower', cmap='viridis')
    plt.xlabel('Retention Time (RT)')
    plt.ylabel('m/z')
    plt.title(title)
    plt.colorbar(label='Intensity')
    plt.show()


def splatting_pipeline(gaussians_df,
                           rt_range=(10, 15), mz_range=(150, 160),
                           rt_points=100, mz_points=1000,
                           plot=True):
    """
    Complete splatting pipeline: grid creation, splatting, and optional plotting.

    Parameters:
        gaussians_df (pd.DataFrame): DataFrame with 'rt', 'mz', 'intensities'
        rt_range (tuple): (min, max) range for RT axis
        mz_range (tuple): (min, max) range for m/z axis
        rt_points (int): Number of RT sampling points
        mz_points (int): Number of m/z sampling points
        plot (bool): Whether to visualize the final grid

    Returns:
        grid (2D np.array): The intensity grid (m/z × RT)
        rt_axis (np.array): RT axis
        mz_axis (np.array): m/z axis
    """
    rt_axis, mz_axis, grid = create_sampling_grid(rt_range, mz_range, rt_points, mz_points)
    splat_gaussians_to_grid(gaussians_df, rt_axis, mz_axis, grid)
    
    if plot:
        plot_grid(grid, rt_axis, mz_axis, title='Splatting Grid (Pipeline Output)')

    return grid, rt_axis, mz_axis


def splatted_grid_to_npy(grid, mz_axis, rt_axis, base_filename="splatted_grid", smoothed=False):
    """
    Save grid and its axes to a .npz file (optionally smoothed).

    Parameters:
        grid (2D np.array): The splatted intensity grid
        mz_axis (1D np.array): m/z axis corresponding to grid rows
        rt_axis (1D np.array): RT axis corresponding to grid columns
        base_filename (str): Base name for the output file
        smoothed (bool): Whether to apply Gaussian smoothing
    """
    if smoothed:
        grid = gaussian_filter(grid, sigma=1)  # small smoothing
        base_filename = "smoothed_" + base_filename

    output_folder = "data/splatted"
    os.makedirs(output_folder, exist_ok=True)

    filename = os.path.join(output_folder, f"{base_filename}.npz")
    counter = 1
    while os.path.exists(filename):
        filename = os.path.join(output_folder, f"{base_filename}_{counter}.npz")
        counter += 1

    np.savez(filename, grid=grid, mz_axis=mz_axis, rt_axis=rt_axis)

    print(f"{'Smoothed ' if smoothed else ''}Grid with axes saved as: {filename}")