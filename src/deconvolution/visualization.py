"""
Visualization tools for GMM and Ridge-based deconvolution methods.
"""
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
import numpy as np


import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

def plot_horizontal_gmm(grid, mz_axis, rt_axis, gmm, scaler, region_idx, mz_shrink):
    """
    Plot GMM ellipses on top of intensity heatmap, undoing scaling and anisotropy.
    
    Args:
        grid (ndarray): 2D intensity grid (m/z x RT).
        mz_axis (ndarray): m/z axis values.
        rt_axis (ndarray): RT axis values.
        gmm (GaussianMixture): Fitted GMM model.
        scaler (StandardScaler): Used to unscale GMM means and covariances.
        region_idx (int): Index of the region for title.
        mz_shrink (float): Anisotropy scaling factor applied to m/z.
    """
    grid = np.array(grid, dtype=float)
    plt.figure(figsize=(8, 6))
    extent = [rt_axis[0], rt_axis[-1], mz_axis[0], mz_axis[-1]]

    plt.imshow(grid, extent=extent, origin='lower', aspect='auto', cmap='viridis')
    plt.title(f"Region {region_idx} – GMM Overlay ({gmm.n_components} component(s))")
    plt.xlabel("RT")
    plt.ylabel("m/z")
    plt.colorbar(label='Intensity')

    for mean, cov in zip(gmm.means_, gmm.covariances_):
        # Undo standardization
        mean_unscaled = scaler.inverse_transform(mean.reshape(1, -1))[0]
        cov_unscaled = cov * (scaler.scale_ ** 2)

        # Undo anisotropy scaling on m/z
        mean_unscaled[0] *= mz_shrink
        cov_unscaled[0, :] *= mz_shrink
        cov_unscaled[:, 0] *= mz_shrink


        # Eigen decomposition for ellipse shape
        vals, vecs = np.linalg.eigh(cov_unscaled)
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]

        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

        # Correct width/height: match to RT (x-axis) and m/z (y-axis)
        width = 2 * np.sqrt(vals[1])  # RT
        height = 2 * np.sqrt(vals[0])  # m/z


        ell = Ellipse(
            xy=(mean_unscaled[1], mean_unscaled[0]),  # (RT, m/z)
            width=width,
            height=height,
            angle=angle,
            edgecolor='red',
            fc='None',
            lw=2
        )
        plt.gca().add_patch(ell)

    plt.tight_layout()
    plt.show()




def plot_residual_heatmap(grid, mz_axis, rt_axis, gmm, scaler, region_idx):
    # Create coordinate grid
    grid = np.array(grid, dtype=float)  # ← Add this line
    mz_coords, rt_coords = np.meshgrid(mz_axis, rt_axis, indexing='ij')
    X_grid = np.column_stack([mz_coords.ravel(), rt_coords.ravel()])


    mz_range = mz_axis.max() - mz_axis.min()
    rt_range = rt_axis.max() - rt_axis.min()
    mz_boost = rt_range / mz_range 

    # Apply same anisotropy as used in model training (IMPORTANT!)
    X_grid_aniso = X_grid.copy()
    X_grid_aniso[:, 0] *= mz_boost
    X_grid_scaled = scaler.transform(X_grid_aniso)

    # Reconstruct intensity from GMM
    recon = np.zeros(X_grid_scaled.shape[0])
    for mean, cov, weight in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        rv = multivariate_normal(mean=mean, cov=cov)
        recon += weight * rv.pdf(X_grid_scaled)

    recon_grid = recon.reshape(grid.shape)

    # Avoid division by zero
    if recon_grid.max() != 0:
        recon_grid *= grid.max() / recon_grid.max()
    else:
        print("Warning: Reconstructed grid is all zeros.")
        recon_grid[:] = 0  # Just to be explicit

    residual = grid - recon_grid

    # Plot
    plt.figure(figsize=(8, 6))
    extent = [rt_axis[0], rt_axis[-1], mz_axis[0], mz_axis[-1]]
    plt.imshow(residual, extent=extent, origin='lower', aspect='auto', cmap='coolwarm')
    plt.title(f"Region {region_idx} – Residual Heatmap")
    plt.xlabel("RT")
    plt.ylabel("m/z")
    plt.colorbar(label='Residual (Actual - GMM)')
    plt.tight_layout()
    plt.show()


def plot_ridges_on_grid(grid, mz_axis, rt_axis, ridges):
    grid = np.array(grid, dtype=float)
    plt.figure(figsize=(8, 6))
    extent = [rt_axis[0], rt_axis[-1], mz_axis[0], mz_axis[-1]]
    plt.imshow(grid, extent=extent, origin='lower', aspect='auto', cmap='viridis')
    plt.title("Ridge Overlay")
    plt.xlabel("RT")
    plt.ylabel("m/z")
    plt.colorbar(label='Intensity')

    for ridge in ridges:
        mz_indices, rt_indices = zip(*ridge)
        mz_values = mz_axis[list(mz_indices)]
        rt_values = rt_axis[list(rt_indices)]
        plt.plot(rt_values, mz_values, color='red', linewidth=1)

    plt.show()