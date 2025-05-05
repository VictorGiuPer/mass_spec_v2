from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
import numpy as np


def plot_horizontal_gmm(grid, mz_axis, rt_axis, gmm, scaler, region_idx, mz_boost):
    from matplotlib.patches import Ellipse

    grid = np.array(grid, dtype=float)
    plt.figure(figsize=(8, 6))
    extent = [rt_axis[0], rt_axis[-1], mz_axis[0], mz_axis[-1]]
    
    plt.imshow(grid, extent=extent, origin='lower', aspect='auto', cmap='viridis')
    plt.title(f"Region {region_idx} – GMM Overlay ({gmm.n_components} component(s))")
    plt.xlabel("RT")
    plt.ylabel("m/z")
    plt.colorbar(label='Intensity')

    for mean, cov in zip(gmm.means_, gmm.covariances_):
        mean_orig = scaler.inverse_transform(mean.reshape(1, -1))[0]
        mean_orig[0] /= mz_boost  # ← UNDO dynamic anisotropy

        # Ellipse parameters from cov
        v, w = np.linalg.eigh(cov)
        v = 2.0 * np.sqrt(v)  # stddev scaling
        angle = np.degrees(np.arctan2(w[1, 0], w[0, 0]))

        # Undo anisotropy scaling
        v[0] /= mz_boost  # ← UNDO here too

        # Convert to ellipse width/height in original units
        width = v[1] * scaler.scale_[1]  # RT
        height = v[0] * scaler.scale_[0]  # m/z

        ell = Ellipse(
            xy=(mean_orig[1], mean_orig[0]),  # (RT, m/z)
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