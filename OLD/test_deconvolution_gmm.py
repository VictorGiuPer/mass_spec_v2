import numpy as np
import sys
import os

# Add parent folder (mass_spec_project) to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.deconvolution.peak_deconvolver import PeakDeconvolver
from src.deconvolution.visualization import plot_residual_heatmap, plot_horizontal_gmm

# Load the preprocessed suspicious peaks
data = np.load("data/suspicious_peaks/suspicious_peaks.npz", allow_pickle=True)
grids = data["grids"]
mz_axes = data["mz_axes"]
rt_axes = data["rt_axes"]

# Initialize the GMM-based deconvolver
deconvolver = PeakDeconvolver(method="gmm")

# Process each region
for i, (grid, mz, rt) in enumerate(zip(grids, mz_axes, rt_axes)):
    result = deconvolver.model.fit(grid, mz, rt, region_index=i)  # uses new modular fit()

    if result:
        plot_horizontal_gmm(grid, mz, rt, result["gmm"], result["scaler"], i, result["mz_boost"])
        plot_residual_heatmap(grid, mz, rt, result["gmm"], result["scaler"], i)
