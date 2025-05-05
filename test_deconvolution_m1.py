import numpy as np
import sys
import os

# Add parent folder (mass_spec_project) to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.deconvolution.peak_deconvolver import PeakDeconvolver
from src.deconvolution.visualization import plot_residual_heatmap

data = np.load("data/suspicious_peaks/suspicious_peaks.npz", allow_pickle=True)
grids, mz_axes, rt_axes = data["grids"], data["mz_axes"], data["rt_axes"]

deconvolver = PeakDeconvolver(method="gmm")

for i, (grid, mz, rt) in enumerate(zip(grids, mz_axes, rt_axes)):
    print(f"\nRegion {i}")
    result = deconvolver.fit(grid, mz, rt, region_index=i)
    plot_residual_heatmap(grid, mz, rt, result["gmm"], result["scaler"], i)
