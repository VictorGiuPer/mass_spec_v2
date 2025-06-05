import numpy as np
import sys
import os

# Add parent folder (mass_spec_project) to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.deconvolution.peak_deconvolver import PeakDeconvolver
from src.deconvolution.visualization import plot_ridges_on_grid

# Load the data
data = np.load("data/suspicious_peaks/suspicious_peaks.npz", allow_pickle=True)
grids = data["grids"]
mz_axes = data["mz_axes"]
rt_axes = data["rt_axes"]
ds_rt = data["d_rt"]
dds_rt = data["dd_rt"]


grid_labels = []

# Initialize the deconvolver with the 'ridge_walk' method
deconvolver = PeakDeconvolver(method="ridge_walk")

# Iterate over the regions
for i, (grid, mz_axis, rt_axis, d_rt, dd_rt) in enumerate(zip(grids, mz_axes, rt_axes, ds_rt, dds_rt)):
    print(f"\nRegion {i}")
    result = deconvolver.model.fit(grid, d_rt=d_rt, dd_rt=dd_rt)
    if result is not None:
        plot_ridges_on_grid(grid, mz_axis, rt_axis, deconvolver.model.ridges)
        print(f"Fusion Score: {result[i-1]['fusion_score']:.3f}")
        label = "overlap" if result[i-1]['fusion_score'] >= 0.5 else "single"
        print(label)
        grid_labels.append(label)