import sys
import os

# Add parent folder (mass_spec_project) to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import classes and functions
from src.data_generation import GridGenerator, GaussianGenerator
from src.data_generation.export_utils import zoom_grid, gaussians_grid_to_json, gaussians_grid_to_mzml
from src.data_generation.plotting import plot_grid, plot_gaussians_grid

# Initialize grid and gaussian generators separately
grid_gen = GridGenerator()
gauss_gen = GaussianGenerator()

# Step 1: Generate Grid
df = grid_gen.generate_grid(
    rt_start=10, rt_end=15, rt_steps=100, rt_variation=0.1,
    mz_start=150, mz_end=160, mz_min_steps=990, mz_max_steps=1010, mz_variation=0.1
)

# Step 2: Define Peaks
peak_params = [
    {"rt_center": 13.0, "mz_center": 154.5, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 18000},
    {"rt_center": 13.35, "mz_center": 154.4, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 45000},
]

# Step 3: Generate Gaussians
df2 = gauss_gen.generate_gaussians(grid=df, peak_params=peak_params, noise_std=0.3)

# Step 4: Zoom Grid
df_zoom = zoom_grid(df2)

# Step 5: Print some output
print(df_zoom.head())

# Step 6: Plot results
# plot_grid(df_zoom)
# plot_gaussians_grid(df_zoom)

gaussians_grid_to_json(df_zoom)
gaussians_grid_to_mzml(df_zoom)

gaussians_grid_to_json(df2)
gaussians_grid_to_mzml(df2)
