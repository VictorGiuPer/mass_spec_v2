import sys
import os
import pandas as pd

# === Setup: Adjust Import Paths ===
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# === Imports ===
from src.data_generation import GridGenerator, GaussianGenerator
from src.data_generation.export_utils import zoom_grid, gaussians_grid_to_json, gaussians_grid_to_mzml
from src.data_generation.plotting import plot_grid, plot_gaussians_grid
from src.data_generation.splatting import splatting_pipeline, splatted_grid_to_npy

""" # === Step 1: Generate Empty Grid Structure ===
grid_gen = GridGenerator()
df = grid_gen.generate_grid(
    rt_start=10, rt_end=15, rt_steps=100, rt_variation=0.1,
    mz_start=150, mz_end=160, mz_min_steps=990, mz_max_steps=1010, mz_variation=0.1
)

# === Step 2: Define Gaussian Peak Parameters ===
peak_params = [
    {"rt_center": 13.0, "mz_center": 154.5, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 18000},
    {"rt_center": 13.35, "mz_center": 154.4, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 45000},
]

# === Step 3: Generate Gaussian Peaks on the Grid ===
gauss_gen = GaussianGenerator()
df2 = gauss_gen.generate_gaussians(grid=df, peak_params=peak_params, noise_std=0.3)

# === Step 4: Zoom Grid (Remove low-signal values) ===
df_zoom = zoom_grid(df2)

# === Step 5: Quick Inspection of Zoomed Grid ===
print(df_zoom.head())

# === Step 6: Plot Grid (optional visualization functions) ===
plot_grid(df_zoom)
plot_gaussians_grid(df_zoom)

# === Step 7: Optional Export to JSON or mzML Formats ===
gaussians_grid_to_json(df_zoom)
gaussians_grid_to_mzml(df_zoom)
gaussians_grid_to_json(df2)
gaussians_grid_to_mzml(df2)
 """
# === Step 8: Load JSON Version for Splatting ===
gaussians_grid = pd.read_json(r"C:\Users\victo\VSCode Folder\mass_spec_project\data\json\gaussians_grid.json")

# === Step 9: Convert Sparse Gaussians to Dense Grid ===
grid, rt_axis, mz_axis = splatting_pipeline(
    gaussians_df=gaussians_grid,
    rt_range=(10, 15),
    mz_range=(150, 160),
    rt_points=100,
    mz_points=1000,
    plot=True
)

splatted_grid_to_npy(grid=grid)