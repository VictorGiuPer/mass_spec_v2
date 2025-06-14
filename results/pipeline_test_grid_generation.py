import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scipy.ndimage import gaussian_filter
from src.generation import GridGenerator, GaussianGenerator
from src.generation.splatting import splatting_pipeline, splatted_grid_to_npy

# Create directory if it doesn't exist
os.makedirs("test_ext", exist_ok=True)

# Test cases from test_cases.py
from results.pipeline_test_cases import TEST_CASES

# File mapping from test_pipeline.py
FILE_MAP = {
    "Two clearly separated peaks": "TEST_CASE.npz",
    "Strong overlap between two peaks": "TEST_CASE_1.npz",
    "Four peaks: 2 overlap, 2 isolated": "TEST_CASE_2.npz",
    "Cluster of 3 overlapping peaks": "TEST_CASE_3.npz",
    "Close but not overlapping peaks": "TEST_CASE_4.npz",
    "Intense + weak overlap": "TEST_CASE_5.npz",
    "Five peaks: 3 spaced, 2 overlapping": "TEST_CASE_6.npz",
}

# Initialize generators
grid_gen = GridGenerator()
gauss_gen = GaussianGenerator()

# Generate grid files for each test case
for label, peak_params, _, _, _ in TEST_CASES:
    print(f"Generating grid for: {label}")
    
    # Generate base grid
    df = grid_gen.generate_grid(
        rt_start=10, rt_end=15, rt_steps=100, rt_variation=0.1,
        mz_start=150, mz_end=160, mz_min_steps=990, mz_max_steps=1010, mz_variation=0.1
    )
    
    # Apply Gaussian peaks
    df2 = gauss_gen.generate_gaussians(grid=df, peak_params=peak_params, noise_std=0.3)
    
    # Convert to dense grid
    grid, rt_axis, mz_axis = splatting_pipeline(
        gaussians_df=df2,
        rt_range=(10, 15),
        mz_range=(150, 160),
        rt_points=100,
        mz_points=1000,
        plot=False,
    )

    # Apply 2D smoothing
    grid = gaussian_filter(grid, sigma=(0.6, 1.2))

    # Clean label to use as filename
    safe_label = label.lower().replace(" ", "_").replace(":", "").replace("-", "").replace("__", "_")
    filename = f"{safe_label}.npz"

    # Save to new test_ext folder
    output_path = os.path.join("test_ext", filename)
    # np.savez(output_path, grid=grid, rt_axis=rt_axis, mz_axis=mz_axis)
    print(f"Saved to: {output_path}")

