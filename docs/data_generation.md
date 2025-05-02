# Data Generation Module Documentation

This module simulates LC-MS data by:

- Generating irregular retention time (RT) and mass-to-charge (m/z) grids,
- Applying synthetic Gaussian peaks,
- Providing plotting and export utilities,
- Saving and loading LC-MS datasets.

Ideal for testing algorithms and models in mass spectrometry workflows.

---

## Module Structure

| File | Purpose |
|:-----|:--------|
| `grid_generator.py` | Generate RT/mz sampling grid with irregular spacing |
| `gaussian_generator.py` | Apply Gaussian peaks and noise to the grid |
| `plotting.py` | Visualize RT/mz scatterplots and intensity heatmaps |
| `export_utils.py` | Save and load grids as JSON or mzML files |

---

## Classes and Functions

### `GridGenerator`

Located in `grid_generator.py`.

```python
from src.data_generation import GridGenerator
```

**Methods:**

`generate_grid(rt_start, rt_end, rt_steps, rt_variation, mz_start, mz_end, mz_min_steps, mz_max_steps, mz_variation) -> pd.DataFrame`

Generates a grid with irregular RT and m/z spacing.

**Returns:**

A DataFrame with:

`rt`: Retention time points

`mz`: List of m/z arrays per RT

### `GaussianGenerator`
Located in `gaussian_generator.py.`

```python
from src.data_generation import GaussianGenerator
```

**Methods:**

`generate_gaussians(grid: pd.DataFrame, peak_params: list, noise_std=0.1) -> pd.DataFrame`

Applies Gaussian peaks and optional noise to the grid.

**Parameters:**

`grid`: Output from `generate_grid`

`peak_params`: List of peak definitions

`noise_std`: Noise standard deviation

**Returns:**

Original DataFrame with an added `intensities` column.

### Plotting Functions
Located in `plotting.py`.

```python
from src.data_generation.plotting import plot_grid, plot_gaussians_grid
plot_grid(df): Scatterplot of RT vs m/z points
```

`plot_gaussians_grid(df, title="Interpolated mz Heatmap", zoom=False, mz_points=1000)`: Heatmap of interpolated 
intensities

**Parameters:**

`df`: DataFrame with `rt`, `mz`, and optionally `intensities`

`title``: Title for the plot

`zoom`: Optional dict `{ 'xlim': (xmin, xmax), 'ylim': (ymin, ymax) }`

`mz_points`: Number of interpolated m/z bins

### Export Functions
Located in `export_utils.py`.

```python
from src.data_generation.export_utils import grid_to_json, gaussians_grid_to_json, gaussians_grid_to_mzml, load_mzml, zoom_grid
```

`grid_to_json(df, base_filename="gen_grid")`: Save grid to JSON

`gaussians_grid_to_json(df, base_filename="gen_gaussians_grid")`: Save intensity data to JSON

`gaussians_grid_to_mzml(df, output_path_base="output.mzML")`: Save data to mzML format

`load_mzml(filepath)`: Load mzML file into a DataFrame

`zoom_grid(df)`: Crop RT and m/z ranges interactively


### Splatting Functions

Located in `splatting.py`.

```python
from src.data_generation.splatting import create_sampling_grid, splat_gaussians_to_grid, splatting_pipeline, plot_grid
```

`create_sampling_grid(rt_range, mz_range, rt_points, mz_points)`: Create a regular RT × m/z grid initialized with zeros

`splat_gaussians_to_grid(gaussians_df, rt_axis, mz_axis, grid)`: Hard-bin Gaussian peak intensities into the grid

`splatting_pipeline(gaussians_df, rt_range, mz_range, rt_points, mz_points, plot)`: Full wrapper to generate and populate the grid

`plot_grid(grid, rt_axis, mz_axis, title)`: Visualize the 2D intensity grid as a heatmap

`save_grid_to_npy(grid, base_filename="splatted_grid")`: Save the 2D NumPy intensity grid as a `.npy` file with auto-incrementing filenames if duplicates exist.


### Example Peak Parameters
```python
peak_params = [
    {"rt_center": 13.0, "mz_center": 154.5, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 18000},
    {"rt_center": 13.35, "mz_center": 154.4, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 45000}
]
```

### Full Example Workflow

```python
from src.data_generation import GridGenerator, GaussianGenerator
from src.data_generation.export_utils import zoom_grid
from src.data_generation.plotting import plot_grid, plot_gaussians_grid

# Initialize generators
grid_gen = GridGenerator()
gauss_gen = GaussianGenerator()

# Generate grid
df = grid_gen.generate_grid(
    rt_start=10, rt_end=15, rt_steps=100, rt_variation=0.1,
    mz_start=150, mz_end=160, mz_min_steps=990, mz_max_steps=1010, mz_variation=0.1
)

# Apply Gaussian peaks
df2 = gauss_gen.generate_gaussians(grid=df, peak_params=peak_params, noise_std=0.3)

# Zoom the grid
df_zoom = zoom_grid(df2)

# Plot the data
plot_grid(df_zoom)
plot_gaussians_grid(df_zoom)

# Save the data
from src.data_generation.export_utils import grid_to_json, gaussians_grid_to_json
grid_to_json(df)
gaussians_grid_to_json(df2)
```