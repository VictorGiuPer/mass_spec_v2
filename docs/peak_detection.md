# Peak Detection Module Documentation
This module identifies and extracts signal regions (peaks) from 2D LC-MS intensity grids by:
- Detecting active zones above a global intensity threshold,
- Expanding bounding boxes using a flood-fill strategy,
- Cropping these regions for downstream analysis,
- Visualizing extracted boxes within the full grid.

The core logic is encapsulated in the Localizer class for reusable, object-oriented peak detection workflows.

---

## Module Structure

| File | Purpose |
|:-----|:--------|
| `localization.py` | Encapsulates peak detection and cropping into the Localizer class. |
| `detect_suspicious.py` | Evaluates whether the signal is suspicious based on shape characteristics. |

---

## Classes and Methods
### Localizer
Located in localization.py.

```python
from src.peak_detection.localization import Localizer
```
The Localizer class manages a smoothed LC-MS grid and allows you to iteratively extract active signal regions.

**Constructor**
```python
Localizer(grid, mz_axis, rt_axis, processed_mask=None)
```

**Parameters:**

- `grid (2D np.array)`: Smoothed intensity grid (m/z × RT)
- `mz_axis (1D np.array)`: m/z axis values
- `rt_axis (1D np.array)`: Retention time axis values
- `processed_mask (2D np.array, optional)`: Boolean/string mask for tracking processed regions

---

`find_next_active_box(global_intensity_thresh=0.01, local_margin=2) -> dict or None`

Find the next unprocessed region with signal intensity above a given threshold and return its bounding box.

**Returns:**

A dict containing `mz_min`, `mz_max`, `rt_min`, `rt_max`, and their corresponding index bounds, or None if no region is found.

---

`crop_box(box: dict) -> (cropped_grid, cropped_mz_axis, cropped_rt_axis)`

Extract the portion of the intensity grid and axes within a bounding box.

**Parameters:**

`box (dict)`: Output of find_next_active_box

**Returns:**

`cropped_grid (2D array)`: Subsection of the grid

`cropped_mz_axis (1D array)`: Subsection of m/z axis

`cropped_rt_axis (1D array)`: Subsection of RT axis

---

`plot_box_on_grid(box: dict, title="Detected Box on Intensity Grid")` -> None
Display the full intensity grid with the selected box outlined in red.

**Parameters:**

`box (dict)`: Bounding box to visualize

`title (str)`: Optional title for the plot

`_grow_box_from_start(active_mask, start_mz_idx, start_rt_idx) -> (int, int, int, int)`

Internal method that expands a box from a starting coordinate using flood-fill.

**Returns:**

Tuple of index bounds: `mz_min_idx`, `mz_max_idx`, `rt_min_idx`, `rt_max_idx`

---

**Example Usage**
```python
localizer = Localizer(grid, mz_axis, rt_axis)

while 'unprocessed' in localizer.processed_mask:
    box = localizer.find_next_active_box(global_intensity_thresh=0.001)
    if box is None:
        break

    cropped_grid, cropped_mz, cropped_rt = localizer.crop_box(box)
    localizer.plot_box_on_grid(box)
```

---

### SuspicionDetector
Located in `detect_suspicious.py.`

```python
from src.peak_detection.detect_suspicious import SuspicionDetector
```
The SuspicionDetector class operates on a cropped LC-MS region (from a bounding box) to evaluate whether the signal is suspicious based on its shape characteristics.

**Constructor**
```python
SuspicionDetector(cropped_grid, cropped_mz_axis)
```
**Parameters:**

`cropped_grid (2D np.array)`: Intensity sub-grid (m/z × RT)

`cropped_mz_axis (1D np.array)`: Corresponding m/z values for grid rows

---

**Core Methods**

`detect_suspicious(plot=True, min_zone_size=3, merge_gap=2) -> bool`

Performs the full signal anomaly analysis:
- Evaluates peak width stability
- Detects sharp slope changes
- Flags curvature flattening

**Returns:**

`True` if any suspicious patterns are found, otherwise `False`

If `plot=True`, diagnostic plots are shown.

`estimate_peak_widths(relative_threshold=0.05) -> np.array`

Estimates peak width at each RT index by applying a global intensity threshold across m/z slices.

`check_width_growth(threshold=0.2) -> np.array`

Detects sudden growth in peak width between adjacent `RT` values.

`check_slope_anomaly(threshold=0.5) -> np.array`

Flags regions where the slope of signal intensity across RT changes sharply.

`check_curvature_flatness(threshold=0.2) -> np.array`

Detects areas with flat second-derivative curvature in the RT dimension, often associated with overlapping or distorted peaks.

---

**Zone Grouping**

`group_suspicious_zones(suspicious_mask, min_zone_size=3, merge_gap=2) -> List[Tuple[int, int]]`

Groups consecutive suspicious points into contiguous zones. Optionally merges nearby zones if the gap is small.

**Returns:**

A list of `(start_index, end_index)` tuples representing suspicious time intervals

---

**Derivative Computation**

`compute_derivatives() -> Tuple[np.array, ...]`

Computes:

First derivatives: `d_grid_mz`, `d_grid_rt`

Second derivatives: `dd_grid_mz`, `dd_grid_rt`

These are used for slope and curvature anomaly checks.

---

**Visualization**

`plot_suspicious_signals(width_growth, slope_anomaly, curvature_flat, suspicious_mask)`

Displays four subplots for diagnostic inspection of anomaly sources:

- Peak width growth

- Slope anomaly

- Curvature flatness

- Combined suspicious points


**Overview**
| What is Tested | What It Catches | When It Triggers |
|:-------------|:----------------|:----------------|
| Sudden width expansion | Early peak growth | If width grows too fast between neighboring RT points. |
| Slope anomalies | Top of peak | If slope behavior becomes too sharp or irregular. |
| Curvature flattening | Top and sides of peak | If the peak top becomes too flat (less pointy). |
| Combination | Everything flagged | If any of the three features say "suspicious". |

---

`plot_zones_over_width(zones, title="Suspicious Zones over Width Curve")`

Plots peak width across RT with highlighted suspicious zones in red.

---

**Utility Methods**

`mark_box(processed_mask, box, label)`

Static method to update a processed_mask with a specific label ("suspicious" or "processed"), based on box index bounds.

`save_to_npz(cropped_grids, mz_axes, rt_axes, base_filename="suspicious_peaks")`

Static method to save a set of cropped regions to a single `.npz` file.

Auto-increments the filename if the file already exists

Stores all grids and their respective axes



