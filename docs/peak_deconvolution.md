## `deconvolution` Folder

This module handles peak deconvolution for suspicious regions in mass spectrometry data. It supports multiple methods for detecting overlapping peaks.

---

### `methods` Subfolder

Contains individual deconvolution algorithms.

#### `gmm.py` – Gaussian Mixture Model (GMM)

- **Class**: `GMMDeconvolver`  
- **Purpose**: Uses a 2D Gaussian Mixture Model to identify and separate overlapping peaks in intensity maps.  
- **Key Features**:
  - Automatic selection of the best GMM component count via BIC and heuristics.
  - Scaling and anisotropy correction for proper clustering.
  - Optionally snaps peaks to local maxima for refinement.
- **Inputs**:
  - `grid`: 2D intensity grid.
  - `mz_axis`, `rt_axis`: axes defining the mass/charge and retention time.
- **Outputs**:
  - Dictionary with peak locations, confidence metrics, and overlap flags.

#### `ridge_walking.py` – Ridge Walker

- **Class**: `RidgeWalker`  
- **Purpose**: Tracks ridgelines across the retention time axis to detect peak structures.  
- **Key Features**:
  - Detects and extends ridges based on slope and intensity.
  - Analyzes ridge pairs to detect overlaps using a composite fusion score.
- **Inputs**:
  - `grid`, `d_rt`, `dd_rt`: intensity and derivatives along RT.
- **Outputs**:
  - Peak list, overlap info, ridge metadata.

#### `wavelet.py` – Wavelet Transform

- **Class**: `WaveletDeconvolver`  
- **Purpose**: Applies 2D wavelet analysis to detect peaks and estimate overlap.  
- **Key Features**:
  - Uses continuous wavelet transform (CWT) along both axes.
  - Identifies peaks as local blobs in the wavelet response.
  - Optionally pads input grid to reduce edge effects.
- **Inputs**:
  - `grid`, with optional `mz_axis` and `rt_axis`.
- **Outputs**:
  - Peak list, overlap flag, transformed grid.

---

### `peak_deconvolver.py`

- **Class**: `PeakDeconvolver`  
- **Purpose**: Unified interface for running one of the supported deconvolution methods.  
- **Usage**:
  ```python
  model = PeakDeconvolver(method="gmm").model
  result = model.fit(grid, mz_axis, rt_axis)

Supported Methods: "gmm", "ridge_walk", "wavelet"

### `visualization.py`
Functions:

- **plot_horizontal_gmm**: Plots GMM ellipses over intensity heatmaps.

- **plot_residual_heatmap**: Shows residuals between original and modeled intensities.

- **plot_ridges_on_grid**: Displays ridges tracked by RidgeWalker.

Purpose: Visualize the spatial layout and separation of detected peaks.
