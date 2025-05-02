
import numpy as np
import matplotlib.pyplot as plt
import os

def estimate_peak_widths(grid, mz_axis, relative_threshold=0.05) -> np.array:
    """
    Estimate peak widths across RT by using a fixed intensity threshold.
    """
    widths = np.zeros(grid.shape[1])
    global_max = np.max(grid)
    cutoff = global_max * relative_threshold

    for j in range(grid.shape[1]):
        column = grid[:, j]
        indices_above_cutoff = np.where(column >= cutoff)[0]

        if len(indices_above_cutoff) > 1:
            width = mz_axis[indices_above_cutoff[-1]] - mz_axis[indices_above_cutoff[0]]
            widths[j] = width
        else:
            widths[j] = 0
    return widths


def check_width_growth(peak_widths, threshold=0.2):
    """
    Detect sudden increases in peak width over adjacent RT indices.
    """
    width_diff = np.diff(peak_widths)
    growth_mask = width_diff > (threshold * np.maximum(peak_widths[:-1], 1e-6))
    return np.concatenate([growth_mask, [False]])


def check_slope_anomaly(d_grid_rt, threshold=0.5):
    """
    Detect unusually steep changes in signal intensity using precomputed 1st derivative (RT axis).
    
    Parameters:
        d_grid_rt (2D np.array): First derivative along RT axis.
        threshold (float): Fraction of the maximum slope to consider anomalous.
        
    Returns:
        np.array: Boolean mask of slope anomalies along RT.
    """
    d_grid_rt_1d = np.max(np.abs(d_grid_rt), axis=0)
    max_slope = np.max(d_grid_rt_1d)
    anomaly_mask = d_grid_rt_1d > (threshold * max_slope)
    return np.concatenate([anomaly_mask, [False]])


def check_curvature_flatness(dd_grid_rt, threshold=0.2, width_gate=None):
    """
    Detect flattening of curvature using precomputed 2nd derivative (RT axis).
    
    Parameters:
        dd_grid_rt (2D np.array): Second derivative along RT axis.
        threshold (float): Fraction of the maximum curvature used as cutoff.
        width_gate (np.array or None): Optional peak width array to suppress flat noise.
        
    Returns:
        np.array: Boolean mask of flat curvature anomalies along RT.
    """
    dd_grid_rt_1d = np.max(np.abs(dd_grid_rt), axis=0)
    max_curvature = np.max(dd_grid_rt_1d)
    flat_mask_raw = dd_grid_rt_1d < (threshold * max_curvature)

    if width_gate is not None:
        width_gate = width_gate[:-2]  # align due to 2 diffs
        flat_mask = flat_mask_raw[:len(width_gate)] & (width_gate > 0.01)
        return np.concatenate([flat_mask, [False, False]])
    else:
        return np.concatenate([flat_mask_raw, [False, False]])


def group_suspicious_zones(suspicious_mask, min_zone_size=3, merge_gap=2):
    """
    Group consecutive suspicious points into zones and optionally merge nearby ones.
    """
    zones = []
    start_idx = None

    for i, val in enumerate(suspicious_mask):
        if val and start_idx is None:
            start_idx = i
        elif not val and start_idx is not None:
            if (i - start_idx) >= min_zone_size:
                zones.append((start_idx, i - 1))
            start_idx = None

    if start_idx is not None and (len(suspicious_mask) - start_idx) >= min_zone_size:
        zones.append((start_idx, len(suspicious_mask) - 1))

    if merge_gap > 0 and len(zones) > 1:
        merged_zones = []
        current_start, current_end = zones[0]

        for next_start, next_end in zones[1:]:
            if next_start - current_end <= merge_gap:
                current_end = next_end
            else:
                merged_zones.append((current_start, current_end))
                current_start, current_end = next_start, next_end

        merged_zones.append((current_start, current_end))
        zones = merged_zones

    return zones


def plot_zones_over_width(peak_widths, zones, title="Suspicious Zones over Width Curve"):
    """
    Plot peak width curve and highlight suspicious zones.
    """
    plt.figure(figsize=(14, 6))
    plt.plot(peak_widths, label='Estimated Peak Width', color='black')

    for (start, end) in zones:
        plt.axvspan(start, end, color='red', alpha=0.3)

    plt.xlabel('RT Index')
    plt.ylabel('Width (m/z units)')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_suspicious_signals(peak_widths, width_growth, slope_anomaly, curvature_flat, suspicious_mask):
    """
    Plot multiple diagnostic subplots for different signal anomaly checks.
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

    axes[0].plot(peak_widths, label='Width')
    axes[0].plot(np.where(width_growth, peak_widths, np.nan), 'ro', label='Width Growth')
    axes[0].legend()
    axes[0].set_title('Width Growth Check')

    axes[1].plot(peak_widths, label='Width')
    axes[1].plot(np.where(slope_anomaly, peak_widths, np.nan), 'go', label='Slope Anomalies')
    axes[1].legend()
    axes[1].set_title('Slope Anomaly Check')

    axes[2].plot(peak_widths, label='Width')
    axes[2].plot(np.where(curvature_flat, peak_widths, np.nan), 'bo', label='Curvature Flattening')
    axes[2].legend()
    axes[2].set_title('Curvature Flattening Check')

    axes[3].plot(peak_widths, label='Width')
    axes[3].plot(np.where(suspicious_mask, peak_widths, np.nan), 'mo', label='Suspicious Points')
    axes[3].legend()
    axes[3].set_title('Combined Suspicious Points')

    plt.tight_layout()
    plt.show()


def compute_derivatives(grid):
    """
    Compute first and second derivatives along m/z and RT dimensions.

    Parameters:
        grid (2D np.array): The input intensity grid.

    Returns:
        Tuple of arrays:
            d_grid_mz, d_grid_rt, dd_grid_mz, dd_grid_rt
    """
    d_grid_mz = np.diff(grid, axis=0)
    d_grid_rt = np.diff(grid, axis=1)
    dd_grid_mz = np.diff(d_grid_mz, axis=0)
    dd_grid_rt = np.diff(d_grid_rt, axis=1)

    return d_grid_mz, d_grid_rt, dd_grid_mz, dd_grid_rt


def detect_suspicious(grid, mz_axis, width_cutoff=0.02, plot=True, min_zone_size=3, merge_gap=2):
    peak_widths = estimate_peak_widths(grid, mz_axis, relative_threshold=width_cutoff)

    # Compute derivatives once
    _, d_grid_rt, _, dd_grid_rt = compute_derivatives(grid)

    width_growth = check_width_growth(peak_widths)
    slope_anomaly = check_slope_anomaly(d_grid_rt)
    curvature_flat = check_curvature_flatness(dd_grid_rt, width_gate=peak_widths)

    suspicious_mask = width_growth | slope_anomaly | curvature_flat
    suspicious = np.any(suspicious_mask)

    if plot:
        plot_suspicious_signals(peak_widths, width_growth, slope_anomaly, curvature_flat, suspicious_mask)
        zones = group_suspicious_zones(suspicious_mask, min_zone_size=min_zone_size, merge_gap=merge_gap)
        plot_zones_over_width(peak_widths, zones)

    return suspicious


def mark_box(processed_mask, box, label):
    """
    Mark a given box inside the processed_mask array with a label.

    Parameters:
        processed_mask (2D array): Mask array to modify (e.g., dtype=object).
        box (dict): Box with 'mz_min_idx', 'mz_max_idx', 'rt_min_idx', 'rt_max_idx'.
        label (str): Label to assign (e.g., 'processed', 'suspicious').
    """
    mz_slice = slice(box['mz_min_idx'], box['mz_max_idx'] + 1)
    rt_slice = slice(box['rt_min_idx'], box['rt_max_idx'] + 1)
    processed_mask[mz_slice, rt_slice] = label

def suspicious_regions_to_npz(cropped_grids, mz_axes, rt_axes, base_filename="suspicious_peaks"):
    """
    Save all suspicious regions (cropped grids and their axes) to a .npz file.

    Parameters:
        cropped_grids (List[np.array]): List of 2D suspicious subgrids.
        mz_axes (List[np.array]): List of m/z axes for each subgrid.
        rt_axes (List[np.array]): List of RT axes for each subgrid.
        base_filename (str): Base name for the saved .npz file.
    """
    output_folder = "data/suspicious_peaks"
    os.makedirs(output_folder, exist_ok=True)

    filename = os.path.join(output_folder, f"{base_filename}.npz")
    counter = 1
    while os.path.exists(filename):
        filename = os.path.join(output_folder, f"{base_filename}_{counter}.npz")
        counter += 1

    np.savez(filename,
             grids=np.array(cropped_grids, dtype=object),
             mz_axes=np.array(mz_axes, dtype=object),
             rt_axes=np.array(rt_axes, dtype=object))

    print(f"Suspicious peaks saved to: {filename}")