import numpy as np
import matplotlib.pyplot as plt

def find_next_active_box(grid, processed_mask, mz_axis, rt_axis,
                         global_intensity_thresh=0.01, local_margin=2):
    """
    Find the next active (non-processed, strong) region and grow a box around it.

    Parameters:
        grid (2D array): Smoothed intensity grid (m/z × RT)
        processed_mask (2D array): Boolean mask indicating already processed areas
        mz_axis (1D array): m/z values corresponding to grid rows
        rt_axis (1D array): RT values corresponding to grid columns
        global_intensity_thresh (float): Global intensity cutoff for "active" signal
        local_margin (int): Margin to pad around detected active region

    Returns:
        box (dict): {'mz_min', 'mz_max', 'rt_min', 'rt_max'} or None if no region found
    """
    global_max = np.max(grid)
    cutoff = global_max * global_intensity_thresh

    active_mask = (grid > cutoff) & (processed_mask == 'unprocessed')

    mz_bins, rt_bins = active_mask.shape
    found = np.argwhere(active_mask)

    if found.size == 0:
        return None

    start_mz_idx, start_rt_idx = found[0]
    mz_min_idx, mz_max_idx, rt_min_idx, rt_max_idx = grow_box_from_start(
        active_mask, start_mz_idx, start_rt_idx
    )

    mz_min_idx = max(mz_min_idx - local_margin, 0)
    mz_max_idx = min(mz_max_idx + local_margin, mz_bins - 1)
    rt_min_idx = max(rt_min_idx - local_margin, 0)
    rt_max_idx = min(rt_max_idx + local_margin, rt_bins - 1)

    box = {
        'mz_min': mz_axis[mz_min_idx],
        'mz_max': mz_axis[mz_max_idx],
        'rt_min': rt_axis[rt_min_idx],
        'rt_max': rt_axis[rt_max_idx],
        'mz_min_idx': mz_min_idx,
        'mz_max_idx': mz_max_idx,
        'rt_min_idx': rt_min_idx,
        'rt_max_idx': rt_max_idx
    }

    return box


def grow_box_from_start(active_mask, start_mz_idx, start_rt_idx):
    """
    Expand a bounding box from a starting active pixel using simple flood-fill.
    """
    mz_bins, rt_bins = active_mask.shape
    visited = np.zeros_like(active_mask, dtype=bool)
    to_visit = [(start_mz_idx, start_rt_idx)]

    mz_min_idx = mz_max_idx = start_mz_idx
    rt_min_idx = rt_max_idx = start_rt_idx

    while to_visit:
        mz_idx, rt_idx = to_visit.pop()

        if (mz_idx < 0 or mz_idx >= mz_bins or
            rt_idx < 0 or rt_idx >= rt_bins or
            visited[mz_idx, rt_idx] or
            not active_mask[mz_idx, rt_idx]):
            continue

        visited[mz_idx, rt_idx] = True

        mz_min_idx = min(mz_min_idx, mz_idx)
        mz_max_idx = max(mz_max_idx, mz_idx)
        rt_min_idx = min(rt_min_idx, rt_idx)
        rt_max_idx = max(rt_max_idx, rt_idx)

        neighbors = [
            (mz_idx - 1, rt_idx), (mz_idx + 1, rt_idx),
            (mz_idx, rt_idx - 1), (mz_idx, rt_idx + 1)
        ]
        to_visit.extend(neighbors)

    return mz_min_idx, mz_max_idx, rt_min_idx, rt_max_idx


def crop_box(grid, mz_axis, rt_axis, box):
    """
    Extract a cropped version of the grid and axes inside a given box.

    Returns:
        cropped_grid, cropped_mz_axis, cropped_rt_axis
    """
    mz_slice = slice(box['mz_min_idx'], box['mz_max_idx'] + 1)
    rt_slice = slice(box['rt_min_idx'], box['rt_max_idx'] + 1)

    cropped_grid = grid[mz_slice, rt_slice]
    cropped_mz_axis = mz_axis[mz_slice]
    cropped_rt_axis = rt_axis[rt_slice]

    return cropped_grid, cropped_mz_axis, cropped_rt_axis


def plot_box_on_grid(grid, box, mz_axis, rt_axis, title="Detected Box on Intensity Grid"):
    """
    Plot the full grid and overlay the detected box region.

    Parameters:
        grid (2D array): Full intensity grid.
        box (dict): Dictionary with 'mz_min_idx', 'mz_max_idx', 'rt_min_idx', 'rt_max_idx'.
        mz_axis (1D array): m/z axis values.
        rt_axis (1D array): Retention time axis values.
        title (str): Plot title.
    """

    fig, ax = plt.subplots(figsize=(10, 6))
    extent = [rt_axis[0], rt_axis[-1], mz_axis[0], mz_axis[-1]]

    img = ax.imshow(grid,
                    extent=extent,
                    aspect='auto',
                    origin='lower',
                    cmap='viridis')

    mz_min = mz_axis[box['mz_min_idx']]
    mz_max = mz_axis[box['mz_max_idx']]
    rt_min = rt_axis[box['rt_min_idx']]
    rt_max = rt_axis[box['rt_max_idx']]

    rect = plt.Rectangle((rt_min, mz_min), rt_max - rt_min, mz_max - mz_min,
                         linewidth=2, edgecolor='red', facecolor='none', label='Detected Box')
    ax.add_patch(rect)

    plt.colorbar(img, ax=ax, label='Intensity')
    ax.set_xlabel('Retention Time (RT)')
    ax.set_ylabel('m/z')
    ax.set_title(title)
    ax.legend()
    plt.grid(False)
    plt.show()
