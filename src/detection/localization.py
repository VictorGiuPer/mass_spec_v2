import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


class Localizer:
    def __init__(self, grid, mz_axis, rt_axis, processed_mask=None):
        """
        Initialize the Localizer with a grid and its axes.
        
        Parameters:
            grid (2D np.array): Smoothed intensity grid (m/z × RT)
            mz_axis (1D np.array): m/z values
            rt_axis (1D np.array): RT values
            processed_mask (2D np.array or None): Optional pre-existing mask
        """
        self.grid = grid
        self.mz_axis = mz_axis
        self.rt_axis = rt_axis
        self.processed_mask = processed_mask if processed_mask is not None else \
                              np.full_like(grid, 'unprocessed', dtype=object)

    def find_next_active_box(self, global_intensity_thresh=0.10, local_margin=2):
        """
        Find the next unprocessed high-intensity region and return bounding box info.
        """
        global_max = np.max(self.grid)
        cutoff = global_max * global_intensity_thresh

        active_mask = (self.grid > cutoff) & (self.processed_mask == 'unprocessed')

        mz_bins, rt_bins = active_mask.shape
        found = np.argwhere(active_mask)

        if found.size == 0:
            return None

        start_mz_idx, start_rt_idx = found[0]
        mz_min_idx, mz_max_idx, rt_min_idx, rt_max_idx = self._grow_box_from_start(
            active_mask, start_mz_idx, start_rt_idx, max_gap=1, min_local_drop=1.2
        )

        mz_min_idx = max(mz_min_idx - local_margin, 0)
        mz_max_idx = min(mz_max_idx + local_margin, mz_bins - 1)
        rt_min_idx = max(rt_min_idx - local_margin, 0)
        rt_max_idx = min(rt_max_idx + local_margin, rt_bins - 1)

        box = {
            'mz_min': self.mz_axis[mz_min_idx],
            'mz_max': self.mz_axis[mz_max_idx],
            'rt_min': self.rt_axis[rt_min_idx],
            'rt_max': self.rt_axis[rt_max_idx],
            'mz_min_idx': mz_min_idx,
            'mz_max_idx': mz_max_idx,
            'rt_min_idx': rt_min_idx,
            'rt_max_idx': rt_max_idx
        }

        # print(f"Returning box from seed {start_mz_idx}, {start_rt_idx}")

        return box

    def _grow_box_from_start(self, active_mask, start_mz_idx, start_rt_idx, max_gap=1, min_local_drop=0.3):
        """
        Flood-fill algorithm with stricter separation rules.
        Only grows into neighbors whose intensity hasn't dropped too much from the seed.
        """
        mz_bins, rt_bins = active_mask.shape
        visited = np.zeros_like(active_mask, dtype=bool)
        to_visit = [(start_mz_idx, start_rt_idx)]

        seed_intensity = self.grid[start_mz_idx, start_rt_idx]
        min_allowed_intensity = seed_intensity * (1 - min_local_drop)

        mz_min_idx = mz_max_idx = start_mz_idx
        rt_min_idx = rt_max_idx = start_rt_idx

        while to_visit:
            mz_idx, rt_idx = to_visit.pop()

            if (mz_idx < 0 or mz_idx >= mz_bins or
                rt_idx < 0 or rt_idx >= rt_bins or
                visited[mz_idx, rt_idx] or
                not active_mask[mz_idx, rt_idx]):
                continue

            # Check local drop before visiting
            if self.grid[mz_idx, rt_idx] < min_allowed_intensity:
                continue

            visited[mz_idx, rt_idx] = True

            mz_min_idx = min(mz_min_idx, mz_idx)
            mz_max_idx = max(mz_max_idx, mz_idx)
            rt_min_idx = min(rt_min_idx, rt_idx)
            rt_max_idx = max(rt_max_idx, rt_idx)

            for dmz in range(-max_gap, max_gap + 1):
                for drt in range(-max_gap, max_gap + 1):
                    if dmz == 0 and drt == 0:
                        continue
                    neighbor = (mz_idx + dmz, rt_idx + drt)
                    if (0 <= neighbor[0] < mz_bins) and (0 <= neighbor[1] < rt_bins):
                        to_visit.append(neighbor)

        return mz_min_idx, mz_max_idx, rt_min_idx, rt_max_idx


    def crop_box(self, box):
        """
        Extract the sub-grid and axes defined by the given box.
        """
        mz_slice = slice(box['mz_min_idx'], box['mz_max_idx'] + 1)
        rt_slice = slice(box['rt_min_idx'], box['rt_max_idx'] + 1)

        cropped_grid = self.grid[mz_slice, rt_slice]
        cropped_mz_axis = self.mz_axis[mz_slice]
        cropped_rt_axis = self.rt_axis[rt_slice]

        return cropped_grid, cropped_mz_axis, cropped_rt_axis

    def plot_box_on_grid(self, box, title="Detected Box on Intensity Grid"):
        """
        Visualize the grid with the current box overlaid in red.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        extent = [self.rt_axis[0], self.rt_axis[-1], self.mz_axis[0], self.mz_axis[-1]]

        img = ax.imshow(self.grid,
                        extent=extent,
                        aspect='auto',
                        origin='lower',
                        cmap='viridis')

        mz_min = self.mz_axis[box['mz_min_idx']]
        mz_max = self.mz_axis[box['mz_max_idx']]
        rt_min = self.rt_axis[box['rt_min_idx']]
        rt_max = self.rt_axis[box['rt_max_idx']]

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
