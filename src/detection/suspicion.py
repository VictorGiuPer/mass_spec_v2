import numpy as np
import matplotlib.pyplot as plt
import os

class SuspicionDetector:
    def __init__(self, cropped_grid, cropped_mz_axis):
        """
        Initialize the detector with a cropped intensity region.

        Parameters:
            cropped_grid (2D np.array): Intensity sub-grid (m/z × RT)
            cropped_mz_axis (1D np.array): Corresponding m/z values for rows
        """
        self.grid = cropped_grid
        self.mz_axis = cropped_mz_axis
        self.peak_widths = self.estimate_peak_widths()
        _, self.d_grid_rt, _, self.dd_grid_rt = self.compute_derivatives()

    def estimate_peak_widths(self, relative_threshold=0.05):
        widths = np.zeros(self.grid.shape[1])
        global_max = np.max(self.grid)
        cutoff = global_max * relative_threshold

        for j in range(self.grid.shape[1]):
            column = self.grid[:, j]
            indices_above_cutoff = np.where(column >= cutoff)[0]
            if len(indices_above_cutoff) > 1:
                width = self.mz_axis[indices_above_cutoff[-1]] - self.mz_axis[indices_above_cutoff[0]]
                widths[j] = width
            else:
                widths[j] = 0
        return widths

    def check_width_growth(self, threshold=0.2):
        width_diff = np.diff(self.peak_widths)
        growth_mask = width_diff > (threshold * np.maximum(self.peak_widths[:-1], 1e-6))
        return np.concatenate([growth_mask, [False]])

    def check_slope_anomaly(self, threshold=0.5):
        d_rt_1d = np.max(np.abs(self.d_grid_rt), axis=0)
        max_slope = np.max(d_rt_1d)
        anomaly_mask = d_rt_1d > (threshold * max_slope)
        return np.concatenate([anomaly_mask, [False]])

    def check_curvature_flatness(self, threshold=0.2):
        dd_rt_1d = np.max(np.abs(self.dd_grid_rt), axis=0)
        max_curvature = np.max(dd_rt_1d)
        flat_mask_raw = dd_rt_1d < (threshold * max_curvature)

        width_gate = self.peak_widths[:-2]  # align
        flat_mask = flat_mask_raw[:len(width_gate)] & (width_gate > 0.01)
        return np.concatenate([flat_mask, [False, False]])

    def group_suspicious_zones(self, suspicious_mask, min_zone_size=3, merge_gap=2):
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

    def detect_suspicious(self, plot=True, min_zone_size=3, merge_gap=2):
        width_growth = self.check_width_growth()
        slope_anomaly = self.check_slope_anomaly()
        curvature_flat = self.check_curvature_flatness()
        suspicious_mask = width_growth | slope_anomaly | curvature_flat
        suspicious = np.any(suspicious_mask)

        if plot:
            self.plot_suspicious_signals(width_growth, slope_anomaly, curvature_flat, suspicious_mask)
            zones = self.group_suspicious_zones(suspicious_mask, min_zone_size, merge_gap)
            self.plot_zones_over_width(zones)

        return suspicious

    def compute_derivatives(self):
        d_grid_mz = np.diff(self.grid, axis=0)
        d_grid_rt = np.diff(self.grid, axis=1)
        dd_grid_mz = np.diff(d_grid_mz, axis=0)
        dd_grid_rt = np.diff(d_grid_rt, axis=1)
        return d_grid_mz, d_grid_rt, dd_grid_mz, dd_grid_rt

    def plot_zones_over_width(self, zones, title="Suspicious Zones over Width Curve"):
        plt.figure(figsize=(14, 6))
        plt.plot(self.peak_widths, label='Estimated Peak Width', color='black')
        for (start, end) in zones:
            plt.axvspan(start, end, color='red', alpha=0.3)
        plt.xlabel('RT Index')
        plt.ylabel('Width (m/z units)')
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_suspicious_signals(self, width_growth, slope_anomaly, curvature_flat, suspicious_mask):
        fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
        axes[0].plot(self.peak_widths, label='Width')
        axes[0].plot(np.where(width_growth, self.peak_widths, np.nan), 'ro', label='Width Growth')
        axes[0].legend()
        axes[0].set_title('Width Growth Check')

        axes[1].plot(self.peak_widths, label='Width')
        axes[1].plot(np.where(slope_anomaly, self.peak_widths, np.nan), 'go', label='Slope Anomalies')
        axes[1].legend()
        axes[1].set_title('Slope Anomaly Check')

        axes[2].plot(self.peak_widths, label='Width')
        axes[2].plot(np.where(curvature_flat, self.peak_widths, np.nan), 'bo', label='Curvature Flattening')
        axes[2].legend()
        axes[2].set_title('Curvature Flattening Check')

        axes[3].plot(self.peak_widths, label='Width')
        axes[3].plot(np.where(suspicious_mask, self.peak_widths, np.nan), 'mo', label='Suspicious Points')
        axes[3].legend()
        axes[3].set_title('Combined Suspicious Points')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def mark_box(processed_mask, box, label):
        mz_slice = slice(box['mz_min_idx'], box['mz_max_idx'] + 1)
        rt_slice = slice(box['rt_min_idx'], box['rt_max_idx'] + 1)
        processed_mask[mz_slice, rt_slice] = label

    @staticmethod
    def save_to_npz(cropped_grids, mz_axes, rt_axes, base_filename="suspicious_peaks"):
        output_folder = "data/suspicious_regions"
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
        print(f"Suspicious regions saved to: {filename}")