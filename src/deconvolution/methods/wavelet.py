import numpy as np
from scipy.signal import cwt, morlet2
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

class WaveletDeconvolver:
    def __init__(self, min_intensity=5e4, scale_range=(1, 5), overlap_threshold=2, wavelet_func=None, pad_grid=False):
        self.min_intensity = min_intensity
        self.scale_range = scale_range
        self.overlap_threshold = overlap_threshold
        self.wavelet_func = wavelet_func if wavelet_func else morlet2
        self.pad_grid = pad_grid

    def fit(self, grid, mz_axis=None, rt_axis=None):
        mz_len, rt_len = grid.shape

        if mz_axis is None:
            mz_axis = np.arange(mz_len)
        if rt_axis is None:
            rt_axis = np.arange(rt_len)

        pad_size = int(3 * max(self.scale_range)) if self.pad_grid else 0

        if self.pad_grid:
            grid, rt_axis, rt_offset = self._pad_rt_axis_and_grid(grid, rt_axis, pad=pad_size)
        else:
            rt_offset = 0

        scales = np.arange(*self.scale_range)
        transformed_grid = np.zeros((len(scales), mz_len, grid.shape[1]))

        for i in range(mz_len):
            cwt_result = cwt(grid[i, :], self.wavelet_func, scales)
            transformed_grid[:, i, :] = np.abs(cwt_result)

        max_response = np.max(transformed_grid, axis=0)

        if self.pad_grid:
            max_response = max_response[:, rt_offset:-rt_offset]

        peaks = self._detect_peaks(max_response)
        clusters = self._cluster_peaks(peaks)

        num_clusters = len(set(clusters[clusters >= 0])) if len(peaks) > 0 and len(clusters) > 0 else 0

        cropped_grid = grid if not self.pad_grid else grid[:, rt_offset:-rt_offset]
        max_intensity = np.max(cropped_grid)
        min_required_intensity = 1e5

        overlap_detected = (num_clusters >= 2 and max_intensity >= min_required_intensity)

        return {
            "overlap_detected": overlap_detected,
            "num_peaks_in_overlap": num_clusters if overlap_detected else None,
            "transformed_grid": max_response,
            "peaks": peaks,
            "clusters": clusters
        }

    def _pad_rt_axis_and_grid(self, grid, rt_axis, pad=10):
        padded_grid = np.pad(grid, pad_width=((0, 0), (pad, pad)), mode='reflect')

        rt_step = rt_axis[1] - rt_axis[0]
        left_pad = rt_axis[0] - np.arange(pad, 0, -1) * rt_step
        right_pad = rt_axis[-1] + np.arange(1, pad + 1) * rt_step
        padded_rt_axis = np.concatenate([left_pad, rt_axis, right_pad])

        return padded_grid, padded_rt_axis, pad

    def _detect_peaks(self, transformed_grid):
        peaks = []
        mz_len, rt_len = transformed_grid.shape

        for i in range(1, mz_len - 1):
            for j in range(1, rt_len - 1):
                neighborhood = transformed_grid[i - 1:i + 2, j - 1:j + 2]
                if transformed_grid[i, j] > self.min_intensity and transformed_grid[i, j] == np.max(neighborhood):
                    peaks.append((i, j))

        return peaks

    def _cluster_peaks(self, peaks):
        if not peaks:
            return np.array([])

        peak_array = np.array(peaks)
        if len(peak_array) > 1:
            db = DBSCAN(eps=3.0, min_samples=1).fit(peak_array)
            return db.labels_
        else:
            return np.array([0])
        
    @staticmethod
    def plot_wavelet_result(grid, transformed, peaks, clusters, title=""):
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))

        axs[0].imshow(grid, origin="lower", aspect="auto")
        axs[0].set_title("Original Intensity Grid")
        axs[0].set_xlabel("RT axis")
        axs[0].set_ylabel("MZ axis")

        axs[1].imshow(transformed, origin="lower", aspect="auto")
        axs[1].set_title("Wavelet Transformed + Detected Peaks")
        axs[1].set_xlabel("RT axis")
        axs[1].set_ylabel("MZ axis")

        for idx, (mz_idx, rt_idx) in enumerate(peaks):
            color = "red" if clusters[idx] >= 0 else "gray"
            axs[1].plot(rt_idx, mz_idx, "o", color=color, markersize=5)

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()