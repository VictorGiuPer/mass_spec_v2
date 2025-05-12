import numpy as np
from scipy.signal import cwt, morlet2
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from skimage.feature import blob_log

print(blob_log)

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

        # Compute wavelet response
        scales = np.geomspace(self.scale_range[0],
                              self.scale_range[1], num=30)
        response_map = self._wavelet_transform_2d(grid, scales)

        # Trim padding if applied
        if self.pad_grid:
            response_map = response_map[:, rt_offset:-rt_offset]

        # Peak detection
        peaks = self._detect_blobs(response_map)

        # Optional: determine if overlap likely based on number of peaks
        num_peaks = len(peaks)
        overlap_detected = num_peaks >= self.overlap_threshold

        # Dummy clusters for compatibility with plot function
        clusters = np.arange(num_peaks)

        return {
            "overlap_detected": overlap_detected,
            "num_peaks_in_overlap": num_peaks if overlap_detected else None,
            "transformed_grid": response_map,
            "peaks": peaks,
            "clusters": clusters
        }



    def _wavelet_transform_2d(self, grid, scales):
        mz_len, rt_len = grid.shape
        wmz = np.zeros((len(scales), mz_len, rt_len))
        wrt = np.zeros((len(scales), mz_len, rt_len))

        # Apply 1D CWT along RT (axis 1)
        for i in range(mz_len):
            wrt[:, i, :] = np.abs(cwt(grid[i, :], self.wavelet_func, scales))

        # Apply 1D CWT along MZ (axis 0)
        for j in range(rt_len):
            wmz[:, :, j] = np.abs(cwt(grid[:, j], self.wavelet_func, scales))

        combined = (wrt + wmz) / 2  # Or np.sqrt(wrt * wmz)
        return np.max(combined, axis=0)  # 2D response map
    
    def _detect_blobs(self, response_map):
        blobs = blob_log(response_map, min_sigma=1, max_sigma=5, threshold=0.02)
        peaks = [(int(b[0]), int(b[1])) for b in blobs]
        return peaks


    def _pad_rt_axis_and_grid(self, grid, rt_axis, pad=10):
        padded_grid = np.pad(grid, pad_width=((0, 0), (pad, pad)), mode='reflect')

        rt_step = rt_axis[1] - rt_axis[0]
        left_pad = rt_axis[0] - np.arange(pad, 0, -1) * rt_step
        right_pad = rt_axis[-1] + np.arange(1, pad + 1) * rt_step
        padded_rt_axis = np.concatenate([left_pad, rt_axis, right_pad])

        return padded_grid, padded_rt_axis, pad

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