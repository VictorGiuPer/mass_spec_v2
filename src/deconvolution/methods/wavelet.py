import numpy as np
from scipy.signal import cwt, morlet2
import matplotlib.pyplot as plt
from skimage.feature import blob_log

class WaveletDeconvolver:
    # PUBLIC INTERFACE
    def __init__(self, scale_range=(1, 5), overlap_threshold=2, wavelet_func=None, pad_grid=False):
        """
        Initialize the wavelet-based deconvolver.

        Args:
            scale_range (tuple): Range of wavelet scales (small to large).
            overlap_threshold (int): Number of peaks to count as an overlap.
            wavelet_func (callable): Wavelet function (default: morlet2).
            pad_grid (bool): Whether to pad the grid to reduce edge effects.
        """

        self.scale_range = scale_range
        self.overlap_threshold = overlap_threshold
        self.wavelet_func = wavelet_func if wavelet_func else morlet2
        self.pad_grid = pad_grid

    def fit(self, grid, mz_axis=None, rt_axis=None):
        """
        Run the wavelet transform and detect peaks on the given grid.

        Args:
            grid (2D array): Intensity matrix (MZ x RT).
            mz_axis (1D array): MZ values (optional).
            rt_axis (1D array): RT values (optional).

        Returns:
            dict: Deconvolution result with peaks, transformed grid, and flags.
        """
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
            rt_axis = rt_axis[rt_offset:-rt_offset]
        # Peak detection
        peaks = self._detect_blobs(response_map)


        max_response = np.max(response_map)
        if peaks:
            peak_responses = [response_map[int(mz), int(rt)] for mz, rt in peaks]
            mean_peak_response = float(np.mean(peak_responses))
            norm_mean_peak_response = mean_peak_response / (max_response + 1e-8)
        else:
            norm_mean_peak_response = 0.0
        # Optional: determine if overlap likely based on number of peaks
        num_peaks = len(peaks)
        overlap_detected = num_peaks >= self.overlap_threshold

        # Dummy clusters for compatibility with plot function
        clusters = np.arange(num_peaks)

        # Map indices back to actual (mz, rt) values
        peak_locations = [(float(mz_axis[mz_idx]), float(rt_axis[rt_idx])) for mz_idx, rt_idx in peaks]

        return {
            "overlap_detected": overlap_detected,
            "num_peaks_in_overlap": num_peaks if overlap_detected else None,
            "transformed_grid": response_map,
            "peaks": peaks,  # still raw indices
            "clusters": clusters,
            "peak_locations": peak_locations,
            "mean_peak_response": norm_mean_peak_response
        }

    # WAVELET CORE
    def _wavelet_transform_2d(self, grid, scales):
        """
        Compute 2D wavelet response from independent MZ and RT transforms.

        Args:
            grid (2D array): Input intensity grid.
            scales (1D array): Scales for the wavelet transform.

        Returns:
            2D array: Combined wavelet response map.
        """
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
    
    # PREPROCESSING HELPER
    def _pad_rt_axis_and_grid(self, grid, rt_axis, pad=10):
        """
        Pad the RT axis and grid to mitigate boundary artifacts.

        Args:
            grid (2D array): Original intensity grid.
            rt_axis (1D array): Original RT axis.
            pad (int): Number of points to pad on each side.

        Returns:
            tuple: (padded_grid, padded_rt_axis, pad_offset)
        """
        padded_grid = np.pad(grid, pad_width=((0, 0), (pad, pad)), mode='reflect')

        rt_step = rt_axis[1] - rt_axis[0]
        left_pad = rt_axis[0] - np.arange(pad, 0, -1) * rt_step
        right_pad = rt_axis[-1] + np.arange(1, pad + 1) * rt_step
        padded_rt_axis = np.concatenate([left_pad, rt_axis, right_pad])

        return padded_grid, padded_rt_axis, pad
    
    # POST PROCESSING
    def _detect_blobs(self, response_map):
        """
        Detect local blobs/peaks in the wavelet-transformed grid.

        Args:
            response_map (2D array): Wavelet response map.

        Returns:
            list[(int, int)]: List of peak (mz_idx, rt_idx) coordinates.
        """
        blobs = blob_log(response_map, min_sigma=1, max_sigma=5, threshold=0.02)
        peaks = [(int(b[0]), int(b[1])) for b in blobs]
        return peaks

    # VISUALIZATION
    @staticmethod
    def plot_wavelet_result(transformed, peaks, clusters=None, title=""):
        """
        Plot the wavelet-transformed result with peak markers.

        Args:
            transformed (2D array): Wavelet response map.
            peaks (list): Peak coordinates (mz_idx, rt_idx).
            clusters (list or None): Cluster labels or dummy indices (optional).
            title (str): Plot title.
        """
        plt.figure(figsize=(8, 6))
        plt.imshow(transformed, origin="lower", aspect="auto")
        plt.title("Wavelet Transformed + Detected Peaks")
        plt.xlabel("RT axis")
        plt.ylabel("MZ axis")

        if peaks:
            for idx, (mz_idx, rt_idx) in enumerate(peaks):
                color = "red"
                if clusters is not None and len(clusters) > idx and clusters[idx] >= 0:
                    color = "red"
                elif clusters is not None:
                    color = "gray"
                plt.plot(rt_idx, mz_idx, "o", color=color, markersize=5)
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
