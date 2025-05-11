import numpy as np
from scipy.signal import cwt, morlet2
from sklearn.cluster import DBSCAN

class WaveletDeconvolver:
    def __init__(self, min_intensity=5e4, scale_range=(1, 10), overlap_threshold=2, wavelet_func=None):
        self.min_intensity = min_intensity  # Increased minimum intensity threshold
        self.scale_range = scale_range
        self.overlap_threshold = overlap_threshold  # Increased to require at least 2 peaks
        # Use morlet2 instead of morlet (which is deprecated)
        self.wavelet_func = wavelet_func if wavelet_func else morlet2

    def fit(self, grid, mz_axis=None, rt_axis=None):
        """
        Apply the CWT (Continuous Wavelet Transform) to the intensity grid.
        """
        # Get grid dimensions
        mz_len, rt_len = grid.shape
        
        # Create default axes if not provided
        if mz_axis is None:
            mz_axis = np.arange(mz_len)
        if rt_axis is None:
            rt_axis = np.arange(rt_len)
            
        # Apply 1D CWT along the RT axis for each m/z value
        scales = np.arange(*self.scale_range)
        transformed_grid = np.zeros((len(scales), mz_len, rt_len))
        
        for i in range(mz_len):
            # Apply CWT to each m/z slice (along RT)
            # Take absolute value to handle complex output
            cwt_result = cwt(grid[i, :], self.wavelet_func, scales)
            transformed_grid[:, i, :] = np.abs(cwt_result)
        
        # Collapse scales by taking maximum response at each point
        max_response = np.max(transformed_grid, axis=0)
        
        # Detect peaks in the transformed grid
        peaks = self._detect_peaks(max_response)
        
        # Cluster peaks to handle overlapping peaks
        clusters = self._cluster_peaks(peaks)
        
        # Count unique clusters (excluding noise cluster -1)
        if len(peaks) > 0 and len(clusters) > 0:
            num_clusters = len(set(clusters[clusters >= 0]))
        else:
            num_clusters = 0
            
        # Determine if there's an overlap based on number of clusters
        max_intensity = np.max(grid)
        
        # Higher intensity threshold to avoid detecting overlaps in low-intensity regions
        min_required_intensity = 1e5
        
        # Only detect overlap if intensity is high enough and we have multiple clusters
        # Require at least 2 clusters to consider it an overlap
        overlap_detected = (num_clusters >= 2 and 
                           max_intensity >= min_required_intensity)
        
        # Build result similar to other deconvolvers
        result = {
            "overlap_detected": overlap_detected,
            "num_peaks_in_overlap": num_clusters if overlap_detected else None,
            "transformed_grid": max_response,
            "peaks": peaks,
            "clusters": clusters
        }
        
        return result

    def _detect_peaks(self, transformed_grid):
        """
        Identify local maxima (peaks) from the wavelet-transformed grid.
        """
        peaks = []
        mz_len, rt_len = transformed_grid.shape
        
        # Find local maxima in 2D grid
        for i in range(1, mz_len-1):
            for j in range(1, rt_len-1):
                # Check if point is greater than all 8 neighbors
                neighborhood = transformed_grid[i-1:i+2, j-1:j+2]
                if transformed_grid[i, j] > self.min_intensity and transformed_grid[i, j] == np.max(neighborhood):
                    peaks.append((i, j))
        
        return peaks

    def _cluster_peaks(self, peaks):
        """
        Use DBSCAN to cluster close peaks, resolving overlaps.
        """
        if not peaks:
            return np.array([])
            
        peak_array = np.array(peaks)
        if len(peak_array) > 1:
            # Increased eps to make clustering more aggressive (merge more peaks)
            db = DBSCAN(eps=3.0, min_samples=1).fit(peak_array)
            labels = db.labels_
            return labels
        else:
            # If only one peak, return a single cluster
            return np.array([0])
