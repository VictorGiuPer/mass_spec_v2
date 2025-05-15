import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import maximum_filter, label, find_objects


class GMMDeconvolver:
    """
    Gaussian Mixture Model-based peak deconvolution for 2D intensity grids.

    Attributes:
        min_intensity (float): Intensity threshold for considering points.
        max_components (int): Maximum number of GMM components to try.
        saturation (float): Confidence scaling for BIC delta.
    """
    def __init__(self, min_intensity=2e4, max_components=3, saturation=20.0):
        """
        Initialize the deconvolver with optional thresholds and settings.

        Args:
            min_intensity (float): Minimum intensity threshold to consider a point.
            max_components (int): Maximum number of components for GMM fitting.
            saturation (float): Scaling factor for confidence normalization.
        """
        self.min_intensity = min_intensity
        self.max_components = max_components
        self.saturation = saturation
        self.results = []

    def fit(self, grid, mz_axis, rt_axis, region_index=0, plot_func=None):
        """
        Apply GMM deconvolution to a 2D intensity grid.

        Args:
            grid (ndarray): 2D intensity grid.
            mz_axis (ndarray): m/z axis corresponding to the grid rows.
            rt_axis (ndarray): Retention time axis corresponding to columns.
            region_index (int): Identifier for this region (for logging).
            plot_func (callable): Optional visualization callback.

        Returns:
            dict: A dictionary containing deconvolution results and metadata.
        """
        self.grid = np.array(grid, dtype=float)
        self.mz_axis = np.array(mz_axis, dtype=float)
        self.rt_axis = np.array(rt_axis, dtype=float)
        self.region_index = region_index
        # self.min_intensity = np.percentile(self.grid, 95) * 0.5

        self._prepare_coordinates()
        self._filter_points()

        
        # === Not enough data to even fit ===
        if self.X_filtered.shape[0] < 10:
            return {
                "region_index": region_index,
                "overlap_detected": False,
                "num_peaks_in_overlap": None,
                "means": [],
                "detection_reason": "too_few_points"
            }

        try:
            self._apply_anisotropy_scaling()
            self._scale_features()
            self._fit_gmm_models()

            result = self._build_result()
            self.results.append(result)

            if plot_func:
                plot_func(self.grid, self.mz_axis, self.rt_axis, self.best_gmm,
                        self.scaler, region_index, self.mz_shrink)


            return result

        except Exception as e:
            print(f"[GMM] Exception during fit: {e}")

            return {
                "region_index": region_index,
                "overlap_detected": False,
                "num_peaks_in_overlap": None,
                "means": [],
                "detection_reason": f"fit_failed: {str(e)}"
            }


# DATA PREPARATION
    def _prepare_coordinates(self):
        """Convert the input grid into coordinate + intensity format (flattened)."""
        mz_coords, rt_coords = np.meshgrid(self.mz_axis, self.rt_axis, indexing='ij')
        self.X = np.column_stack([mz_coords.ravel(), rt_coords.ravel()])
        self.intensity = self.grid.ravel()

    def _filter_points(self):
        """Filter out points below the intensity threshold."""
        mask = self.intensity > self.min_intensity
        self.X_filtered = self.X[mask]
        self.intensity_filtered = self.intensity[mask]

    def _apply_anisotropy_scaling(self):
        """Scale the m/z axis relative to retention time to normalize anisotropy."""
        # Compute step size in each dimension
        mz_step = np.mean(np.diff(self.mz_axis))
        rt_step = np.mean(np.diff(self.rt_axis))

        # Compute and store shrink factor for m/z
        self.mz_shrink = (mz_step / rt_step) * 1.5

        # Apply shrinkage to m/z dimension
        self.X_aniso = self.X_filtered.copy()
        self.X_aniso[:, 0] /= self.mz_shrink

    def _scale_features(self):
        """Apply z-score normalization to the filtered and scaled input data."""
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X_aniso)


# MODEL FITTING
    def _fit_gmm_models(self):
        """Fit GMMs with 1 to max_components and select the best model based on adjusted BIC."""
        self.model_diagnostics = {}  # New dictionary for diagnostics
        self.gmms = []
        self.bics = []

        # Fit GMMs for k = 1 to max_components
        for k in range(1, self.max_components + 1):
            try:
                gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=0)
                init_means = self._smart_initialization(k)
                gmm.means_init = init_means
                gmm.fit(self.X_scaled)

                bic = gmm.bic(self.X_scaled)
                penalty = self._coverage_penalty(gmm)
                bic += 2 * penalty  # or tune this multiplier

                self.gmms.append(gmm)
                self.bics.append(bic)

                self.model_diagnostics[k] = {
                    "gmm": gmm,
                    "bic": bic,
                    "init_means": init_means,
                    "converged": gmm.converged_,
                    "n_iter": gmm.n_iter_
                }

            except Exception as e:
                self.gmms.append(None)
                self.bics.append(np.inf)
                self.model_diagnostics[k] = {
                    "gmm": None,
                    "bic": np.inf,
                    "init_means": None,
                    "error": str(e)
                }

        # Choose best model by BIC (ignoring failed fits)
        valid_bics = [bic if gmm is not None else np.inf for gmm, bic in zip(self.gmms, self.bics)]
        self.best_k = int(np.argmin(valid_bics)) + 1
        self.best_gmm = self.gmms[self.best_k - 1]
        self._align_means_to_local_maxima(self.best_gmm)
        self.confidence = self.bics[0] - self.bics[self.best_k - 1]
        self.bic_label = self._bic_support_label(self.confidence)
        self.confidence_pct_raw = min(1.0, self.confidence / self.saturation) * 100

        # Override if 2 or more strong local maxima exist and best_k == 1
        if self.best_k == 1 and getattr(self, "strong_peak_count", 0) >= 2:
            print(f"[GMM] Overriding best_k=1 due to {self.strong_peak_count} strong local maxima.")
            self.best_k = 2
            self.best_gmm = self.gmms[1]  # index 1 = 2 components
            self._align_means_to_local_maxima(self.best_gmm)
            self.confidence = self.bics[0] - self.bics[1]
            self.bic_label = self._bic_support_label(self.confidence)
            self.confidence_pct_raw = min(1.0, self.confidence / self.saturation) * 100
            
        # Suppress oversplitting if BIC curve is flat
        if self.best_k > 1:
            bic_deltas = np.diff(self.bics[:self.best_k])

        if self.best_k > 1:
            self.separation_score = self._cluster_separation_score(self.best_gmm)
            self.confidence_pct_adjusted = self.confidence_pct_raw * self.separation_score
            self._validate_multicomponent_model()
        else:
            self.separation_score = None
            self.confidence_pct_adjusted = self.confidence_pct_raw
            print(f"[GMM] Single-component model selected (ΔBIC={self.confidence:.2f})")

    def _smart_initialization(self, k):
        """
        Estimate GMM initial means based on top local maxima.

        Args:
            k (int): Number of components to initialize.

        Returns:
            ndarray: Initial mean estimates in scaled feature space.
        """
        # Reshape to 2D grid
        intensity_2d = self.grid.copy()

        # === Step 1: Find local maxima ===
        neighborhood_size = 3
        local_max = (maximum_filter(intensity_2d, size=neighborhood_size) == intensity_2d)
        labeled, _ = label(local_max)
        slices = find_objects(labeled)

        peaks = []
        for dy, dx in slices:
            if dy is None or dx is None:
                continue
            i = (dy.start + dy.stop - 1) // 2
            j = (dx.start + dx.stop - 1) // 2
            intensity = intensity_2d[i, j]
            mz = self.mz_axis[i]
            rt = self.rt_axis[j]
            peaks.append(((mz, rt), intensity))

        # === Step 2: Sort and pick top-k peaks ===
        peaks.sort(key=lambda x: -x[1])  # descending intensity
        top_peaks = peaks[:k]

        # === Step 3: Transform to scaled coordinate system ===
        coords = np.array([[mz / self.mz_shrink, rt] for (mz, rt), _ in top_peaks])
        X_scaled = self.scaler.transform(coords)

        self.strong_peak_count = len([p for p in top_peaks if p[1] > self.min_intensity * 1.5])

        return X_scaled


# POST FIT VALIDATION
    def _validate_multicomponent_model(self):
        """Reject poor multi-peak fits using BIC, separation, and spatial heuristics."""
        weights = self.best_gmm.weights_
        means = self.scaler.inverse_transform(self.best_gmm.means_)
        dists = [np.linalg.norm(means[i] - means[j]) for i in range(len(means)) for j in range(i + 1, len(means))]

        reasons = []
        if self.confidence < 3:
            reasons.append("low_BIC")
        if weights.min() < 0.01:
            reasons.append("low_weight")


        # Special case: allow overlapping peaks if they are balanced and moderately close
        if len(means) == 2 and len(dists) == 1:
            dist = dists[0]
            if 0.25 <= dist <= 0.5:
                weight_ratio = min(weights) / max(weights)
                if weight_ratio > 0.8:
                    print(f"[GMM] Special case: overlapping peaks at dist={dist:.2f}, balanced weights")
                    reasons.clear()

        if reasons:
            print(f"[GMM] Rejected best_k={self.best_k} due to: {', '.join(reasons)}")
            self.best_k = 1
            self.best_gmm = self.gmms[0]
            self.bic_label = "rejected_" + "_".join(reasons)
            self.confidence_pct_adjusted = 0
            self.separation_score = None
        else:
            print(f"[GMM] Accepted best_k={self.best_k} (ΔBIC={self.confidence:.2f}, support={self.bic_label}, sep={self.separation_score:.2f})")

    def _align_means_to_local_maxima(self, gmm):
        """
        Optionally snap GMM means to nearby local maxima if Mahalanobis-close.
        """
        if not hasattr(self, "local_maxima_scaled"):
            return  # Skip if local maxima are not defined

        new_means = gmm.means_.copy()
        for i, mean in enumerate(gmm.means_):
            best_pt = None
            best_dist = float("inf")

            for pt in self.local_maxima_scaled:
                diff = pt - mean
                try:
                    inv_cov = np.linalg.inv(gmm.covariances_[i])
                    dist = np.sqrt(diff.T @ inv_cov @ diff)
                except np.linalg.LinAlgError:
                    continue

                if dist < best_dist:
                    best_dist = dist
                    best_pt = pt

            if best_dist < 1.5:  # Only snap if Mahalanobis distance is close
                print(f"[GMM] Snapping mean {i} to nearby local max (Mahalanobis dist={best_dist:.2f})")
                new_means[i] = best_pt

        gmm.means_ = new_means
    
    def _cluster_separation_score(self, gmm):
        """
        Compute separation between GMM clusters using Mahalanobis distance.

        Returns:
            float: Separation score ∈ [0, 1].
        """
        if gmm.n_components < 2:
            return 0

        min_dist = float("inf")
        for i in range(len(gmm.means_)):
            for j in range(i + 1, len(gmm.means_)):
                mu_i, mu_j = gmm.means_[i], gmm.means_[j]
                cov_avg = (gmm.covariances_[i] + gmm.covariances_[j]) / 2

                try:
                    inv_cov = np.linalg.inv(cov_avg)
                    diff = mu_i - mu_j
                    dist = np.sqrt(diff.T @ inv_cov @ diff)
                    min_dist = min(min_dist, dist)
                except np.linalg.LinAlgError:
                    continue

        return min(1.0, min_dist / 3.0) if min_dist < float("inf") else 0

    def _bic_support_label(self, delta_bic):
            if delta_bic < 2:
                return "weak"
            elif delta_bic < 6:
                return "positive"
            elif delta_bic < 10:
                return "strong"
            else:
                return "very strong"


# DIAGNOSTICS AND UTILITIES
    def _coverage_penalty(self, gmm):
        """
        Apply penalty based on component overlap with low-intensity areas.

        Args:
            gmm (GaussianMixture): GMM to evaluate.

        Returns:
            float: Penalty value to adjust BIC.
        """
        from matplotlib.patches import Ellipse

        penalty = 0
        mz_coords, rt_coords = np.meshgrid(self.mz_axis, self.rt_axis, indexing='ij')
        coords = np.column_stack([mz_coords.ravel(), rt_coords.ravel()])
        intensities = self.grid.ravel()

        for i, (mean, cov) in enumerate(zip(gmm.means_, gmm.covariances_)):
            # Scale back to unshrunken space
            mean_unscaled = self.scaler.inverse_transform([mean])[0]
            cov_unscaled = self._unscale_covariance(cov)

            # Ellipse radius ≈ 1 standard deviation (~68% coverage)
            vals, vecs = np.linalg.eigh(cov_unscaled)
            angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            width, height = 2 * np.sqrt(vals)

            ell = Ellipse(xy=mean_unscaled, width=width, height=height, angle=angle)
            inside = np.array([ell.contains_point(xy) for xy in coords])
            if inside.any():
                avg_intensity = np.mean(intensities[inside])
                penalty += max(0, 1.0 - (avg_intensity / np.max(self.grid)))  # lower avg = higher penalty
            else:
                penalty += 1.0  # nothing inside = full penalty

        return penalty

    def _unscale_covariance(self, cov_scaled):
        """
        Undo feature scaling and axis shrinkage on a covariance matrix.

        Args:
            cov_scaled (ndarray): Covariance matrix in scaled space.

        Returns:
            ndarray: Covariance matrix in original data space.
        """
        scale_factors = self.scaler.scale_
        cov_unscaled = cov_scaled * np.outer(scale_factors, scale_factors)
        cov_unscaled[0, :] *= self.mz_shrink
        cov_unscaled[:, 0] *= self.mz_shrink
        return cov_unscaled

    def _build_result(self):
        """Assemble output dictionary for the selected GMM result."""
        overlap_detected = self.best_k > 1
        num_overlap_events = self.best_k - 1 if overlap_detected else 0
        num_peaks_in_overlap = self.best_k if overlap_detected else None

        # Unscale means back to original coordinate space
        unscaled_means = self.scaler.inverse_transform(self.best_gmm.means_)
        unscaled_means[:, 0] *= self.mz_shrink

        return {
            # === Core Classification Outputs ===
            "region_index": self.region_index,
            "overlap_detected": overlap_detected,                     # ✅ Category 1
            "num_overlap_events": num_overlap_events,                 # ✅ Category 2
            "num_peaks_in_overlap": num_peaks_in_overlap,             # ✅ Category 2
            "peak_locations": unscaled_means,                         # ✅ Category 3

            # === Diagnostics for Trust / Interpretation ===
            "confidence": self.confidence,                            # raw ΔBIC
            "bic_support": self.bic_label,                            # e.g. "strong"
            "separation_score": self.separation_score,                # Mahalanobis
            "confidence_percent_raw": self.confidence_pct_raw,
            "confidence_percent_adjusted": self.confidence_pct_adjusted,
            "bic_scores": self.bics,

            # === Model Internals ===
            "mz_shrink": self.mz_shrink,
            "weights": self.best_gmm.weights_,
            "covariances": self.best_gmm.covariances_,
            "gmm": self.best_gmm,
            "scaler": self.scaler,

            # Optional tagging for external logic
            "model_type": "GMM"
        }

