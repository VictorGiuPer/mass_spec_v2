import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

class GMMDeconvolver:
    def __init__(self, min_intensity=5e4, max_components=4, saturation=20.0):
        """
        Args:
            min_intensity (float): Minimum intensity threshold to consider a point.
            max_components (int): Max number of components for GMM fitting.
            saturation (float): Controls confidence normalization (BIC delta scaling).
        """
        self.min_intensity = min_intensity
        self.max_components = max_components
        self.saturation = saturation
        self.results = []

    def fit(self, grid, mz_axis, rt_axis, region_index=0, plot_func=None):
        """
        Main entry point to fit GMM to a subregion.
        """
        self.grid = np.array(grid, dtype=float)
        self.mz_axis = np.array(mz_axis, dtype=float)
        self.rt_axis = np.array(rt_axis, dtype=float)
        self.region_index = region_index

        self._prepare_coordinates()
        self._filter_points()

        if self.X_filtered.shape[0] < 10:
            print(f"Region {region_index}: Too few points to fit GMM.")
            return None

        self._apply_anisotropy_scaling()
        self._scale_features()
        self._fit_gmm_models()

        result = self._build_result()
        self.results.append(result)

        if plot_func:
            plot_func(self.grid, self.mz_axis, self.rt_axis, self.best_gmm,
                      self.scaler, region_index, self.mz_boost)

        return result

    def _prepare_coordinates(self):
        # Convert grid to a flat list of (mz, rt) coordinates with corresponding intensities
        mz_coords, rt_coords = np.meshgrid(self.mz_axis, self.rt_axis, indexing='ij')
        self.X = np.column_stack([mz_coords.ravel(), rt_coords.ravel()])
        self.intensity = self.grid.ravel()

    def _filter_points(self):
        # Filter points above intensity threshold
        mask = self.intensity > self.min_intensity
        self.X_filtered = self.X[mask]
        self.intensity_filtered = self.intensity[mask]

    def _apply_anisotropy_scaling(self):
        # Scale mz axis to compensate for anisotropy
        mz_step = np.mean(np.diff(self.mz_axis))
        rt_step = np.mean(np.diff(self.rt_axis))
        self.mz_boost = (rt_step / mz_step) * 1.5  # Optional boost factor

        self.X_aniso = self.X_filtered.copy()
        self.X_aniso[:, 0] *= self.mz_boost  # Scale m/z dimension only

    def _scale_features(self):
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X_aniso)

    def _fit_gmm_models(self):
        self.gmms = []
        self.bics = []

        # Fit GMMs for k = 1 to max_components
        for k in range(1, self.max_components + 1):
            gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=0)
            gmm.means_init = self._smart_initialization(k)
            gmm.fit(self.X_scaled)
            self.gmms.append(gmm)
            self.bics.append(gmm.bic(self.X_scaled))

        # Choose best model by BIC
        self.best_k = np.argmin(self.bics) + 1
        self.best_gmm = self.gmms[self.best_k - 1]
        self.confidence = self.bics[0] - self.bics[self.best_k - 1]
        self.bic_label = self._bic_support_label(self.confidence)
        self.confidence_pct_raw = min(1.0, self.confidence / self.saturation) * 100

        if self.best_k > 1:
            self.separation_score = self._cluster_separation_score(self.best_gmm)
            self.confidence_pct_adjusted = self.confidence_pct_raw * self.separation_score
            self._validate_multicomponent_model()
        else:
            self.separation_score = None
            self.confidence_pct_adjusted = self.confidence_pct_raw
            print(f"[GMM] Single-component model selected (ΔBIC={self.confidence:.2f})")

    def _validate_multicomponent_model(self):
        # Apply heuristics to reject bad multi-component models
        weights = self.best_gmm.weights_
        means = self.scaler.inverse_transform(self.best_gmm.means_)
        dists = [np.linalg.norm(means[i] - means[j]) for i in range(len(means)) for j in range(i + 1, len(means))]

        reasons = []
        if self.confidence < 4:
            reasons.append("low_BIC")
        if self.separation_score is not None and self.separation_score < 0.5:
            reasons.append("low_sep")
        if weights.min() < 0.05:
            reasons.append("low_weight")
        if dists and min(dists) < 0.25:
            reasons.append("close_centers")

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

    def _build_result(self):
        overlap_detected = self.best_k > 1
        num_overlap_events = self.best_k - 1 if overlap_detected else 0
        num_peaks_in_overlap = self.best_k if overlap_detected else None

        return {
            # Core evaluation metrics
            "region_index": self.region_index,
            "overlap_detected": overlap_detected,
            "num_overlap_events": num_overlap_events,
            "num_peaks_in_overlap": num_peaks_in_overlap,

            # Diagnostics
            "confidence": self.confidence,
            "bic_support": self.bic_label,
            "separation_score": self.separation_score,
            "confidence_percent_raw": self.confidence_pct_raw,
            "confidence_percent_adjusted": self.confidence_pct_adjusted,
            "bic_scores": self.bics,
            "mz_boost": self.mz_boost,

            # Raw GMM data
            "means": self.scaler.inverse_transform(self.best_gmm.means_),
            "covariances": self.best_gmm.covariances_,
            "weights": self.best_gmm.weights_,
            "gmm": self.best_gmm,
            "scaler": self.scaler,
        }


    def _bic_support_label(self, delta_bic):
        if delta_bic < 2:
            return "weak"
        elif delta_bic < 6:
            return "positive"
        elif delta_bic < 10:
            return "strong"
        else:
            return "very strong"

    def _cluster_separation_score(self, gmm):
        # Measures Mahalanobis distance between component means
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

    def _smart_initialization(self, k):
        # KMeans++-like initialization weighted by intensity
        if k == 1:
            max_idx = np.argmax(self.intensity_filtered)
            return np.array([self.X_scaled[max_idx]])

        selected = []
        max_idx = np.argmax(self.intensity_filtered)
        selected.append(max_idx)

        while len(selected) < k:
            distances = []
            for i in range(len(self.X_scaled)):
                if i in selected:
                    continue
                min_dist = min(np.linalg.norm(self.X_scaled[i] - self.X_scaled[j]) for j in selected)
                intensity_weight = self.intensity_filtered[i] / np.max(self.intensity_filtered)
                distances.append((i, min_dist * intensity_weight))

            if not distances:
                selected.append(selected[-1])
                continue

            next_idx = max(distances, key=lambda x: x[1])[0]
            selected.append(next_idx)

        return np.array([self.X_scaled[i] for i in selected])
