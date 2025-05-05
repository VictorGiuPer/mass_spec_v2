from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import numpy as np
from ..utils import delta_bic_to_confidence_score

class GMMDeconvolver:
    def __init__(self, min_intensity=3e4, max_components=2):
        self.min_intensity = min_intensity
        self.max_components = max_components
        self.results = []

    def fit(self, grid, mz_axis, rt_axis, region_index=0, plot_func=None):
        mz_coords, rt_coords = np.meshgrid(mz_axis, rt_axis, indexing='ij')
        X = np.column_stack([mz_coords.ravel(), rt_coords.ravel()])
        intensity = grid.ravel()

        mask = intensity > self.min_intensity
        X_filtered = X[mask]

        if X_filtered.shape[0] < 10:
            print(f"Region {region_index}: Too few points to fit GMM.")
            return None

        # Apply anisotropy
        mz_range = mz_axis.max() - mz_axis.min()
        rt_range = rt_axis.max() - rt_axis.min()
        mz_boost = rt_range / mz_range

        X_aniso = X_filtered.copy()
        X_aniso[:, 0] *= mz_boost
        X_aniso[:, 1] *= 1.0  # RT unchanged

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_aniso)

        # Fit GMM models and compute BIC
        gmms = []
        bics = []
        for k in range(1, self.max_components + 1):
            gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=0)
            gmm.fit(X_scaled)
            gmms.append(gmm)
            bics.append(gmm.bic(X_scaled))

        best_k = np.argmin(bics) + 1
        best_gmm = gmms[best_k - 1]
        confidence = bics[0] - bics[1] if best_k == 2 else 0
        confidence_pct = delta_bic_to_confidence_score(confidence)

        result_dict = {
            "region_index": region_index,
            "best_k": best_k,
            "mz_boost": mz_boost,
            "bic_scores": bics,
            "confidence": confidence,
            "confidence_percent": confidence_pct,
            "means": scaler.inverse_transform(best_gmm.means_),
            "covariances": best_gmm.covariances_,
            "weights": best_gmm.weights_,
            "gmm": best_gmm,
            "scaler": scaler,
        }

        self.results.append(result_dict)

        print(f"Region {region_index}: Best model has {best_k} peak(s). ΔBIC = {confidence:.2f} → Confidence ≈ {confidence_pct:.1f}%")

        if plot_func:
            plot_func(grid, mz_axis, rt_axis, best_gmm, scaler, region_index, mz_boost)

        return result_dict
