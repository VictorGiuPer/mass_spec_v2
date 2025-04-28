import numpy as np
import pandas as pd

class GaussianGenerator:
    """
    A class to apply Gaussian peaks and noise to an RT/MZ grid.
    """

    def __init__(self):
        pass

    def generate_gaussians(self, grid: pd.DataFrame, peak_params: list, noise_std=0.1) -> pd.DataFrame:
        intensities = []
        for rt_val, mz_array in zip(grid["rt"], grid["mz"]):
            mz_array = np.array(mz_array)
            raw_intensity = np.zeros_like(mz_array)
            for peak in peak_params:
                peak_intensity = self._gaussian_pdf(rt_val, mz_array,
                                                    peak["rt_center"], peak["mz_center"],
                                                    peak["rt_sigma"], peak["mz_sigma"],
                                                    peak["amplitude"])
                raw_intensity += peak_intensity
            noise = np.random.normal(0, noise_std, size=raw_intensity.shape)
            noisy_intensity = np.clip(raw_intensity + noise, 0, None)
            intensities.append(noisy_intensity.tolist())
        result = grid.copy()
        result["intensities"] = intensities
        return result

    def _gaussian_pdf(self, rt, mz_array, center_rt, center_mz, sigma_rt, sigma_mz, amplitude=1):
        rt_gauss = np.exp(-0.5 * ((rt - center_rt) / sigma_rt) ** 2)
        mz_gauss = np.exp(-0.5 * ((mz_array - center_mz) / sigma_mz) ** 2)
        return amplitude * (1 / (2 * np.pi * sigma_mz * sigma_rt)) * rt_gauss * mz_gauss
