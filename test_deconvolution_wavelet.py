import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import logging

# Optional: Adjust path if running from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.deconvolution.methods.wavelet import WaveletDeconvolver
from test_cases import TEST_CASES

# Map test case names to .npz files
FILE_MAP = {
    "Two clearly separated peaks": "TEST_CASE.npz",
    "Strong overlap between two peaks": "TEST_CASE_1.npz",
    "Four peaks: 2 overlap, 2 isolated": "TEST_CASE_2.npz",
    "Cluster of 3 overlapping peaks": "TEST_CASE_3.npz",
    "Close but not overlapping peaks": "TEST_CASE_4.npz",
    "Intense + weak overlap": "TEST_CASE_5.npz",
    "Five peaks: 3 spaced, 2 overlapping": "TEST_CASE_6.npz",
}

DATA_FOLDER = "test_splatted_smooth"  # or change to "test_splatted" if needed

logging.basicConfig(level=logging.INFO)

def load_test_data(label):
    fname = FILE_MAP.get(label)
    if not fname:
        raise ValueError(f"No .npz file mapping found for test case: '{label}'")
    path = os.path.join(DATA_FOLDER, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    data = np.load(path)
    return data["grid"], data["mz_axis"], data["rt_axis"]

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

def run_wavelet_only_tests():
    for label, *_ in TEST_CASES:
        logging.info(f"\n=== Wavelet Test: {label} ===")
        try:
            grid, mz_axis, rt_axis = load_test_data(label)
            wd = WaveletDeconvolver(min_intensity=1e4, scale_range=(1, 15), overlap_threshold=2)
            result = wd.fit(grid, mz_axis, rt_axis)

            logging.info(f"Overlap detected: {result['overlap_detected']}")
            logging.info(f"Number of peaks in overlap: {result['num_peaks_in_overlap']}")
            logging.info(f"Total peaks detected: {len(result['peaks'])}")

            plot_wavelet_result(grid, result["transformed_grid"], result["peaks"], result["clusters"], title=label)

        except Exception as e:
            logging.error(f"Test failed for '{label}': {e}")

if __name__ == "__main__":
    run_wavelet_only_tests()
