import logging
import numpy as np
import os

from src.generation import GridGenerator, GaussianGenerator
from src.generation.splatting import splatting_pipeline, splatted_grid_to_npy
from src.detection.localization import Localizer
from src.detection.suspicion import SuspicionDetector
from src.detection.utils import mark_box
from src.deconvolution.peak_deconvolver import PeakDeconvolver
from src.deconvolution.visualization import plot_horizontal_gmm, plot_ridges_on_grid
from results.pipeline_test_cases import TEST_CASES

# === Setup Logging ===
logging.basicConfig(
    filename="test_peak_pipeline.txt",
    filemode="w",
    encoding="utf-8",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# === File mapping ===
FILE_MAP = {
    "Variant - Two clearly separated peaks 1": "TEST_CASE.npz",
    "Variant - Strong overlap between two peaks 1": "TEST_CASE_1.npz",
    "Variant - Four peaks: 2 overlap, 2 isolated 1": "TEST_CASE_2.npz",
    "Variant - Cluster of 3 overlapping peaks 1": "TEST_CASE_3.npz",
    "Variant - Close but not overlapping peaks 1": "TEST_CASE_4.npz",
    "Variant - Intense + weak overlap 1": "TEST_CASE_5.npz",
    "Variant - Five peaks: 3 spaced, 2 overlapping ": "TEST_CASE_6.npz",
}

def check_deconv_results(method, found_regions, found_peaks, expected_regions, expected_peaks):
    if found_regions != expected_regions:
        logging.error(f"FAILED: {method} overlap region count mismatch - expected {expected_regions}, got {found_regions}")
    else:
        logging.info(f"PASSED: {method} overlap region count correct.")

    if expected_regions > 0:
        if found_peaks != expected_peaks:
            logging.error(f"FAILED: {method} peak count in overlaps mismatch - expected {expected_peaks}, got {found_peaks}")
        else:
            logging.info(f"PASSED: {method} peak count in overlaps correct.")



def run_test_case(label, peak_params, expected_box_count, expected_overlap_count, expected_peaks_in_overlap, check_overlap=False):
    logging.info(f"\n=== Running Test: {label} ===")
    logging.info(f"Input peak parameters: {peak_params}")
    logging.info(f"Expected: {expected_box_count} suspicious boxes, {expected_overlap_count} overlaps, {expected_peaks_in_overlap} peaks in overlaps")

    def load_splatted_grid(label, smoothed=False):
        filename = FILE_MAP.get(label)
        if not filename:
            raise ValueError(f"No file mapping found for test label: '{label}'")
        folder = "test_splatted_smooth" if smoothed else "test_splatted"
        path = os.path.join(folder, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing precomputed grid file: {path}")
        with np.load(path) as data:
            return data["grid"], data["rt_axis"], data["mz_axis"]

    grid, rt_axis, mz_axis = load_splatted_grid(label, smoothed=True)
    processed_mask = np.full(grid.shape, fill_value='unprocessed', dtype=object)
    lzr = Localizer(grid=grid, mz_axis=mz_axis, rt_axis=rt_axis, processed_mask=processed_mask)

    suspicious_boxes = []
    cropped_data = []

    while 'unprocessed' in processed_mask:
        box = lzr.find_next_active_box(global_intensity_thresh=0.10, local_margin=2)
        if not box:
            break

        cropped_grid, cropped_mz, cropped_rt = lzr.crop_box(box=box)
        sd = SuspicionDetector(cropped_grid=cropped_grid, cropped_mz_axis=cropped_mz, cropped_rt_axis=cropped_rt)
        is_suspicious, _ = sd.detect_suspicious(plot=False)
        mark_box(processed_mask, box, label='processed')

        if is_suspicious:
            suspicious_boxes.append(box)
            cropped_data.append((cropped_grid, cropped_mz, cropped_rt, sd.d_grid_rt, sd.dd_grid_rt))

    suspicious_count = len(suspicious_boxes)
    logging.info(f"Detected boxes: {suspicious_count}")

    """
    if suspicious_count != expected_box_count:
        logging.error(f"FAILED: Box test failed: expected {expected_box_count}, got {suspicious_count}")
    else:
        logging.info("PASSED: Box test passed.")

    if not check_overlap:
        return
    """

    gmm = PeakDeconvolver(method="gmm")
    ridge = PeakDeconvolver(method="ridge_walk")
    wavelet = PeakDeconvolver(method="wavelet", pad_grid=True)

    gmm_confirmed = ridge_confirmed = wavelet_confirmed = 0
    gmm_peaks = ridge_peaks = wavelet_peaks = 0

    for i, (grid, mz, rt, d_rt, dd_rt) in enumerate(cropped_data):
        logging.info(f"\n--- Region {i + 1} ---")

        # === GMM ===
        gmm_result = gmm.model.fit(grid, mz, rt, region_index=i)
        if gmm_result and gmm_result.get("overlap_detected"):
            num_peaks = gmm_result.get("num_peaks_in_overlap", 0)
            gmm_confirmed += 1
            gmm_peaks += num_peaks
            logging.info(f"GMM: Detected {num_peaks} peaks (ΔBIC={gmm_result.get('confidence', 0):.2f}, support={gmm_result.get('bic_support', 'N/A')}) → overlap confirmed")
        else:
            logging.info("GMM: Detected single peak → no overlap")

        # === Ridge ===
        ridge_result = ridge.model.fit(grid, d_rt, dd_rt)
        plot_ridges_on_grid(grid, mz, rt, ridge.model.ridges)
        if ridge_result is None:
            logging.info("Ridge: Analysis failed (no output)")
        elif ridge_result.get("overlap_detected", False):
            num_peaks = ridge_result.get("num_peaks_in_overlap", 0)
            num_events = ridge_result.get("num_overlap_events", 0)
            num_ridges = ridge_result.get("num_ridges_tracked", '?')
            ridge_confirmed += 1
            ridge_peaks += num_peaks
            logging.info(f"Ridge: Detected overlap involving {num_peaks} peaks (from {num_ridges} ridges, {num_events} event{'s' if num_events != 1 else ''})")
        else:
            ridges = ridge_result.get("num_ridges_tracked", 0)
            logging.info(f"Ridge: {ridges} ridges tracked, no overlaps detected")

        # === Wavelet ===
        wavelet_result = wavelet.model.fit(grid, mz_axis, rt_axis)
        # wavelet.model.plot_wavelet_result(grid, wavelet_result["transformed_grid"], wavelet_result["peaks"], wavelet_result["clusters"], title=f"{label} - Region {i + 1}")

        if wavelet_result:
            num_peaks = wavelet_result.get("num_peaks_in_overlap", 0)
            if wavelet_result.get("overlap_detected", False) and num_peaks >= 2:
                wavelet_confirmed += 1
                wavelet_peaks += num_peaks
                logging.info(f"Wavelet: Detected {num_peaks} peaks → overlap confirmed")
            else:
                logging.info(f"Wavelet: Detected {num_peaks} peak(s), not enough for overlap")
        else:
            logging.info("Wavelet: No overlap detected")

    logging.info(f"\n=== Test Summary: {label} ===")

    # === Evaluation Summaries ===
    check_deconv_results("GMM", gmm_confirmed, gmm_peaks, expected_overlap_count, expected_peaks_in_overlap)
    check_deconv_results("Ridge", ridge_confirmed, ridge_peaks, expected_overlap_count, expected_peaks_in_overlap)
    check_deconv_results("Wavelet", wavelet_confirmed, wavelet_peaks, expected_overlap_count, expected_peaks_in_overlap)

def main():
    failures = 0
    if not TEST_CASES:
        logging.warning("No test cases found. Make sure test_cases.py is correctly populated.")

    for i, (label, peak_params, exp_suspicious, exp_overlap, exp_peaks_in_overlap) in enumerate(TEST_CASES, start=1):
        try:
            run_test_case(label, peak_params, exp_suspicious, exp_overlap, exp_peaks_in_overlap, check_overlap=True)
        except Exception as e:
            logging.error(f"Unhandled error during test '{label}': {e}")
            failures += 1

    if failures == 0:
        logging.info("\n=== All tests completed successfully ===")
    else:
        logging.warning(f"\n=== {failures} tests had errors or failed ===")

if __name__ == "__main__":
    main()
