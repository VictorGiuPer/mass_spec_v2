import logging
import numpy as np
import os

from src.generation import GridGenerator, GaussianGenerator
from src.generation.splatting import splatting_pipeline, splatted_grid_to_npy
from src.detection.localization import Localizer
from src.detection.suspicion import SuspicionDetector
from src.detection.utils import mark_box
from src.deconvolution.peak_deconvolver import PeakDeconvolver
from src.deconvolution.visualization import plot_ridges_on_grid, plot_horizontal_gmm

# === Setup Logging ===
logging.basicConfig(
    filename="test_peak_pipeline.txt",  # or .log or just leave blank
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

from test_cases import TEST_CASES

if not TEST_CASES:
    logging.warning("No test cases found. Make sure test_cases.py is correctly populated.")

FILE_MAP = {
    "Two clearly separated peaks": "TEST_CASE.npz",
    "Strong overlap between two peaks": "TEST_CASE_1.npz",
    "Four peaks: 2 overlap, 2 isolated": "TEST_CASE_2.npz",
    "Cluster of 3 overlapping peaks": "TEST_CASE_3.npz",
    "Close but not overlapping peaks": "TEST_CASE_4.npz",
    "Intense + weak overlap": "TEST_CASE_5.npz",
    "Five peaks: 3 spaced, 2 overlapping": "TEST_CASE_6.npz",
}

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

    box_count = []
    suspicious_boxes = []
    cropped_grids, cropped_mz_axes, cropped_rt_axes, d_rt_grids, dd_rt_grids = [], [], [], [], []

    while 'unprocessed' in processed_mask:
        box = lzr.find_next_active_box(global_intensity_thresh=0.10, local_margin=2)
        if not box:
            break

        box_count.append(box)
        cropped_grid, cropped_mz, cropped_rt = lzr.crop_box(box=box)
        sd = SuspicionDetector(cropped_grid=cropped_grid, cropped_mz_axis=cropped_mz, cropped_rt_axis=cropped_rt)
        is_suspicious, _ = sd.detect_suspicious(plot=False)
        mark_box(processed_mask, box, label='processed')

        if is_suspicious:
            suspicious_boxes.append(box)
            cropped_grids.append(cropped_grid)
            cropped_mz_axes.append(cropped_mz)
            cropped_rt_axes.append(cropped_rt)
            d_rt_grids.append(sd.d_grid_rt)
            dd_rt_grids.append(sd.dd_grid_rt)

    suspicious_count = len(suspicious_boxes)
    logging.info(f"Detected boxes: {suspicious_count}")

    if suspicious_count != expected_box_count:
        logging.error(f"FAILED: Box test failed: expected {expected_box_count}, got {suspicious_count}")
    else:
        logging.info(f"PASSED: Box test passed.")

    if not check_overlap:
        return
    ########################################################
    gmm_deconvolver = PeakDeconvolver(method="gmm")
    ridge_deconvolver = PeakDeconvolver(method="ridge_walk")
    wavelet_deconvolver = PeakDeconvolver(method="wavelet")

    confirmed_overlaps_gmm = 0
    total_peaks_in_overlap_gmm = 0

    confirmed_overlaps_ridge = 0
    total_peaks_in_overlap_ridge = 0

    confirmed_overlaps_wavelet = 0
    total_peaks_in_overlap_wavelet = 0
    ########################################################

    for i, (grid, mz, rt, d_rt, dd_rt) in enumerate(zip(cropped_grids, cropped_mz_axes, cropped_rt_axes, d_rt_grids, dd_rt_grids)):
        logging.info(f"\n--- Region {i + 1} ---")
        gmm_result = gmm_deconvolver.model.fit(grid, mz, rt, region_index=i)

        # ==== GMM ====
        if gmm_result and gmm_result.get("overlap_detected", False):
            num_peaks = gmm_result.get("num_peaks_in_overlap", 0)
            confirmed_overlaps_gmm += 1
            total_peaks_in_overlap_gmm += num_peaks
            logging.info(
                f"GMM: Detected {num_peaks} peaks (ΔBIC={gmm_result.get('confidence', 0):.2f}, "
                f"support={gmm_result.get('bic_support', 'N/A')}) → overlap confirmed"
            )
        else:
            logging.info("GMM: Detected single peak → no overlap")

        ridge_result = ridge_deconvolver.model.fit(grid, d_rt, dd_rt)
        # plot_ridges_on_grid(grid, mz, rt, ridge_deconvolver.model.ridges)
        # ==== Ridge ====
        try:
            if ridge_result is None:
                logging.info("Ridge: Analysis failed (no output)")
            elif ridge_result.get("overlap_detected", False):
                num_peaks = ridge_result.get("num_peaks_in_overlap", 0)
                num_events = ridge_result.get("num_overlap_events", 0)
                num_ridges = ridge_result.get("num_ridges_tracked", '?')
                confirmed_overlaps_ridge += 1
                total_peaks_in_overlap_ridge += num_peaks

                logging.info(
                    f"Ridge: Detected overlap involving {num_peaks} peaks "
                    f"(from {num_ridges} ridges, {num_events} event{'s' if num_events != 1 else ''})"
                )
            else:
                total_ridges = ridge_result.get("num_ridges_tracked", 0)
                if total_ridges == 0:
                    logging.info("Ridge: No ridges found")
                elif total_ridges == 1:
                    logging.info("Ridge: Single ridge found (no overlap)")
                else:
                    logging.info(f"Ridge: {total_ridges} ridges tracked, no overlaps detected")
        except Exception as e:
            logging.info(f"Ridge: Exception occurred during analysis: {e}")
        
        # ==== Wavelet ==== 
        wavelet_result = wavelet_deconvolver.model.fit(grid)  # Use the wavelet method
        
        if wavelet_result:
            if wavelet_result.get("overlap_detected", False):
                num_peaks = wavelet_result.get("num_peaks_in_overlap", 0)
                # Only count this as an overlap if it has at least 2 peaks
                if num_peaks >= 2:
                    confirmed_overlaps_wavelet += 1
                    total_peaks_in_overlap_wavelet += num_peaks
                    logging.info(f"Wavelet: Detected {num_peaks} peaks → overlap confirmed")
                else:
                    logging.info(f"Wavelet: Detected {num_peaks} peak(s), but not enough for overlap")
            else:
                logging.info("Wavelet: No overlap detected")

    # === Evaluation: GMM
    if confirmed_overlaps_gmm != expected_overlap_count:
        logging.error(f"FAILED: GMM overlap region count mismatch - expected {expected_overlap_count}, got {confirmed_overlaps_gmm}")
    else:
        logging.info("PASSED: GMM overlap region count correct.")

    if expected_overlap_count > 0:
        if total_peaks_in_overlap_gmm != expected_peaks_in_overlap:
            logging.error(f"FAILED: GMM peak count in overlaps mismatch - expected {expected_peaks_in_overlap}, got {total_peaks_in_overlap_gmm}")
        else:
            logging.info("PASSED: GMM peak count in overlaps correct.")

    # === Evaluation: Ridge
    if confirmed_overlaps_ridge != expected_overlap_count:
        logging.error(f"FAILED: Ridge overlap region count mismatch - expected {expected_overlap_count}, got {confirmed_overlaps_ridge}")
    else:
        logging.info("PASSED: Ridge overlap region count correct.")

    if expected_overlap_count > 0:
        if total_peaks_in_overlap_ridge != expected_peaks_in_overlap:
            logging.error(f"FAILED: Ridge peak count in overlaps mismatch - expected {expected_peaks_in_overlap}, got {total_peaks_in_overlap_ridge}")
        else:
            logging.info("PASSED: Ridge peak count in overlaps correct.")

        # === Evaluation: Wavelet ===
    if confirmed_overlaps_wavelet != expected_overlap_count:
        logging.error(f"FAILED: Wavelet overlap region count mismatch - expected {expected_overlap_count}, got {confirmed_overlaps_wavelet}")
    else:
        logging.info("PASSED: Wavelet overlap region count correct.")

    if expected_overlap_count > 0:
        if total_peaks_in_overlap_wavelet != expected_peaks_in_overlap:
            logging.error(f"FAILED: Wavelet peak count in overlaps mismatch - expected {expected_peaks_in_overlap}, got {total_peaks_in_overlap_wavelet}")
        else:
            logging.info("PASSED: Wavelet peak count in overlaps correct.")

def main():
    failures = 0
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
