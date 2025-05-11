import logging
import numpy as np
import os
from collections import defaultdict

from src.generation import GridGenerator, GaussianGenerator
from src.detection.localization import Localizer
from src.detection.suspicion import SuspicionDetector
from src.detection.utils import mark_box
from src.deconvolution.peak_deconvolver import PeakDeconvolver
from src.deconvolution.visualization import plot_horizontal_gmm
from test_cases import TEST_CASES

# === Logging setup ===
logging.basicConfig(
    filename="test_peak_pipeline_hardened.txt",
    filemode="w",
    encoding="utf-8",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

FILE_MAP = {
    "Two clearly separated peaks": "TEST_CASE.npz",
    "Strong overlap between two peaks": "TEST_CASE_1.npz",
    "Four peaks: 2 overlap, 2 isolated": "TEST_CASE_2.npz",
    "Cluster of 3 overlapping peaks": "TEST_CASE_3.npz",
    "Close but not overlapping peaks": "TEST_CASE_4.npz",
    "Intense + weak overlap": "TEST_CASE_5.npz",
    "Five peaks: 3 spaced, 2 overlapping": "TEST_CASE_6.npz",
}

# === Utilities ===
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


def normalize_intensity(grid, mode="zscore"):
    if mode == "zscore":
        mean = np.mean(grid)
        std = np.std(grid)
        return (grid - mean) / std if std > 0 else grid
    elif mode == "log":
        return np.log1p(grid)
    else:
        raise ValueError(f"Unsupported normalization mode: {mode}")

def get_dynamic_threshold(grid, method="percentile", value=95):
    if method == "percentile":
        return np.percentile(grid, value)
    elif method == "fraction":
        return np.max(grid) * value
    else:
        raise ValueError(f"Unsupported thresholding method: {method}")

def load_grid(label, smoothed=True):
    filename = FILE_MAP.get(label)
    folder = "test_splatted_smooth" if smoothed else "test_splatted"
    path = os.path.join(folder, filename)
    if not filename or not os.path.exists(path):
        raise FileNotFoundError(f"Missing grid file: {path}")
    with np.load(path) as data:
        return data["grid"], data["rt_axis"], data["mz_axis"]

def get_region_truth(expected_num_boxes, expected_overlap_count):
    return [True] * expected_overlap_count + [False] * (expected_num_boxes - expected_overlap_count)

# === Main Testing Function ===
def run_model_suite_on_test_case(label, peak_params, expected_num_boxes, expected_overlap_count,
                                  expected_peaks_in_overlap=None,
                                  normalization="zscore", threshold_mode="percentile"):
    logging.info(f"\n=== Running Test: {label} ===")
    logging.info(f"Input peak parameters: {peak_params}")
    logging.info(f"Expected: {expected_num_boxes} suspicious boxes, {expected_overlap_count} overlaps, {expected_peaks_in_overlap} peaks in overlaps")

    grid, rt_axis, mz_axis = load_grid(label)
    processed_mask = np.full(grid.shape, fill_value='unprocessed', dtype=object)
    lzr = Localizer(grid=grid, mz_axis=mz_axis, rt_axis=rt_axis, processed_mask=processed_mask)

    suspicious_boxes = []
    cropped_regions = []

    while 'unprocessed' in processed_mask:
        box = lzr.find_next_active_box(global_intensity_thresh=0.10, local_margin=2)
        if not box:
            break

        cropped_grid, cropped_mz, cropped_rt = lzr.crop_box(box)
        sd = SuspicionDetector(cropped_grid=cropped_grid, cropped_mz_axis=cropped_mz, cropped_rt_axis=cropped_rt)
        is_suspicious, _ = sd.detect_suspicious(plot=False)
        mark_box(processed_mask, box, label='processed')

        if is_suspicious:
            suspicious_boxes.append(box)
            cropped_regions.append((cropped_grid, cropped_mz, cropped_rt, sd.d_grid_rt, sd.dd_grid_rt))

    logging.info(f"Suspicious boxes detected: {len(suspicious_boxes)}")

    if len(suspicious_boxes) != expected_num_boxes:
        logging.error(f"FAILED: Box test failed: expected {expected_num_boxes}, got {len(suspicious_boxes)}")
    else:
        logging.info("PASSED: Box test passed.")


    region_truth = get_region_truth(expected_num_boxes, expected_overlap_count)
    model_outputs = defaultdict(list)

    gmm = PeakDeconvolver(method="gmm")
    gmm_peaks = 0
    gmm_confirmed = 0

    for region_idx, (raw_grid, mz, rt, d_rt, dd_rt) in enumerate(cropped_regions):
        norm_grid = normalize_intensity(raw_grid, mode=normalization)
        dyn_thresh = get_dynamic_threshold(norm_grid, method=threshold_mode, value=95)

        # GMM
        gmm.model.min_intensity = dyn_thresh
        gmm_result = gmm.model.fit(norm_grid, mz, rt, region_index=region_idx, plot_func=plot_horizontal_gmm)
        model_outputs["GMM"].append({
            "region_index": region_idx,
            "result": gmm_result,
            "overlap_detected": gmm_result.get("overlap_detected", False) if gmm_result else False,
            "num_peaks": gmm_result.get("num_peaks_in_overlap", 0) if gmm_result else 0,
            "peak_locations": gmm_result.get("means", []) if gmm_result else []
        })
        
    # === Evaluation Metrics Per Model ===
    for model_name, outputs in model_outputs.items():
        tp = fp = fn = 0
        peak_count_errors = []
        localization_errors = []

        confirmed_overlaps = 0
        total_detected_peaks = 0

        for i, model_output in enumerate(outputs):
            logging.info(f"\n--- Region {i + 1} ---")

            detected = model_output["overlap_detected"]
            ground_truth = region_truth[i]
            if detected and ground_truth:
                tp += 1
            elif detected and not ground_truth:
                fp += 1
            elif not detected and ground_truth:
                fn += 1

            if detected:
                num_peaks = model_output.get("num_peaks", 0)
                confidence = model_output.get("confidence", 0)
                support = model_output.get("bic_support", "N/A")
                confirmed_overlaps += 1
                total_detected_peaks += num_peaks
                logging.info(f"{model_name}: Detected {num_peaks} peaks (ΔBIC={confidence:.2f}, support={support}) → overlap confirmed")
            else:
                logging.info(f"{model_name}: Detected single peak → no overlap")

            # Peak Count Accuracy
            if expected_peaks_in_overlap is not None:
                detected_peaks = model_output.get("num_peaks_in_overlap")
                if detected_peaks is not None:
                    peak_count_errors.append(abs(detected_peaks - expected_peaks_in_overlap))


            # Localization Accuracy
            detected_locations = model_output.get("peak_locations", [])
            if detected_locations:
                gt_locations = np.array([[p["mz_center"], p["rt_center"]] for p in peak_params])
                if len(detected_locations) == len(gt_locations):
                    distances = np.linalg.norm(detected_locations - gt_locations, axis=1)
                    localization_errors.extend(distances)

        # Detection Summary
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        logging.info(f"\n[{model_name}] Region-Level Detection Summary:")
        logging.info(f"  TP={tp}, FP={fp}, FN={fn}")
        logging.info(f"  Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}")

        if peak_count_errors:
            avg_peak_count_error = np.mean(peak_count_errors)
            logging.info(f"[Category 2] Average Peak Count Error: {avg_peak_count_error:.2f}")

        if localization_errors:
            avg_localization_error = np.mean(localization_errors)
            logging.info(f"[Category 3] Average Localization Error: {avg_localization_error:.2f}")

        # Summary Check (like check_deconv_results)
        check_deconv_results(model_name, confirmed_overlaps, total_detected_peaks, expected_overlap_count, expected_peaks_in_overlap)


# === Run All Test Cases ===
def main():
    for label, peak_params, expected_boxes, expected_overlaps, expected_peaks_in_overlap in TEST_CASES:
        try:
            run_model_suite_on_test_case(label, peak_params, expected_boxes, expected_overlaps, expected_peaks_in_overlap)
        except Exception as e:
            logging.error(f"Error in test '{label}': {e}")

if __name__ == "__main__":
    main()
