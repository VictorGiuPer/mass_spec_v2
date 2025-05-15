import logging
import numpy as np
import os
from collections import defaultdict

from src.detection.localization import Localizer
from src.detection.suspicion import SuspicionDetector
from src.detection.utils import mark_box
from src.deconvolution.peak_deconvolver import PeakDeconvolver

import logging
import numpy as np
import os
from collections import defaultdict

from src.detection.localization import Localizer
from src.detection.suspicion import SuspicionDetector
from src.detection.utils import mark_box
from src.deconvolution.peak_deconvolver import PeakDeconvolver
from src.deconvolution.visualization import plot_horizontal_gmm, plot_ridges_on_grid
from test_cases import TEST_CASES
from scipy.optimize import linear_sum_assignment


# === Logging setup ===
logging.basicConfig(
    filename="test_peak_pipeline_hardened.txt",
    filemode="w",
    encoding="utf-8",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# UTILITY
def check_deconv_results(method, found_regions, found_peaks, expected_regions, expected_peaks):
    """Log pass/fail status for region and peak count matches."""
    if found_regions != expected_regions:
        logging.error(f"FAILED: {method} overlap region count mismatch - expected {expected_regions}, got {found_regions}")
    else:
        logging.info(f"PASSED: {method} overlap region count correct.")

    if expected_regions > 0:
        if found_peaks != expected_peaks:
            logging.error(f"FAILED: {method} peak count in overlaps mismatch - expected {expected_peaks}, got {found_peaks}")
        else:
            logging.info(f"PASSED: {method} peak count in overlaps correct.")

# I/O HELPER
def load_grid(label, smoothed=True):
    """Load a test grid and its axes from disk."""
    # Generate safe filename from label
    safe_label = label.lower().replace(" ", "_").replace(":", "").replace("-", "").replace("__", "_")
    filename = f"{safe_label}.npz"
    folder = "test_ext" if smoothed else "test_ext_raw"  # or just "test_ext" if you only have one version
    path = os.path.join(folder, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing grid file: {path}")

    with np.load(path) as data:
        return data["grid"], data["rt_axis"], data["mz_axis"]

# TRUTH LABEL HELPER
def get_region_truth(expected_num_boxes, expected_overlap_count):
    """Generate a list of expected overlap labels for evaluation."""
    return [True] * expected_overlap_count + [False] * (expected_num_boxes - expected_overlap_count)

# MAIN TESTING FUNCTION
def run_model_suite_on_test_case(label, peak_params, expected_num_boxes, expected_overlap_count,
                                  expected_peaks_in_overlap=None):
    """Run detection and deconvolution models on a test case label."""
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

        # lzr.plot_box_on_grid(box, title=label)

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

    model_outputs = defaultdict(list)

    gmm = PeakDeconvolver(method="gmm")
    ridge_walk = PeakDeconvolver(method="ridge_walk")
    wavelet = PeakDeconvolver(method="wavelet", pad_grid=True)

    for region_idx, (raw_grid, mz, rt, d_rt, dd_rt) in enumerate(cropped_regions):
           
        # GMM
        gmm_result = gmm.model.fit(raw_grid, mz, rt, region_index=region_idx)#, plot_func=plot_horizontal_gmm)
        model_outputs["GMM"].append({
            "region_index": region_idx,
            "result": gmm_result,
            "overlap_detected": gmm_result.get("overlap_detected", False) if gmm_result else False,
            "num_peaks": gmm_result.get("num_peaks_in_overlap", 0) if gmm_result else 0,
            "peak_locations": gmm_result.get("peak_locations", []) if gmm_result else []
        })
        # print(model_outputs["GMM"][-1]["peak_locations"])
        
        # Ridge Walk
        ridge_result = ridge_walk.model.fit(raw_grid, d_rt, dd_rt, mz, rt)
        # Optional: visualize ridges
        # plot_ridges_on_grid(raw_grid, mz, rt, ridge_walk.model.ridges, title=f"{label} - Region {region_idx + 1}")

        model_outputs["RidgeWalker"].append({
            "region_index": region_idx,
            "result": ridge_result,
            "overlap_detected": ridge_result.get("overlap_detected", False),
            "num_peaks": ridge_result.get("num_peaks_in_overlap", 0) if ridge_result else 0,
            "peak_locations": ridge_result.get("peak_locations", []) if ridge_result else []
        })

        # print(model_outputs["RidgeWalker"][-1]["peak_locations"])
 
        # Wavelet
        wavelet_result = wavelet.model.fit(raw_grid, mz, rt)
        # wavelet.model.plot_wavelet_result(raw_grid, wavelet_result["transformed_grid"], wavelet_result["peaks"], wavelet_result["clusters"], title=f"{label} - Region {region_idx + 1}")
        model_outputs["Wavelet"].append({
            "region_index": region_idx,
            "result": wavelet_result,
            "overlap_detected": wavelet_result.get("overlap_detected", False),
            "num_peaks": wavelet_result.get("num_peaks_in_overlap", 0) if wavelet_result else 0,
            "peak_locations": wavelet_result.get("peak_locations", []) if wavelet_result else []
        })
        # print(model_outputs["Wavelet"][-1]["peak_locations"])

    for model_name, outputs in model_outputs.items():
        evaluate_model_outputs(outputs, model_name, peak_params, expected_overlap_count, expected_peaks_in_overlap)

# EVALUATION
def evaluate_model_outputs(model_outputs, model_name, peak_params, expected_overlap_count, expected_peaks_in_overlap):
    """Evaluate deconvolution results against ground truth."""
    peak_count_errors = []
    localization_errors = []

    overlap_detected_anywhere = any(output["overlap_detected"] for output in model_outputs)
    overlap_expected = expected_overlap_count > 0

    tp = fp = fn = tn = 0
    if overlap_detected_anywhere and overlap_expected:
        tp = 1
    elif overlap_detected_anywhere and not overlap_expected:
        fp = 1
    elif not overlap_detected_anywhere and overlap_expected:
        fn = 1
    else:
        tn = 1

    confirmed_overlaps = sum(1 for output in model_outputs if output["overlap_detected"])
    total_detected_peaks = sum(output.get("num_peaks", 0) for output in model_outputs if output["overlap_detected"])

    for i, model_output in enumerate(model_outputs):
        logging.info(f"\n--- Region {i + 1} ---")
        detected = model_output["overlap_detected"]
        num_peaks = model_output.get("num_peaks", 0)
        confidence = model_output.get("confidence", 0)
        support = model_output.get("bic_support", "N/A")

        if detected:
            logging.info(f"{model_name}: Detected {num_peaks} peaks (ΔBIC={confidence:.2f}, support={support}) → overlap confirmed")
        else:
            logging.info(f"{model_name}: Detected single peak → no overlap")

        if expected_peaks_in_overlap is not None and detected:
            detected_peaks = model_output.get("num_peaks_in_overlap", 0)
            peak_count_errors.append(abs(detected_peaks - expected_peaks_in_overlap))

            detected_locations = np.array(model_output.get("peak_locations", []))
            gt_locations = np.array([[p["mz_center"], p["rt_center"]] for p in peak_params])

            if detected_locations.size and gt_locations.size:
                cost_matrix = np.linalg.norm(
                    detected_locations[:, None, :] - gt_locations[None, :, :], axis=2
                )
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                localization_errors.extend(cost_matrix[row_ind, col_ind])

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-9)

    logging.info(f"\n[{model_name}] Detection Summary (aggregate):")
    logging.info(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    logging.info(f"  Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}, Accuracy={accuracy:.2f}")

    if peak_count_errors:
        logging.info(f"[Category 2] Average Peak Count Error: {np.mean(peak_count_errors):.2f}")

    if localization_errors:
        logging.info(f"[Category 3] Average Localization Error: {np.mean(localization_errors):.2f}")

    check_deconv_results(
        model_name,
        confirmed_overlaps,
        total_detected_peaks,
        expected_overlap_count,
        expected_peaks_in_overlap
    )

# ENTRY POINT
def main():
    """Loop through all test cases and run the full evaluation suite."""
    for label, peak_params, expected_boxes, expected_overlaps, expected_peaks_in_overlap in TEST_CASES:
        try:
            run_model_suite_on_test_case(label, peak_params, expected_boxes, expected_overlaps, expected_peaks_in_overlap)
            logging.info(f"\n\n=== NEW TEST ===")
        except Exception as e:
            logging.error(f"Error in test '{label}': {e}")

if __name__ == "__main__":
    main()



### UNUSED (TRIED IMPLEMENTATION BUT COUNTERPRODUCTIVE ###
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