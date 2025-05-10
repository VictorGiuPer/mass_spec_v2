import logging
import numpy as np
import os

from src.generation import GridGenerator, GaussianGenerator
from src.generation.splatting import splatting_pipeline, splatted_grid_to_npy
from src.detection.localization import Localizer
from src.detection.suspicion import SuspicionDetector
from src.detection.utils import mark_box
from src.deconvolution.peak_deconvolver import PeakDeconvolver
from src.deconvolution.visualization import plot_ridges_on_grid

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

def run_test_case(label, peak_params, expected_box_count, expected_overlap_count, check_overlap=False):
    logging.info(f"\n\n=== Running Test: {label} ===")
    logging.info(f"Input peak parameters: {peak_params}")

    # === Load pre-generated splatted grid ===
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
    # Suspicion detection
    processed_mask = np.full(grid.shape, fill_value='unprocessed', dtype=object)
    lzr = Localizer(grid=grid, mz_axis=mz_axis, rt_axis=rt_axis, processed_mask=processed_mask)
    box_count = []
    suspicious_boxes = []
    cropped_grids = []
    cropped_mz_axes = []
    cropped_rt_axes = []
    d_rt_grids = []
    dd_rt_grids = []

    while 'unprocessed' in processed_mask:
        box = lzr.find_next_active_box(global_intensity_thresh=0.10, local_margin=2)
        if not box:
            break
        
        # Optional Plot
        # lzr.plot_box_on_grid(box)

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
            # Store derivatives for ridge walking
            d_rt_grids.append(sd.d_grid_rt)
            dd_rt_grids.append(sd.dd_grid_rt)

    box_count = len(box_count)
    suspicious_count = len(suspicious_boxes)
    logging.info(f"Detected boxes: {suspicious_count}")

    if box_count != expected_box_count:
        logging.error(f"❌ Box test failed: expected {expected_box_count}, got {suspicious_count}")
    else:
        logging.info(f"✅ Box test passed.")

    if not check_overlap:
        return

    # === Deconvolution with both methods ===
    # Initialize both deconvolvers
    gmm_deconvolver = PeakDeconvolver(method="gmm")
    ridge_deconvolver = PeakDeconvolver(method="ridge_walk")
    
    confirmed_overlaps_gmm = 0
    confirmed_overlaps_ridge = 0
    
    for i, (grid, mz, rt, d_rt, dd_rt) in enumerate(zip(cropped_grids, cropped_mz_axes, cropped_rt_axes, d_rt_grids, dd_rt_grids)):
        logging.info(f"\n--- Region {i+1} ---")
        
        # Run GMM deconvolution
        gmm_result = gmm_deconvolver.model.fit(grid, mz, rt, region_index=i)
        if gmm_result and gmm_result.get("best_k", 1) > 1:
            confirmed_overlaps_gmm += 1
            logging.info(f"GMM: Detected {gmm_result['best_k']} peaks (ΔBIC={gmm_result.get('confidence', 0):.2f}, support={gmm_result.get('bic_support', 'N/A')})")
        else:
            logging.info(f"GMM: Detected single peak")
        
        # Run Ridge Walking deconvolution
        ridge_result = ridge_deconvolver.model.fit(grid, d_rt=d_rt, dd_rt=dd_rt)
        # plot_ridges_on_grid(grid, mz, rt, ridge_deconvolver.model.ridges)
        try:
            if ridge_result is None:
                logging.info("Ridge: Analysis failed (no output)")
            elif isinstance(ridge_result, list):
                num_ridges = len(ridge_result)
                overlaps = [r for r in ridge_result if r.get('is_overlap', False)]

                if num_ridges == 0:
                    logging.info("Ridge: No ridges found")
                elif len(overlaps) == 0 and num_ridges == 1:
                    logging.info("Ridge: Detected single ridge (no overlap)")
                elif len(overlaps) == 0 and num_ridges > 1:
                    logging.info(f"Ridge: Multiple ridges found, but no overlaps detected")
                else:
                    confirmed_overlaps_ridge += 1
                    best_overlap = max(overlaps, key=lambda x: x.get('confidence', 0))
                    logging.info(f"Ridge: Detected overlap (confidence={best_overlap.get('confidence', 0):.2f}, score={best_overlap.get('fusion_score', 0):.2f})")
            else:
                logging.info("Ridge: Unexpected output format")
        except Exception as e:
            logging.info(f"Ridge: Exception occurred during analysis: {e}")

    
    logging.info(f"\nConfirmed overlaps (GMM): {confirmed_overlaps_gmm}")
    logging.info(f"Confirmed overlaps (Ridge): {confirmed_overlaps_ridge}")

    if confirmed_overlaps_gmm != expected_overlap_count:
        logging.error(f"❌ GMM overlap test failed: expected {expected_overlap_count}, got {confirmed_overlaps_gmm}")
    else:
        logging.info(f"✅ GMM overlap test passed.")
        
    if confirmed_overlaps_ridge != expected_overlap_count:
        logging.error(f"❌ Ridge overlap test failed: expected {expected_overlap_count}, got {confirmed_overlaps_ridge}")
    else:
        logging.info(f"✅ Ridge overlap test passed.")

def main():
    failures = 0
    for i, (label, peak_params, exp_suspicious, exp_overlap) in enumerate(TEST_CASES, start=1):
        try:
            run_test_case(label, peak_params, exp_suspicious, exp_overlap, check_overlap=True)
        except Exception as e:
            logging.error(f"❌ Unhandled error during test '{label}': {e}")
            failures += 1

    if failures == 0:
        logging.info("\n=== All tests completed successfully ===")
    else:
        logging.warning(f"\n=== {failures} tests had errors or failed ===")

if __name__ == "__main__":
    main()
