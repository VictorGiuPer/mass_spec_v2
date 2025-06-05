import os
import sys
import time
import csv
import logging
import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment


# ─── Local Imports ─────────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.detection.localization import Localizer
from src.detection.suspicion import SuspicionDetector
from src.detection.utils import mark_box
from src.deconvolution.peak_deconvolver import PeakDeconvolver
from src.deconvolution.visualization import plot_horizontal_gmm, plot_ridges_on_grid
from results.pipeline_test_cases import TEST_CASES


# ─── Logging Setup ─────────────────────────────────────────────────────────
logging.basicConfig(
    filename="pipeline_extended.txt",
    filemode="w",
    encoding="utf-8",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ─── Globals ───────────────────────────────────────────────────────────────
region_rows = []
runtime_stats = {"GMM": 0.0, "RidgeWalker": 0.0, "Wavelet": 0.0}


# ─── Helper Functions ──────────────────────────────────────────────────────
def load_grid(label):
    safe = label.lower().replace(" ", "_").replace(":", "").replace("-", "").replace("__", "_")
    folder = "test_ext"
    path = os.path.join(folder, f"{safe}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing grid file: {path}")
    with np.load(path) as d:
        return d["grid"], d["rt_axis"], d["mz_axis"]

def get_region_truth(n_boxes, n_overlaps):
    return [True]*n_overlaps + [False]*(n_boxes - n_overlaps)


def assign_gt_peaks_to_region(peak_params, mzi, rti):
    # mzi, rti are arrays covering the region in mz and rt
    mz_min, mz_max = mzi.min(), mzi.max()
    rt_min, rt_max = rti.min(), rti.max()
    region_peaks = [p for p in peak_params 
                    if (mz_min <= p["mz_center"] <= mz_max) and
                      (rt_min <= p["rt_center"] <= rt_max)]
    return region_peaks


# ─── Core Pipeline ─────────────────────────────────────────────────────────
def run_model_suite_on_test_case(label, peak_params, expected_boxes, expected_overlaps, expected_peaks_in_overlap):
    # ─ Setup and logging ─
    logging.info(f"\n=== Running Test: {label} ===")
    logging.info(f"Input peak parameters: {peak_params}")
    logging.info(f"Expected: {expected_boxes} boxes, {expected_overlaps} overlaps, {expected_peaks_in_overlap} peaks in overlaps")

    # ─ Load test grid and initialize region processor ─
    grid, rt_axis, mz_axis = load_grid(label)
    processed = np.full(grid.shape, 'unprocessed', object)
    lzr = Localizer(grid, mz_axis, rt_axis, processed)
    cropped = []

    # ─ Identify and crop suspicious regions ─
    while 'unprocessed' in processed:
        box = lzr.find_next_active_box(global_intensity_thresh=0.10, local_margin=2)
        if not box:
            break
        sub, mzi, rti = lzr.crop_box(box)
        sd = SuspicionDetector(sub, mzi, rti)
        is_susp, _ = sd.detect_suspicious(plot=False)
        mark_box(processed, box, 'processed')
        if is_susp:
            cropped.append((sub, mzi, rti, sd.d_grid_rt, sd.dd_grid_rt))

    logging.info(f"Suspicious boxes detected: {len(cropped)}")
    if len(cropped) != expected_boxes:
        logging.error(f"FAILED box count: expected {expected_boxes}, got {len(cropped)}")
    else:
        logging.info("PASSED box count.")

    # ─ Instantiate models ─
    models = {
        "GMM":         PeakDeconvolver(method="gmm").model,
        "RidgeWalker": PeakDeconvolver(method="ridge_walk").model,
        "Wavelet":     PeakDeconvolver(method="wavelet", pad_grid=True).model
    }

    outputs = {name: [] for name in models}

    # ─ Run each model on each suspicious region ─
    for idx, (sub, mzi, rti, d1, d2) in enumerate(cropped):
        region_gt_peaks = assign_gt_peaks_to_region(peak_params, mzi, rti)
        
        # GMM model
        start = time.time()
        res_gmm = models["GMM"].fit(sub, mzi, rti, region_index=idx)#, plot_func=plot_horizontal_gmm)
        runtime_stats["GMM"] += time.time() - start

        num = res_gmm.get("num_peaks_in_overlap")
        conf = res_gmm.get("confidence")
        outputs["GMM"].append({
            "overlap_detected":     bool(res_gmm.get("overlap_detected", False)),
            "num_peaks_in_overlap": int(num or 0),
            "peak_locations":       np.asarray(res_gmm.get("peak_locations", [])),
            "confidence":           float(conf or 0.0),
            "bic_support":          res_gmm.get("bic_support", None),
            "region_gt_peaks":      region_gt_peaks
        })

        # RidgeWalker model
        start = time.time()
        res_rw = models["RidgeWalker"].fit(sub, d1, d2, mzi, rti)
        runtime_stats["RidgeWalker"] += time.time() - start
        # plot_ridges_on_grid(sub, mzi, rti, models["RidgeWalker"].ridges)
        fusion_scores = None

        if res_rw.get("overlap_details"):
            fusion_scores = [od.get("fusion_score") for od in res_rw["overlap_details"] if "fusion_score" in od]
            fusion_score = max(fusion_scores) if fusion_scores else None
        else:
            fusion_score = None
        
        num = res_rw.get("num_peaks_in_overlap")
        outputs["RidgeWalker"].append({
            "overlap_detected":     bool(res_rw.get("overlap_detected", False)),
            "num_peaks_in_overlap": int(num or 0),
            "peak_locations":       np.asarray(res_rw.get("peak_locations", [])),
            "confidence":           None,
            "bic_support":          None,
            "fusion_score":         fusion_score,
            "region_gt_peaks":      region_gt_peaks
        })

        # Wavelet model
        start = time.time()
        res_wt = models["Wavelet"].fit(sub, mzi, rti)
        runtime_stats["Wavelet"] += time.time() - start

        # models["Wavelet"].plot_wavelet_result(res_wt["transformed_grid"], res_wt["peaks"], res_wt["clusters"], title=f"{label} - Region {idx + 1}")     
        num = res_wt.get("num_peaks_in_overlap")
        outputs["Wavelet"].append({
            "overlap_detected":     bool(res_wt.get("overlap_detected", False)),
            "num_peaks_in_overlap": int(num or 0),  
            "peak_locations":       np.asarray(res_wt.get("peak_locations", [])),
            "confidence":           round(float(res_wt.get("mean_peak_response", 0.0)), 4),
            "bic_support":          None,
            "region_gt_peaks":      region_gt_peaks
        })

    # ─ Region-level Logging ────────────────────────────────────────────────
    labels = get_region_truth(expected_boxes, expected_overlaps)
    for model_name, out_list in outputs.items():
        for region_idx, (o, region_data) in enumerate(zip(out_list, cropped)):
            sub, mzi, rti, d1, d2 = region_data

            region_gt_peaks = assign_gt_peaks_to_region(peak_params, mzi, rti)
            gt_mz = np.array([p["mz_center"] for p in region_gt_peaks])
            gt_rt = np.array([p["rt_center"] for p in region_gt_peaks])

            overlap_true = len(gt_mz) > 1

            # Create results row for this region
            row = {
                "group":            label.rsplit(" ",1)[0],
                "label":            label,
                "model":            model_name,
                "region_idx":       region_idx,
                "overlap_true":     overlap_true,
                "overlap_pred":     o["overlap_detected"],
                "count_error":      None,
                "mean_loc_error_mz": None,
                "mean_loc_error_rt": None,
                "confidence":       o.get("confidence", None),
                "bic_support":      o.get("bic_support", None) if model_name == "GMM" else None,
                "fusion_score": o.get("fusion_score", None) if model_name == "RidgeWalker" else None,

            }

            # Compute peak count error
            pred_count = o.get("num_peaks_in_overlap", 0)
            row["count_error"] = abs(pred_count - len(gt_mz))

            # Compute localization errors (m/z and RT) if both predictions and ground truth are available
            peak_locs = o["peak_locations"]
            det_mz = peak_locs[:, 0] if peak_locs.size else np.array([])
            det_rt = peak_locs[:, 1] if (peak_locs.size and peak_locs.shape[1] > 1) else np.array([])

            if det_mz.size and gt_mz.size:
                # Hungarian assignment on m/z
                cost_mz = np.abs(det_mz[:, None] - gt_mz[None, :])
                r, c = linear_sum_assignment(cost_mz)

                row["mean_loc_error_mz"] = float(cost_mz[r, c].mean())
                if det_rt.size and gt_rt.size:
                    row["mean_loc_error_rt"] = float(np.abs(det_rt[r] - gt_rt[c]).mean())
                else:
                    row["mean_loc_error_rt"] = None
            else:
                row["mean_loc_error_mz"] = None
                row["mean_loc_error_rt"] = None

            region_rows.append(row)

    # ─ Summarize results per model ─
    confusions = {}
    for name, out in outputs.items():
        stats = evaluate_model_outputs(
            out, name, peak_params,
            labels, expected_peaks_in_overlap
        )
        confusions[name] = stats

    return confusions


# ─── Evaluation ────────────────────────────────────────────────────────────

def evaluate_model_outputs(model_outputs, model_name,
                           peak_params, expected_labels, expected_peaks):

    # ─ Compute binary classification metrics ─
    preds = [o["overlap_detected"] for o in model_outputs]
    tp = sum(p and e for p, e in zip(preds, expected_labels)) # True Positives
    fp = sum(p and not e for p, e in zip(preds, expected_labels)) # False Positives
    fn = sum((not p) and e for p, e in zip(preds, expected_labels)) # False Negatives
    tn = sum((not p) and (not e) for p, e in zip(preds, expected_labels)) # True Negatives

    precision = tp/(tp+fp) if (tp+fp) else 1.0
    recall    = tp/(tp+fn) if (tp+fn) else 1.0
    f1        = 2*precision*recall/(precision+recall) if (precision+recall) else 0.0
    accuracy  = (tp+tn)/len(expected_labels)

    # ─ Compute peak count errors ─
    count_errs = []
    for o in model_outputs:
        gt_peaks = o.get("region_gt_peaks", [])
        gt_mz = np.array([p["mz_center"] for p in gt_peaks])

        peak_locs = o["peak_locations"]
        det_mz = peak_locs[:, 0] if peak_locs.size else np.array([])

        count_errs.append(abs(len(det_mz) - len(gt_mz)))

    # ─ Collect model confidence scores ─
    confidences = [o["confidence"] for o in model_outputs]

    # ─ Log all summary statistics ─
    logging.info(f"\n[{model_name}] Detection Summary:")
    logging.info(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    logging.info(f"  Precision={precision:.2f}, Recall={recall:.2f}, "
                 f"F1={f1:.2f}, Accuracy={accuracy:.2f}")

    if count_errs:
        logging.info(f"[Cat-2] Avg. Peak-Count Error = {np.mean(count_errs):.2f}")

    logging.info(f"[Cat-4] Confidence Scores = {confidences}")

    # ─ Return confusion matrix stats ─
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


# ─── Main Execution ────────────────────────────────────────────────────────

def main():
    # ─ Group test cases by their label prefix ─
    tests_by_group = defaultdict(list)
    for args in TEST_CASES:
        label = args[0]
        group = label.rsplit(' ', 1)[0]
        tests_by_group[group].append(args)

    # ─ Run all test cases group by group ─
    for group, cases in tests_by_group.items():
        logging.info(f"\n\n=== TEST SECTION: {group} ===")

        # Initialize per-model statistics for aggregation
        agg = {
            'GMM':         defaultdict(int),
            'RidgeWalker': defaultdict(int),
            'Wavelet':     defaultdict(int)
        }

        # ─ Run all test cases in this group ─
        for args in cases:
            try:
                conf = run_model_suite_on_test_case(*args)
            except Exception as e:
                logging.error(f"Error in test '{args[0]}': {e}")
                continue

            # ─ Accumulate confusion matrix stats across test cases ─
            for model_name, stats in conf.items():
                for k, v in stats.items():
                    agg[model_name][k] += v

        # ─ Log aggregated statistics for each model ─
        for model_name, stats in agg.items():
            tp, fp, fn, tn = stats['tp'], stats['fp'], stats['fn'], stats['tn']
            precision = tp/(tp+fp) if (tp+fp)>0 else 1.0
            recall    = tp/(tp+fn) if (tp+fn)>0 else 1.0
            f1        = 2*precision*recall/(precision+recall+1e-9)
            accuracy  = (tp+tn)/(tp+fp+fn+tn+1e-9)
            logging.info(f"\n*** AGGREGATED [{model_name}] for '{group}' ***")
            logging.info(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
            logging.info(f"  Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}, Accuracy={accuracy:.2f}")

        logging.info("\n=== NEW TEST SECTION ===")

    # ─ Dump region-level results to CSV for visualization/analysis ─
    keys = ["group","label","model","region_idx",
            "overlap_true","overlap_pred",
            "count_error","mean_loc_error_mz", "mean_loc_error_rt",
            "confidence","bic_support", "fusion_score"]

    # ─ Log total runtime per model ─
    print("\n=== TOTAL RUNTIME SUMMARY ACROSS ALL SCENARIOS ===")
    total_runtime = sum(runtime_stats.values())
    for method, runtime in runtime_stats.items():
        percent = (runtime / total_runtime) * 100 if total_runtime > 0 else 0
        print(f"{method}: {runtime:.4f} seconds ({percent:.2f}%)")

"""     # ─ Write per-region metrics to disk ─
    with open("region_results_new.csv", "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=keys)
        writer.writeheader()
        for r in region_rows:
            writer.writerow(r) """

# ─ Run main entry point ─
if __name__ == "__main__":
    main()
