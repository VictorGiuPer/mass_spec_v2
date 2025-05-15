import logging
import numpy as np
import os
import csv
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

from src.detection.localization import Localizer
from src.detection.suspicion import SuspicionDetector
from src.detection.utils import mark_box
from src.deconvolution.peak_deconvolver import PeakDeconvolver
from src.deconvolution.visualization import plot_horizontal_gmm, plot_ridges_on_grid
from test_cases import TEST_CASES

# === Logging setup ===
logging.basicConfig(
    filename="REVISED.txt",
    filemode="w",
    encoding="utf-8",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# will hold per-region results across all tests:
region_rows = []
_region_records = []


def load_grid(label, smoothed=True):
    safe = label.lower().replace(" ", "_").replace(":", "").replace("-", "").replace("__", "_")
    folder = "test_ext" if smoothed else "test_ext_raw"
    path = os.path.join(folder, f"{safe}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing grid file: {path}")
    with np.load(path) as d:
        return d["grid"], d["rt_axis"], d["mz_axis"]

def get_region_truth(n_boxes, n_overlaps):
    return [True]*n_overlaps + [False]*(n_boxes - n_overlaps)

def run_model_suite_on_test_case(label, peak_params, expected_boxes, expected_overlaps, expected_peaks_in_overlap):
    logging.info(f"\n=== Running Test: {label} ===")
    logging.info(f"Input peak parameters: {peak_params}")
    logging.info(f"Expected: {expected_boxes} boxes, {expected_overlaps} overlaps, {expected_peaks_in_overlap} peaks in overlaps")

    grid, rt_axis, mz_axis = load_grid(label)
    processed = np.full(grid.shape, 'unprocessed', object)
    lzr = Localizer(grid, mz_axis, rt_axis, processed)
    cropped = []

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

    models = {
        "GMM":         PeakDeconvolver(method="gmm").model,
        "RidgeWalker": PeakDeconvolver(method="ridge_walk").model,
        "Wavelet":     PeakDeconvolver(method="wavelet", pad_grid=True).model
    }

    outputs = {name: [] for name in models}
    for idx, (sub, mzi, rti, d1, d2) in enumerate(cropped):
        # GMM
        res = models["GMM"].fit(sub, mzi, rti, region_index=idx, plot_func=plot_horizontal_gmm)
        num = res.get("num_peaks_in_overlap")
        conf = res.get("confidence")
        outputs["GMM"].append({
            "overlap_detected":     bool(res.get("overlap_detected", False)),
            "num_peaks_in_overlap": int(num or 0),
            "peak_locations":       np.asarray(res.get("peak_locations", [])),
            "confidence":           float(conf or 0.0),
            "bic_support":          res.get("bic_support", None)
        })

        # RidgeWalker
        res = models["RidgeWalker"].fit(sub, d1, d2, mzi, rti)
        num = res.get("num_peaks_in_overlap")
        outputs["RidgeWalker"].append({
            "overlap_detected":     bool(res.get("overlap_detected", False)),
            "num_peaks_in_overlap": int(num or 0),
            "peak_locations":       np.asarray(res.get("peak_locations", [])),
            "confidence":           None,
            "bic_support":          None
        })

        # Wavelet
        res = models["Wavelet"].fit(sub, mzi, rti)
        num = res.get("num_peaks_in_overlap")
        outputs["Wavelet"].append({
            "overlap_detected":     bool(res.get("overlap_detected", False)),
            "num_peaks_in_overlap": int(num or 0),
            "peak_locations":       np.asarray(res.get("peak_locations", [])),
            "confidence":           None,
            "bic_support":          None
        })

    # ── Begin region‐level logging ───────────────────────────────────────────
    labels = get_region_truth(expected_boxes, expected_overlaps)
    for model_name, out_list in outputs.items():
        gt_mz = np.array([p["mz_center"] for p in peak_params])
        for region_idx, (o, truth) in enumerate(zip(out_list, labels)):
            row = {
                "group":            label.rsplit(" ",1)[0],
                "label":            label,
                "model":            model_name,
                "region_idx":       region_idx,
                "overlap_true":     truth,
                "overlap_pred":     o["overlap_detected"],
                "count_error":      None,
                "mean_loc_error":   None,
                "confidence":       o["confidence"],

                # === New fields for GMM only (else None) ===
                "confidence": o["confidence"] if model_name == "GMM" else None,
                "bic_support": o.get("bic_support", None) if model_name == "GMM" else None,
                "separation_score": o.get("separation_score", None) if model_name == "GMM" else None,
                "confidence_percent_raw": o.get("confidence_percent_raw", None) if model_name == "GMM" else None,
                "confidence_percent_adjusted": o.get("confidence_percent_adjusted", None) if model_name == "GMM" else None,
            }
            if truth and o["overlap_detected"]:
                row["count_error"] = abs(o["num_peaks_in_overlap"] - expected_peaks_in_overlap)
                det = o["peak_locations"][:,0] if o["peak_locations"].size else np.array([])
                if det.size and gt_mz.size:
                    cost = np.abs(det[:,None] - gt_mz[None,:])
                    r,c = linear_sum_assignment(cost)
                    row["mean_loc_error"] = float(cost[r,c].mean())
            region_rows.append(row)
    # ── End region‐level logging ─────────────────────────────────────────────

    confusions = {}
    for name, out in outputs.items():
        stats = evaluate_model_outputs(
            out, name, peak_params,
            labels, expected_peaks_in_overlap
        )
        confusions[name] = stats

    return confusions

def evaluate_model_outputs(model_outputs, model_name, peak_params, expected_labels, expected_peaks):
    preds = [o["overlap_detected"] for o in model_outputs]
    tp = sum(p and e for p,e in zip(preds, expected_labels))
    fp = sum(p and not e for p,e in zip(preds, expected_labels))
    fn = sum(not p and e for p,e in zip(preds, expected_labels))
    tn = sum(not p and not e for p,e in zip(preds, expected_labels))

    precision = tp/(tp+fp) if (tp+fp)>0 else 1.0
    recall    = tp/(tp+fn) if (tp+fn)>0 else 1.0
    f1        = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
    accuracy  = (tp+tn)/len(expected_labels)

    count_errs, loc_errs = [], []
    if expected_peaks is not None:
        gt_mz = np.array([p["mz_center"] for p in peak_params])
        for o, truth in zip(model_outputs, expected_labels):
            if truth and o["overlap_detected"]:
                count_errs.append(abs(o["num_peaks_in_overlap"] - expected_peaks))
                det = o["peak_locations"][:,0] if o["peak_locations"].size else np.array([])
                if det.size and gt_mz.size:
                    cost = np.abs(det[:,None] - gt_mz[None,:])
                    r, c = linear_sum_assignment(cost)
                    loc_errs.extend(cost[r,c])

    confidences = [o["confidence"] for o in model_outputs]

    logging.info(f"\n[{model_name}] Detection Summary:")
    logging.info(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    logging.info(f"  Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}, Accuracy={accuracy:.2f}")

    if expected_peaks is None:
        logging.info("[Category 2] skip count error (no overlaps expected).")
        logging.info("[Category 3] skip localization error.")
    else:
        if count_errs:
            logging.info(f"[Category 2] Avg. Peak Count Error = {np.mean(count_errs):.2f}")
        else:
            logging.info("[Category 2] No overlaps → skip count error.")
        if loc_errs:
            logging.info(f"[Category 3] Avg. Localization Error (m/z) = {np.mean(loc_errs):.2f}")
        else:
            logging.info("[Category 3] No localization errors to report.")

    logging.info(f"[Category 4] Confidence Scores = {confidences}")

    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}

def main():
    tests_by_group = defaultdict(list)
    for args in TEST_CASES:
        label = args[0]
        group = label.rsplit(' ', 1)[0]
        tests_by_group[group].append(args)

    for group, cases in tests_by_group.items():
        logging.info(f"\n\n=== TEST SECTION: {group} ===")
        agg = {
            'GMM':         defaultdict(int),
            'RidgeWalker': defaultdict(int),
            'Wavelet':     defaultdict(int)
        }
        for args in cases:
            try:
                conf = run_model_suite_on_test_case(*args)
            except Exception as e:
                logging.error(f"Error in test '{args[0]}': {e}")
                continue
            for model_name, stats in conf.items():
                for k, v in stats.items():
                    agg[model_name][k] += v

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

    # ── dump region‐level results for later plotting ─────────────────────────
    keys = ["group","label","model","region_idx",
            "overlap_true","overlap_pred",
            "count_error","mean_loc_error",
            "confidence","bic_support","separation_score",
            "confidence_percent_raw","confidence_percent_adjusted"]

    with open("region_results_confidence.csv", "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=keys)
        writer.writeheader()
        for r in region_rows:
            writer.writerow(r)
    # ── done ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
