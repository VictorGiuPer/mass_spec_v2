import numpy as np
import csv
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.detection.localization import Localizer
from src.detection.suspicion import SuspicionDetector
from src.detection.utils import mark_box
from src.deconvolution.peak_deconvolver import PeakDeconvolver
from src.deconvolution.visualization import plot_horizontal_gmm, plot_ridges_on_grid


from src.generation.export_utils import load_mzml
from src.generation.visualization import plot_gaussians_grid

import numpy as np
import pandas as pd

def build_grid_from_dataframe(df, rt_bin_size=0.1, mz_bin_size=0.01):
    # Step 1: Flatten all (rt, mz, intensity) triples
    rt_all = []
    mz_all = []
    int_all = []

    for _, row in df.iterrows():
        rt = row["RT"]
        mzs = row["mzarray"]
        ints = row["intarray"]

        rt_all.extend([rt] * len(mzs))
        mz_all.extend(mzs)
        int_all.extend(ints)

    rt_all = np.array(rt_all)
    mz_all = np.array(mz_all)
    int_all = np.array(int_all)

    # Step 2: Create axis bins
    rt_min, rt_max = rt_all.min(), rt_all.max()
    mz_min, mz_max = mz_all.min(), mz_all.max()

    rt_axis = np.arange(rt_min, rt_max + rt_bin_size, rt_bin_size)
    mz_axis = np.arange(mz_min, mz_max + mz_bin_size, mz_bin_size)

    # Step 3: Create empty grid
    grid = np.zeros((len(rt_axis), len(mz_axis)), dtype=np.float32)

    # Step 4: Populate grid
    rt_indices = np.searchsorted(rt_axis, rt_all, side='right') - 1
    mz_indices = np.searchsorted(mz_axis, mz_all, side='right') - 1

    valid = (rt_indices >= 0) & (rt_indices < len(rt_axis)) & \
            (mz_indices >= 0) & (mz_indices < len(mz_axis))

    for r_idx, m_idx, intensity in zip(rt_indices[valid], mz_indices[valid], int_all[valid]):
        grid[r_idx, m_idx] += intensity

    return grid, rt_axis, mz_axis

def grid_to_plot_df(grid, rt_axis, mz_axis):
    df_plot = pd.DataFrame({
        "rt": rt_axis,
        "mz": [mz_axis for _ in range(len(rt_axis))],
        "intensities": [row for row in grid]
    })
    return df_plot


def run_real_data_case(filepath, output_csv="real_LCMS.csv"):

    region_rows = []
    runtime_stats = { "GMM": 0.0, "RidgeWalker": 0.0, "Wavelet": 0.0 }

    # 1. Load real LC-MS data
    data_loaded = load_mzml(filepath)  # <-- your existing reader
    if "ms_level" in data_loaded.columns:
            data_loaded = data_loaded.drop("ms_level", axis=1)
        
    print(data_loaded.head())

    grid, rt_axis, mz_axis = build_grid_from_dataframe(data_loaded)
    plot_grid = grid_to_plot_df(grid=grid, rt_axis=rt_axis, mz_axis=mz_axis)
    # plot_gaussians_grid(plot_grid)


    grid = grid.T   
    # 3. Region detection
    processed = np.full(grid.shape, 'unprocessed', object)
    lzr = Localizer(grid, mz_axis, rt_axis, processed)
    cropped = []

    while 'unprocessed' in processed:
        box = lzr.find_next_active_box(global_intensity_thresh=0.15, local_margin=2)
        if not box:
            break
        sub, mzi, rti = lzr.crop_box(box)
        sd = SuspicionDetector(sub, mzi, rti)
        is_susp, _ = sd.detect_suspicious(plot=False)
        mark_box(processed, box, 'processed')
        # lzr.plot_box_on_grid(box, title="First detected region")
        if is_susp:
            cropped.append((sub, mzi, rti, sd.d_grid_rt, sd.dd_grid_rt))

    print(cropped)

    models = {
        "GMM":         PeakDeconvolver(method="gmm").model,
        "RidgeWalker": PeakDeconvolver(method="ridge_walk").model,
        "Wavelet":     PeakDeconvolver(method="wavelet", pad_grid=True).model
    }

    for region_idx, (sub, mzi, rti, d1, d2) in enumerate(cropped):
        for model_name, model in models.items():
            start = time.time()
            
            if model_name == "GMM":
                res = model.fit(sub, mzi, rti, plot_func=plot_horizontal_gmm)
            elif model_name == "RidgeWalker":
                res = model.fit(sub, d1, d2, mzi, rti)
                plot_ridges_on_grid(sub, mzi, rti, model.ridges, title=f"{model_name} Region {region_idx}")
            elif model_name == "Wavelet":
                res = model.fit(sub, mzi, rti)
                model.plot_wavelet_result(
                        res["transformed_grid"],
                        res["peaks"],
                        res["clusters"],
                        title=f"{model_name} Region {region_idx}"
                    )
            else:
                res = model.fit(sub, mzi, rti)
            
            runtime_stats[model_name] += time.time() - start

            # === Log outputs as before ===
            row = {
                "filepath": filepath,
                "region_idx": region_idx,
                "model": model_name,
                "num_peaks": len(res.get("peak_locations", [])),
                "confidence": res.get("confidence", None),
                "fusion_score": res.get("fusion_score", None),
                "runtime": runtime_stats[model_name],
            }
            region_rows.append(row)

    # 6. Save result table
    with open(output_csv, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=region_rows[0].keys())
        writer.writeheader()
        for r in region_rows:
            writer.writerow(r)

    # 7. Print summary
    total_runtime = sum(runtime_stats.values())
    print("\n=== REAL DATA RUNTIME SUMMARY ===")
    for model, t in runtime_stats.items():
        pct = (t / total_runtime) * 100 if total_runtime else 0
        print(f"{model}: {t:.2f}s ({pct:.1f}%)")

def main():
    real_file = r"C:\Users\\victo\VSCode Folder\UMCG Mass Spec\\3_2_extract_overlap_1.mzML"
    run_real_data_case(real_file, "real_LCMS.csv")


main()