import numpy as np
import os

def mark_box(processed_mask, box, label):
        mz_slice = slice(box['mz_min_idx'], box['mz_max_idx'] + 1)
        rt_slice = slice(box['rt_min_idx'], box['rt_max_idx'] + 1)
        processed_mask[mz_slice, rt_slice] = label

def save_to_npz(cropped_grids, mz_axes, rt_axes, zones, all_d_rt, all_dd_rt, base_filename="suspicious_peaks"):
        output_folder = "data/suspicious_peaks"
        os.makedirs(output_folder, exist_ok=True)

        filename = os.path.join(output_folder, f"{base_filename}.npz")
        counter = 1
        while os.path.exists(filename):
            filename = os.path.join(output_folder, f"{base_filename}_{counter}.npz")
            counter += 1

        save_dict = {
            "grids": np.array(cropped_grids, dtype=object),
            "mz_axes": np.array(mz_axes, dtype=object),
            "rt_axes": np.array(rt_axes, dtype=object),
            "d_rt": np.array(all_d_rt, dtype=object),
            "dd_rt": np.array(all_dd_rt, dtype=object),
            "zones": np.array(zones, dtype=object)
        }

        np.savez(filename, **save_dict)
        print(f"Suspicious regions saved to: {filename}")