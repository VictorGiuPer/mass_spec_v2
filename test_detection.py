import sys
import os
import numpy as np


# Add parent folder (mass_spec_project) to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.peak_detection.localization import find_next_active_box, crop_box, plot_box_on_grid
from src.peak_detection.detect_suspicious import detect_suspicious, mark_box, suspicious_regions_to_npz

# Import classes and functions

data = np.load(r"C:\Users\victo\VSCode Folder\mass_spec_project\data\splatted\smoothed_splatted_grid.npz")
# print(data)
grid = data["grid"]
mz_axis = data["mz_axis"]
rt_axis = data["rt_axis"]

processed_mask = np.full_like(grid, fill_value='unprocessed', dtype=object)
all_boxes = []
suspicious_boxes = []
all_cropped_grids = []
all_cropped_mz_axes = []
all_cropped_rt_axes = []

# Main Processing Loop
while 'unprocessed' in processed_mask:
    # 1. Find next active box
    box = find_next_active_box(grid, processed_mask, mz_axis, rt_axis, global_intensity_thresh=0.001)

    # OPTIONAL: Check box location
    if box is not None:
        plot_box_on_grid(grid, box, mz_axis, rt_axis)
    
    if box is None:
        break  # No more active regions

    # 2. Save box
    all_boxes.append(box)

    # 3. Crop region from the grid
    cropped_grid, cropped_mz_axis, cropped_rt_axis = crop_box(grid, mz_axis, rt_axis, box)

    # 4. Run suspicious signal detection
    is_suspicious = detect_suspicious(cropped_grid, cropped_mz_axis, plot=False)
    print(f"Suspicious: {is_suspicious}")

    # 5. Mark box as processed
    label = 'suspicious' if is_suspicious else 'processed'
    mark_box(processed_mask, box, label=label)

    if is_suspicious:
        suspicious_boxes.append(box)
        print(f"Suspicious boxes found: {len(suspicious_boxes)}")
        all_cropped_grids.append(cropped_grid)
        all_cropped_mz_axes.append(cropped_mz_axis)
        all_cropped_rt_axes.append(cropped_rt_axis)


suspicious_regions_to_npz(all_cropped_grids, all_cropped_mz_axes, all_cropped_rt_axes)