import numpy as np
import sys
import os

# Add parent folder (mass_spec_project) to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.detection.localization import Localizer
from src.detection.suspicion import SuspicionDetector
from src.detection.utils import save_to_npz, mark_box

# Import classes and functions

data = np.load(r"C:\Users\victo\VSCode Folder\mass_spec_project\data\splatted\smoothed_splatted_grid.npz")
# print(data)
grid = data["grid"]
mz_axis = data["mz_axis"]
rt_axis = data["rt_axis"]

processed_mask = np.full(grid.shape, fill_value='unprocessed', dtype=object)

all_boxes = []
suspicious_boxes = []   
all_cropped_grids = []
all_cropped_mz_axes = []
all_cropped_rt_axes = []
all_d_rt = []
all_dd_rt = []

lzr = Localizer(grid=grid, mz_axis=mz_axis, rt_axis=rt_axis, processed_mask=processed_mask)

# Main Processing Loop
while 'unprocessed' in processed_mask:
    # 1. Find next active box
    box = lzr.find_next_active_box(global_intensity_thresh=0.001)

    # OPTIONAL: Check box location
    if box is not None:
        lzr.plot_box_on_grid(box=box)
    
    if box is None:
        break  # No more active regions

    # 2. Save box
    all_boxes.append(box)

    # 3. Crop region from the grid
    cropped_grid, cropped_mz_axis, cropped_rt_axis = lzr.crop_box(box=box)

    sd = SuspicionDetector(cropped_grid=cropped_grid, cropped_mz_axis=cropped_mz_axis, cropped_rt_axis=cropped_rt_axis)
    # 4. Run suspicious signal detection
    is_suspicious, zones = sd.detect_suspicious(plot=True)
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
        all_d_rt.append(sd.d_grid_rt)
        all_dd_rt.append(sd.dd_grid_rt)


# save_to_npz(all_cropped_grids, all_cropped_mz_axes, all_cropped_rt_axes, 
            # zones, all_d_rt=all_d_rt, all_dd_rt=all_dd_rt)