import os
import pandas as pd
from pyopenms import MSExperiment, MzMLFile, MSSpectrum

def grid_to_json(df, base_filename="gen_grid"):
    filename = f"{base_filename}.json"
    counter = 1
    while os.path.exists(filename):
        filename = f"{base_filename}_{counter}.json"
        counter += 1
    df.to_json(filename, index=False)
    print(f"File saved as: {filename}")

def gaussians_grid_to_json(df, base_filename="gen_gaussians_grid"):
    filename = f"{base_filename}.json"
    counter = 1
    while os.path.exists(filename):
        filename = f"{base_filename}_{counter}.json"
        counter += 1
    df.to_json(filename, index=False)
    print(f"File saved as: {filename}")

def gaussians_grid_to_mzml(df, output_path_base="output.mzML"):
    df_prepped = df.rename(columns={"rt": "RT", "mz": "mzarray", "intensities": "intarray"})
    base, ext = os.path.splitext(output_path_base)
    filepath = output_path_base
    count = 1
    while os.path.exists(filepath):
        filepath = f"{base}_{count}{ext}"
        count += 1

    print(f"Saving to: {filepath}")

    exp = MSExperiment()
    for i, row in df_prepped.iterrows():
        spectrum = MSSpectrum()
        spectrum.setRT(row["RT"])
        spectrum.setMSLevel(1)
        spectrum.set_peaks([row["mzarray"], row["intarray"]])
        exp.addSpectrum(spectrum)
    MzMLFile().store(filepath, exp)
    print(f"File saved as: {filepath}")

def load_mzml(filepath):
    experiment = MSExperiment()
    MzMLFile().load(filepath, experiment)
    data_loaded = experiment.get_df()
    print(data_loaded.head())
    return data_loaded

def zoom_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crops the RT-MZ grid to user-specified RT and MZ ranges.
    Ensures mz and intensity arrays stay aligned after cutting.
    """

    print("\nEnter zoom cutoff ranges:")

    rt_min = float(input("Minimum RT: "))
    rt_max = float(input("Maximum RT: "))
    mz_min = float(input("Minimum MZ: "))
    mz_max = float(input("Maximum MZ: "))

    # Step 1: Drop RT rows outside range
    df_zoomed = df[(df["rt"] >= rt_min) & (df["rt"] <= rt_max)].copy()

    # Step 2: For each row, cut mz and intensities arrays together
    def filter_mz_and_intensity(row):
        mz_filtered = []
        intensity_filtered = []
        for mz, intensity in zip(row["mz"], row["intensities"]):
            if mz_min <= mz <= mz_max:
                mz_filtered.append(mz)
                intensity_filtered.append(intensity)
        return pd.Series([mz_filtered, intensity_filtered])

    df_zoomed[["mz", "intensities"]] = df_zoomed.apply(filter_mz_and_intensity, axis=1)
    df_zoomed = df_zoomed[df_zoomed["mz"].map(len) > 0]

    print("Zoomed grid ready.")
    return df_zoomed