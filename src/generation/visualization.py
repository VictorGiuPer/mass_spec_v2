import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def plot_grid(df):
    scatter_data = df.explode('mz')
    fig = px.scatter(scatter_data, x='rt', y='mz', title='RT vs MZ Scatterplot',
                     labels={'rt': 'RT', 'mz': 'MZ'})
    fig.update_layout(width=1200, height=800, template="plotly_white")
    fig.update_traces(marker=dict(size=2, opacity=0.8))
    fig.show()

def plot_gaussians_grid(df, title="Interpolated mz Heatmap", zoom=False, mz_points=1000):
    rt_values = df["rt"].values
    all_mz = np.concatenate(df["mz"].values)
    mz_min, mz_max = all_mz.min(), all_mz.max()
    common_mz = np.linspace(mz_min, mz_max, mz_points)

    interpolated_matrix = []
    for mz, intensity in zip(df["mz"], df["intensities"]):
        interp = interp1d(mz, intensity, kind='linear', bounds_error=False, fill_value=0)
        interpolated_matrix.append(interp(common_mz))

    intensity_matrix = np.array(interpolated_matrix).T

    plt.figure(figsize=(12, 5))
    plt.imshow(intensity_matrix, extent=[rt_values[0], rt_values[-1], common_mz[-1], common_mz[0]],
               aspect='auto', cmap='viridis')
    plt.colorbar(label='Intensity')
    plt.xlabel('Retention Time (min)')
    plt.ylabel('mz (Da)')
    plt.title(title)
    plt.show()

    fig = px.imshow(intensity_matrix, origin='lower', aspect='auto',
                    labels=dict(x="Retention Time", y="m/z", color="Intensity"),
                    x=rt_values, y=common_mz, color_continuous_scale="Viridis")
    fig.update_layout(title="2D Image Plot of LC-MS Data", width=800, height=600)
    fig.show()
