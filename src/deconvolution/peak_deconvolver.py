from .methods.gmm import GMMDeconvolver
from .visualization import plot_horizontal_gmm
# from .methods.method_2 import Method2Deconvolver  # future

class PeakDeconvolver:
    def __init__(self, method="gmm", **kwargs):
        if method == "gmm":
            self.model = GMMDeconvolver(**kwargs)
        elif method =="ridge_walk":
            self.model = RidgeWalker(**kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def fit(self, grid, mz_axis, rt_axis, region_index=0, plot_func=plot_horizontal_gmm):
        return self.model.fit(
            grid, mz_axis, rt_axis, region_index,
            plot_func=plot_func
        )

