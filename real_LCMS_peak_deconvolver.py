from real_LCMS_gmm import LCMS_GMMDeconvolver
from real_LCMS_ridge_walking import LCMS_RidgeWalker
from real_LCMS_wavelet import LCMS_WaveletDeconvolver
from src.deconvolution.visualization import plot_horizontal_gmm
# from .methods.method_2 import Method2Deconvolver  # future

class LCMS_PeakDeconvolver:
    def __init__(self, method="gmm", **kwargs):
        if method == "gmm":
            self.model = LCMS_GMMDeconvolver(**kwargs)
        elif method =="ridge_walk":
            self.model = LCMS_RidgeWalker(**kwargs)
        elif method == "wavelet":
            self.model = LCMS_WaveletDeconvolver(**kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")