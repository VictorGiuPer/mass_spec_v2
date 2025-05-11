from .methods.gmm import GMMDeconvolver
from .methods.ridge_walking import RidgeWalker
from .methods.wavelet import WaveletDeconvolver
from .visualization import plot_horizontal_gmm
# from .methods.method_2 import Method2Deconvolver  # future

class PeakDeconvolver:
    def __init__(self, method="gmm", **kwargs):
        if method == "gmm":
            self.model = GMMDeconvolver(**kwargs)
        elif method =="ridge_walk":
            self.model = RidgeWalker(**kwargs)
        elif method == "wavelet":
            self.model = WaveletDeconvolver(**kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")