import numpy as np
import pandas as pd

class GridGenerator:
    """
    A class to generate retention time (RT) and mass-to-charge (MZ) sampling grids with controlled random variations.
    """

    def __init__(self):
        pass

    def generate_grid(self, rt_start, rt_end, rt_steps, rt_variation,
                      mz_start, mz_end, mz_min_steps, mz_max_steps, mz_variation) -> pd.DataFrame:
        self.rt_start = rt_start
        self.rt_end = rt_end
        self.rt_steps = rt_steps
        self.rt_variation = rt_variation
        self.mz_start = mz_start
        self.mz_end = mz_end
        self.mz_min_steps = mz_min_steps
        self.mz_max_steps = mz_max_steps
        self.mz_variation = mz_variation

        rt_points = self._rt_variable_linspace()
        mz_column = [self._mz_variable_linspace() for _ in rt_points]
        return pd.DataFrame({"rt": rt_points, "mz": mz_column})

    def _rt_variable_linspace(self) -> list:
        if self.rt_steps < 2:
            raise ValueError("rt_steps must be at least 2")
        regular = np.linspace(self.rt_start, self.rt_end, self.rt_steps)
        steps = np.diff(regular)
        variation = np.mean(steps) * self.rt_variation
        noise = np.random.uniform(-variation, variation, self.rt_steps - 2)
        noise -= np.mean(noise)
        modified_steps = steps.copy()
        modified_steps[:-1] += noise
        irregular = [self.rt_start]
        for step in modified_steps:
            irregular.append(irregular[-1] + step)
        return irregular

    def _mz_variable_linspace(self) -> list:
        if self.mz_min_steps < 2:
            raise ValueError("mz_min_steps must be at least 2")
        num_steps = np.random.randint(self.mz_min_steps, self.mz_max_steps)
        regular = np.linspace(self.mz_start, self.mz_end, num_steps)
        steps = np.diff(regular)
        variation = np.mean(steps) * self.mz_variation
        noise = np.random.uniform(-variation, variation, num_steps - 2)
        noise -= np.mean(noise)
        modified_steps = steps.copy()
        modified_steps[:-1] += noise
        irregular = [self.mz_start]
        for step in modified_steps:
            irregular.append(irregular[-1] + step)
        return irregular
