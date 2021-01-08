import numpy as np
from pyvf.ext import array_lru_cache, cached_property


class GrowthPattern:
    def adjust(self, mean, std, mean_est, std_est=None):
        """
        Adjust the current prior assumption of mean based on finished estimates

        Parameters
        ----------
        std_est
        mean : array of model mean at each location
        std : array of model standard deviation at each location
        mean_est : array of currently known thresholds through testing
        std_est : array of currently known standard deviation of thresholds through testing

        Returns
        -------
        mean_infer : array of inferred thresholds
        std_infer : array of inferred standard deviation of thresholds

        """
        raise NotImplementedError()


class EmptyGrowthPattern(GrowthPattern):
    def adjust(self, mean, std, mean_est, std_est=None):
        """
        This growth pattern is equivalent to treating every location to be independent
        Returns the prior model mean and std directly without any processing
        This means that no matter what mean_est is,
        every point already has a finite prior mean to start
        """
        return mean, std


class SimpleGrowthPattern(GrowthPattern):
    def __init__(self):
        self.pattern = {}
        self.agg_fun = np.nanmean

    def adjust(self, mean, std, mean_est, std_est=None):
        if std_est is None:
            std_est = np.full_like(std, fill_value=np.nan)

        return self._adjust_mean(mean, mean_est), std_est

    @cached_property
    def _adjust_mean(self):
        @array_lru_cache()
        def cached_fun(mean, mean_est):
            mean_infer = mean.copy()
            for k, v in self.pattern.items():
                offsets = mean_est[v] - mean[v]
                if np.isfinite(offsets).all():
                    aggregate_offset = self.agg_fun(offsets)
                    mean_infer[k] += aggregate_offset
                else:
                    mean_infer[k] = np.nan

            return mean_infer
        return cached_fun


class SimpleP24d2QuadrantGrowth(SimpleGrowthPattern):
    def __init__(self):
        super().__init__()
        self.pattern = {
         0: [12], 1: [12], 4: [12], 5: [12], 6: [12],10: [12],11: [12],13: [12],18: [12],19: [12],20: [12],21: [12],22: [12],
         2: [15], 3: [15], 7: [15], 8: [15], 9: [15],14: [15],16: [15],17: [15],23: [15],24: [15],26: [15],
        27: [38],28: [38],29: [38],30: [38],31: [38],36: [38],37: [38],39: [38],44: [38],45: [38],46: [38],50: [38],51: [38],
        32: [41],33: [41],35: [41],40: [41],42: [41],43: [41],47: [41],48: [41],49: [41],52: [41],53: [41],
        }
