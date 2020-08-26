"""
Interface and implementations for normal population average values for use by perimetry strategies as starting values,
also for analysis of results against this baseline.
Companion GrowthPattern classes codes for interface and implementations for during-test adjustments of the starting values.


Copyright 2020 Bill Runjie Shi
At the Vision and Eye Movements Lab, University of Toronto.
Visit us at: http://www.eizenman.ca/

This file is part of PyVF.

PyVF is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

PyVF is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with PyVF. If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np

from pyvf.strategy import XOD, YOD, PATTERN_P24D2


class Model:
    """
    Base class for models for calculating normal healthy population threshold values
    """
    def __init__(self, eval_pattern, age=None, *args, **kwargs):
        """

        Parameters
        ----------
        eval_pattern : see Strategy
        age : age of the patient in years
        args
        kwargs
        """
        self.args = args
        self.param = kwargs
        self.param['eval_pattern'] = eval_pattern
        self.param['age'] = age

    def get_mean(self):
        raise NotImplementedError()

    def get_std(self):
        raise NotImplementedError()


class AgeLinearModel(Model):
    def __init__(self, eval_pattern, age, model_pattern, intercept, slope, std=None, *args, **kwargs):
        """

        Parameters
        ----------
        eval_pattern
        age
        model_pattern
        intercept
        slope
        std : standard deviation at each of the model_pattern location. Optional. If not provided, this is set to None and any operation that involves std will cause undefined behavior
        args
        kwargs
        """
        super().__init__(eval_pattern, age, *args, **kwargs)
        self.param["intercept"] = intercept
        self.param["slope"] = slope
        self.param["std"] = std
        self.param["model_pattern"] = model_pattern

        # If the input model_pattern is not the same as output eval_pattern, then interpolation must be performed.
        # However this is not currently implemented
        if (len(model_pattern) != len(eval_pattern) or
                not np.all(model_pattern[XOD] == eval_pattern[XOD]) or
                not np.all(model_pattern[YOD] == eval_pattern[YOD])):
            raise ValueError("currently model_pattern must match eval_pattern exactly")

    def get_mean(self):
        return self.param["intercept"] + self.param["slope"] * self.param['age']

    def get_std(self):
        if self.param["std"] is None:
            raise ValueError("std model values were not defined")
        else:
            return self.param["std"]


class Heijl1987p24d2Model(AgeLinearModel):
    """
    References
    --------------------

    .. [1] Heijl A, Lindgren G, Olsson J. Normal variability of static perimetric threshold values across the central
    visual field. Arch Ophthalmol. 1987;105(11):1544‐1549. doi:10.1001/archopht.1987.01060110090039
    """
    def __init__(self, eval_pattern, age, *args, **kwargs):
        model_pattern = PATTERN_P24D2
        intercept = [29.9,28.65,29.25,28.9,31,31.45,31.65,30.9,31.6,31.7,30.75,33.05,33.8,33.95,33.3,32.25,31.95,31.9,29.7,31.95,34,34,34.8,34.95,34.1,0.0,32.7,30.05,32.75,33.9,34.9,35.15,35.35,35.25,0.0,33.15,31.95,33.05,34.4,33.4,34.05,34.35,32.55,32.1,32.55,32.45,33.25,33.1,33.5,33.25,30.9,31.9,32.7,32.65]
        intercept = np.array(intercept)
        slope = [-0.082,-0.049,-0.075,-0.072,-0.066,-0.063,-0.057,-0.056,-0.072,-0.082,-0.067,-0.073,-0.054,-0.061,-0.064,-0.051,-0.063,-0.08,-0.07,-0.067,-0.066,-0.046,-0.05,-0.055,-0.068,0.0,-0.066,-0.081,-0.075,-0.054,-0.058,-0.049,-0.053,-0.077,0.0,-0.065,-0.073,-0.057,-0.048,-0.036,-0.055,-0.063,-0.049,-0.058,-0.073,-0.055,-0.061,-0.06,-0.06,-0.065,-0.056,-0.068,-0.066,-0.067]
        slope = np.array(slope)
        std = [4.7, 4.2, 3.7, 4.3, 3.4, 3.1, 2.8, 3.0, 2.9, 3.0, 3.4, 3.3, 2.4, 2.3, 2.1, 2.2, 3.5, 3.4, 4.2, 2.7, 2.3, 2.3, 2.0, 2.2, 2.4, -1, 3.7, 5.8, 2.2, 2.3, 1.9, 1.6, 1.8, 2.1, -1, 3.3, 3.0, 2.4, 1.8, 2.4, 2.2, 2.4, 3.9, 3.5, 2.7, 3.4, 2.4, 2.4, 2.3, 2.5, 3.7, 2.4, 2.4, 3.0]
        std = np.array(std)

        super().__init__(eval_pattern, age, model_pattern, intercept, slope, std, *args, **kwargs)


class ConstantModel(Model):
    """
    Returns a pre-specified constant for testing purposes
    """
    def __init__(self, eval_pattern, mean, std, *args, **kwargs):
        super().__init__(eval_pattern, *args, **kwargs)
        self.mean = mean * 1.0
        self.param["std"] = std * 1.0

    def get_mean(self):
        return np.full(shape=len(self.param['eval_pattern']), fill_value=self.mean)

    def get_std(self):
        return np.full(shape=len(self.param['eval_pattern']), fill_value=self.param["std"])


class GrowthPattern:
    def adjust(self, mean, std, estimates):
        return mean, std, estimates


class SimpleGrowthPattern(GrowthPattern):
    def __init__(self):
        self.pattern = {}
        self.agg_fun = np.nanmean

    def adjust(self, mean, std, estimates):
        mean_adjusted = mean.copy()
        for k, v in self.pattern.items():
            offsets = estimates[v] - mean[v]
            if np.isfinite(offsets).any():
                aggregate_offset = self.agg_fun(offsets)
                mean_adjusted[k] += aggregate_offset

        return mean_adjusted, std, estimates


class Simplep24d2QuadrantGrowth(SimpleGrowthPattern):
    def __init__(self):
        super().__init__()
        self.pattern = {
         0: [12], 1: [12], 4: [12], 5: [12], 6: [12],10: [12],11: [12],13: [12],18: [12],19: [12],20: [12],21: [12],22: [12],
         2: [15], 3: [15], 7: [15], 8: [15], 9: [15],14: [15],16: [15],17: [15],23: [15],24: [15],26: [15],
        27: [38],28: [38],29: [38],30: [38],31: [38],36: [38],37: [38],39: [38],44: [38],45: [38],46: [38],50: [38],51: [38],
        32: [41],33: [41],35: [41],40: [41],42: [41],43: [41],47: [41],48: [41],49: [41],52: [41],53: [41],
        }
