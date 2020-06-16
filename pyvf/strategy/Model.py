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