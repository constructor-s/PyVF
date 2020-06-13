"""
Testing strategy class, and associated other classes: Normal value model, responder model

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
import os
import logging
try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

import pyvf.resources.saplocmap as saplocmap
from collections import namedtuple

_logger = logging.getLogger(__name__)

def goldmann_diameter(i):
    return np.rad2deg(2.0 * np.arctan2(np.sqrt( (4.0**(i-2.0)) / np.pi), 300.0))

GOLDMANN_0   = goldmann_diameter(0)
GOLDMANN_I   = goldmann_diameter(1)
GOLDMANN_II  = goldmann_diameter(2)
GOLDMANN_III = goldmann_diameter(3)
GOLDMANN_IV  = goldmann_diameter(4)
GOLDMANN_V   = goldmann_diameter(5)

def _load_saplocmap(filename):
    """
    Comment: So apparently reliably and properly reading an asset file within a Python package is quite involved...
    Following https://stackoverflow.com/a/20885799 for now

    Parameters
    ----------
    filename

    Returns
    -------

    """
    _logger.debug("loading: %s", filename)
    dtype = [("xod", np.float32),("yod", np.float32),("loc", np.int32),("size", np.float32),("jmangle", np.float32),("jmslope", np.float32),("region", np.int32)]
    with pkg_resources.open_text(saplocmap, filename) as f:
        return np.loadtxt(f, dtype=dtype, delimiter=",", skiprows=1)

# PATTERN_COLUMNS = ["xod","yod","loc","size","jmangle","jmslope","region"]
#          X_DEGREES, Y_DEGREES, INDEX, GOLDMANN_SIZE, ???, ???, REGION_LABEL_INT
PATTERN_SINGLE = np.array([(3, 3, 0, GOLDMANN_III, 205.37, 0.096369, 1)], dtype=[("xod", np.float32),("yod", np.float32),("loc", np.int32),("size", np.float32),("jmangle", np.float32),("jmslope", np.float32),("region", np.int32)])
PATTERN_P24D2 = _load_saplocmap("saplocmap_p24d2.csv")
PATTERN_P30D2 = _load_saplocmap("saplocmap_p30d2.csv")
PATTERN_P10D2 = _load_saplocmap("saplocmap_p10d2.csv")

STIMULUS_NO_RESPONSE = -1
STIMULUS_NOT_SEEN = 0
STIMULUS_SEEN = 1
STIMULUS_TS_NONE = -1

STIM_XOD = 0
STIM_YOD = 1
STIM_LOC = 2
STIM_SIZE = 3
STIM_THRESHOLD = 4
STIM_RESPONSE = 5
STIM_TSDISP = 6
STIM_TSRESP = 7
XOD = "xod"
YOD = "yod"
LOC = "loc"
SIZE = "size"
RESPONSE = "response"
THRESHOLD = "threshold"
TSDISP = "tsdisp"
TSRESP = "tsresp"
_Stimulus = namedtuple("Stimulus", [XOD,YOD,LOC,SIZE,THRESHOLD,RESPONSE,TSDISP,TSRESP])
class Stimulus(_Stimulus):
    def copy(self, **kwargs):
        new_kwargs = self._asdict()
        new_kwargs.update(kwargs)
        return Stimulus(**new_kwargs)

    @staticmethod
    def to_numpy(data):
        """
        Return a copy of one Stimulus "scalar" or a list of Stimuli as a numpy array that support field access

        Example: Stimulus.to_numpy()


        Parameters
        ----------
        data

        Returns
        -------

        """
        dtype = [(XOD, np.float32), (YOD, np.float32), (LOC, np.int32), (SIZE, np.float32), (THRESHOLD, np.float32),
                 (RESPONSE, np.float32), (TSDISP, np.float32), (TSRESP, np.float32)]
        return np.array(data, dtype=dtype)



class Strategy:
    """
    Base class for strategy, not meant to be used directly

    Steps for using the strategy class
    1. __init__()
    2. stimulus, threshold = get_stimulus_threshold(data1)
    2. while stimulus is not None:
    3.     # Test the stimulus and append to data1
    4.     stimulus, threshold = get_stimulus_threshold(data1)

    data1 : [Stimulus] is a list of Stimuli response records
    stimulus : Stimulus is the next stimulus to be presented to the responder
    threshold : [float] an array of size M, is the current best estimation of threshold (M is the number of SAP locations)

    Design choice: should test_data be a field of this class? Trying to keep this class stateless, which should be
    perfectly fine on modern computers? Is there any process in this that can be vectorized for significant performance
    boost?

    Parameters
    -------------------

    self.param['pattern']: A saplocmap exported from R visualFields package(SAP location map) represented as an M x 6
    matrix. The columns are "xod","yod","loc","size","jmangle","jmslope","region". Currently assume the first three
    columns are present and "loc" is zero-based indexed (the original R package is one-based indexed)

    self.param['blindspot']: A list of zero-based index locations for the bind spots in the pattern; can be an empty
    list

    self.param['model']: Population model object for calculating normal values
    """
    def __init__(self, pattern, blindspot, model, *args, **kwargs):
        self.args = args
        self.param = kwargs
        self.param['pattern'] = pattern
        self.param['blindspot'] = blindspot
        self.param['model'] = model  # type: Model
        self.param['min_db'] = 0
        self.param['max_db'] = 35

    def get_stimulus_threshold(self, data):
        raise NotImplementedError()

    def clip_stimulus_intensity(self, x):
        return np.clip(x, self.param['min_db'], self.param['max_db'])

class DoubleStaircaseStrategy(Strategy):
    def __init__(self, pattern, blindspot, model, step=(4, 2), threshold_func=None, *args, **kwargs):
        """

        Parameters
        ----------
        step
        threshold_func : if not specified, default to DoubleStaircaseStrategy.get_last_seen_threshold
        args
        kwargs
        """
        super().__init__(pattern, blindspot, model, *args, **kwargs)
        self.param["step"] = step
        self.param["repeat"] = False  # Repeat is currently not implemented

        self.param["test_sequence"] = np.arange(len(self.param['pattern']))
        assert len(self.param['pattern']) == len(np.unique(self.param["test_sequence"]))

        if threshold_func is None:
            threshold_func = DoubleStaircaseStrategy.get_last_seen_threshold
        self.param["threshold_func"] = threshold_func

    def get_stimulus_threshold(self, data):
        """
        Terminology:

        When there is no response, we are in the zero-th reversal.

        After the first (indexing: k = 0) response, still in the zero-th reversal.

        If response_k != response_(k-1), response_k and subsequent ones are in the first reversal.

        When another response_j != response_(j-1), response_j is in the second reversal. For a double staircase, this
        is when the staircase ends.


        Parameters
        ----------
        data

        Returns
        -------
        None, threshold if there is no more to be done
        Stimulus, threshold if the next stimulus is selected to be stimulus
        threshold is a list of length M of current threshold estimates
        """
        data = Stimulus.to_numpy(data)
        stimulus = None
        threshold = np.full(len(self.param['pattern']), np.nan)

        # Use m = 1 ... M to index SAP locations
        # right now there is no randomization... just use the test_sequence to rank stimulus test order
        for m in self.param["test_sequence"]:
            # Get subset of relevant data
            data_m = data[data[LOC] == m]  # type: np.ndarray
            data_m = np.sort(data_m, axis=0, order=TSRESP)

            response_sequence = data_m[RESPONSE]
            # We assume no error in the trial threshold implementation,
            # and only look at the trend of seen/not seen to determine staircase state
            reversals, direction = DoubleStaircaseStrategy.get_staircase_stats(response_sequence)

            # Calculate threshold, and, at the same time, produce a stimulus if necessary
            # First check the last presentation response for ceiling or flooring effects
            if (len(data_m) > 0 and
                    data_m[-1][THRESHOLD] >= self.param["max_db"] and
                    data_m[-1][RESPONSE] == STIMULUS_SEEN):
                # Report 1 dB higher than the max. This is just to be symmetric with the flooring case below.
                # I don't know if this is a standard implementation. The 1 dB is chosen arbitrarily.
                threshold[m] = self.param["max_db"] + 1.0
            elif (len(data_m) > 0 and
                    data_m[-1][THRESHOLD] <= self.param["min_db"] and
                    data_m[-1][RESPONSE] == STIMULUS_NOT_SEEN):
                # This is arguable more important than the case above
                # It is my impression that on HFA reports there are some locations reported as 0 and some reported as <0
                # Since min_db is almost always 0 dB, we specify this case as -1 dB, so that this may be reported as <0
                # I don't know if this is a standard implementation. The 1 dB is chosen arbitrarily.
                threshold[m] = self.param["min_db"] - 1.0

            # Handle an edge case - this should never be encountered!
            elif reversals > len(self.param["step"]):
                # More than the desired number of reversals! Invalid state
                raise RuntimeError()

            # Now onto the much more typical cases
            elif reversals == len(self.param["step"]):
                # this location has finished all staircase reversals
                threshold[m] = DoubleStaircaseStrategy.get_last_seen_threshold(data_m)
            else:
                # this location still needs testing
                threshold[m] = DoubleStaircaseStrategy.get_last_seen_threshold(data_m)

                # if no next stimulus has been picked yet
                if stimulus is None:
                    if len(data_m) == 0:
                        # Initial test starts at population mean
                        threshold[m] = self.param["model"].get_mean()[m]
                    else:
                        # Get last threshold and apply the right step size
                        threshold[m] = data_m[-1][THRESHOLD] + self.param["step"][reversals] * direction

                    threshold[m] = self.clip_stimulus_intensity(threshold[m])
                    location = self.param["pattern"][m]
                    stimulus = Stimulus(
                        xod=location[XOD],
                        yod=location[YOD],
                        loc=location[LOC],
                        size=location[SIZE],
                        threshold=threshold[m],
                        response=STIMULUS_NO_RESPONSE,
                        tsdisp=STIMULUS_TS_NONE,
                        tsresp=STIMULUS_TS_NONE
                    )

        return stimulus, threshold

    @staticmethod
    def get_staircase_stats(sequence):
        """

        Parameters
        ----------
        sequence

        Returns
        -------
        (reversals, direction)

        Return the number of reversals that has already happened in the sequence, and which direction is should go next
        indicated by +1 or -1. If sequence is empty, returns 0, 0
        """
        if len(sequence) == 0:
            return 0, 0

        sequence = np.array(sequence, dtype=np.bool)

        # Performance measured with [0, 0, 1, 1, 1, 0]
        # Performance boost by switching to Numpy may be much larger for a long sequence?
        # But we are always dealing with short sequences here.
        # This function is on the order of microseconds anyways

        # Python loop Performance: 0.055 per 10000 runs
        """
        # To simplify logic: assume the sequence starts with False, flip the sequence if this is not the case
        starts_with_true = sequence[0] == True
        if starts_with_true:
            sequence = ~sequence

        reversals = 0
        for i in range(1, len(sequence)):  # We already made sure sequence[0] is False
            if sequence[i]:
                reversals += 1
                sequence = ~sequence

        # Assuming the sequence started with False (not seen),
        # then for all all even reversals (0, 2) the direction is going down <- the direction is -1
        # for all odd reversals (1) the direction is going up <- the modulus is 1
        direction = (reversals % 2) * 2 - 1
        # If the assumption above is not valid, then we want to flip -1 and 1,
        # so that going up is represented by 1 and going down is represented by 0
        if starts_with_true:
            direction = -direction
        """

        # Vectorized version
        # Performance using np.diff: 0.096 per 10000 runs
        # reversals = np.count_nonzero(np.diff(sequence))
        # Performance using bitwise XOR: 0.040 per 10000 runs
        reversals = np.count_nonzero(sequence[1:] ^ sequence[:-1])  # bitwise XOR
        if sequence[0]:
            direction = 1 - (reversals % 2) * 2
        else:
            direction = (reversals % 2) * 2 - 1

        return reversals, direction

    @staticmethod
    def get_last_seen_threshold(data):
        """
        data1[THRESHOLD] are trial thresholds
        data1[RESPONSE] are response results

        If all responses are not seen, then return the last threshold.
        If the data1 list is empty, then return nan

        Parameters
        ----------
        data

        Returns
        -------

        """
        if len(data) == 0:
            return np.nan

        idx_ = np.nonzero(data[RESPONSE] == STIMULUS_SEEN)
        idx_ = idx_[0]
        if len(idx_) == 0:
            idx = -1
        else:
            idx = idx_[-1]
        return data[THRESHOLD][idx]

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


class Responder:
    def get_response(self, stimulus):
        raise NotImplementedError()

class PerfectResponder(Responder):
    """
    Responder with characteristics of a perfect step function with step at the true threshold
    """
    def __init__(self, true_threshold):
        """

        Parameters
        ----------
        true_threshold

        A list of actual threshold. The list must be of length M where M is the length of the pattern used by the
        strategy instance. This responder simply looks up the loc (index) of the stimulus and find the corresponding
        threshold.
        """
        self.threshold = np.array(true_threshold)

    def get_response(self, stimulus):
        if stimulus.threshold < self.threshold[stimulus.loc]:
            # Seen with zero response time
            return stimulus.copy(**{RESPONSE: STIMULUS_SEEN, TSRESP: stimulus[STIM_TSDISP]})
        else:
            # Not seen with zero response time
            return stimulus.copy(**{RESPONSE: STIMULUS_NOT_SEEN, TSRESP: stimulus[STIM_TSDISP]})
