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
        self.param['max_db'] = 40

    def get_stimulus_threshold(self, data):
        raise NotImplementedError()

    def clip_stimulus_intensity(self, x):
        return np.clip(x, self.param['min_db'], self.param['max_db'])

class DoubleStaircaseStrategy(Strategy):
    def __init__(self, pattern, blindspot, model, step=(4, 2), threshold_func=None, repeat_threshold=4.0, *args, **kwargs):
        """

        Parameters
        ----------
        step
        threshold_func : if not specified, default to DoubleStaircaseStrategy.get_last_seen_threshold or the mean of the two determinations (each determined by the last seen threshold)
        repeat_threshold : float. This is to implement the behavior of HFA Full Threshold: If the difference between the first determination and the expected normal value is more than 4 dB, repeat the staircase at the new estimate.
        args
        kwargs
        """
        super().__init__(pattern, blindspot, model, *args, **kwargs)
        self.param["step"] = step
        self.param["repeat_threshold"] = repeat_threshold  # Repeat is currently not implemented

        self.param["test_sequence"] = np.arange(len(self.param['pattern']))
        assert len(self.param['pattern']) == len(np.unique(self.param["test_sequence"]))

        if threshold_func is None:
            threshold_func = DoubleStaircaseStrategy.get_last_seen_threshold_or_mean
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
        stimulus = None  # The next stimulus to be presented
        threshold = np.full(len(self.param['pattern']), np.nan)  # The current best estimate of point thresholds

        # Use m = 1 ... M to index SAP locations
        # right now there is no randomization... just use the test_sequence to rank stimulus test order
        for m in self.param["test_sequence"]:
            # Convenient lambda function for generating next stimulus
            location = self.param["pattern"][m]
            get_new_stimulus_at = lambda db: Stimulus(
                xod=location[XOD],
                yod=location[YOD],
                loc=location[LOC],
                size=location[SIZE],
                threshold=self.clip_stimulus_intensity(db),
                response=STIMULUS_NO_RESPONSE,
                tsdisp=STIMULUS_TS_NONE,
                tsresp=STIMULUS_TS_NONE
            )

            # Get subset of relevant data_m
            data_m = data[data[LOC] == m]  # type: np.ndarray
            # Data should already be sorted, but just be sure
            data_m = np.sort(data_m, axis=0, order=TSRESP)
            # Extract the response sequence column, e.g. [0, 0, 1, 0, 1, 0, 0, 1]
            response_sequence = data_m[RESPONSE]
            # We assume no error in the trial threshold implementation,
            # and only look at the trend of seen/not seen to determine staircase state
            # e.g. reversals_list = [0, 0, 1, 2, 0, 1, 1, 2] where in this example,
            # after the first four responses there is a repeated double determination
            if len(data_m) == 0:
                force_terminate = ()
            else:
                force_terminate = ((data_m[THRESHOLD] >= self.param["max_db"]) &
                                   (data_m[RESPONSE] == STIMULUS_SEEN))
                force_terminate |= ((data_m[THRESHOLD] <= self.param["min_db"]) &
                                    (data_m[RESPONSE] == STIMULUS_NOT_SEEN))
            reversals_list, repeats_list = DoubleStaircaseStrategy.get_staircase_stats(response_sequence,
                                                                         step=self.param["step"],
                                                                         force_terminate=force_terminate)

            # # The direction variable stores which way (math sign) the next step should go
            # if len(response_sequence) == 0:
            #     direction = 0  # The sign is really not important for the first one anyways
            # else:
            #     # mapping: seen = 1 -> 1, next stimulus should have a higher dB,
            #     #      not seen = 0 -> -1
            #     direction = response_sequence[-1] * 2 - 1

            # Calculate threshold, and, at the same time, produce a stimulus if necessary
            # If no test has been done yet and we do not already know what to test next (stimulus is None)
            if len(data_m) == 0:
                threshold[m] = self.param["model"].get_mean()[m]
                if stimulus is None:
                    stimulus = get_new_stimulus_at(threshold[m])
                else:
                    pass

            # Check the last presentation response for ceiling or flooring effects,
            # and this is also termination of the staircase
            elif (data_m[-1][THRESHOLD] >= self.param["max_db"] and
                  data_m[-1][RESPONSE] == STIMULUS_SEEN):
                # Report 0.01 dB higher than the max. This is just to be symmetric with the flooring case below.
                # I don't know if this is a standard implementation. The 0.01 dB is chosen arbitrarily.
                threshold[m] = self.param["max_db"] + 0.01
                if repeats_list[-1] == 0 and abs(threshold[m] - self.param["model"].get_mean()[m]) > self.param["repeat_threshold"]:
                    stimulus = get_new_stimulus_at(threshold[m])
                else:
                    pass  # We are done here

            elif (data_m[-1][THRESHOLD] <= self.param["min_db"] and
                  data_m[-1][RESPONSE] == STIMULUS_NOT_SEEN):
                # This is arguable more important than the case above. It is my impression that on HFA reports there
                # are some locations reported as 0 and some reported as <0. Since min_db is almost always 0 dB,
                # we specify this case as -0.01 dB, so that this may be reported as <0.
                # I don't know if this is a standard implementation. The 0.01 dB is chosen arbitrarily.
                threshold[m] = self.param["min_db"] - 0.01
                if repeats_list[-1] == 0 and abs(threshold[m] - self.param["model"].get_mean()[m]) > self.param["repeat_threshold"]:
                    stimulus = get_new_stimulus_at(threshold[m])
                else:
                    pass  # We are done here

            # Handle an edge case - this should never be encountered!
            elif reversals_list[-1] > len(self.param["step"]):
                # More than the desired number of reversals! Invalid state
                raise RuntimeError()

            # Now onto the much more typical cases
            # We have finished a staircase
            elif reversals_list[-1] == len(self.param["step"]):
                # this location has finished all staircase reversals
                threshold[m] = self.param["threshold_func"](self, data_m)
                if repeats_list[-1] == 0 and abs(threshold[m] - self.param["model"].get_mean()[m]) > self.param["repeat_threshold"]:
                    stimulus = get_new_stimulus_at(threshold[m])
                else:
                    pass  # We are done here

            # We need to continue the staircase
            else:
                # this location still needs testing
                threshold[m] = self.param["threshold_func"](self, data_m)

                # if no next stimulus has been picked yet
                if stimulus is None:
                    # Get last threshold and apply the right step size
                    direction = +1 if data_m[-1][RESPONSE] == STIMULUS_SEEN else -1

                    threshold[m] = data_m[-1][THRESHOLD] + self.param["step"][reversals_list[-1]] * direction
                    threshold[m] = self.clip_stimulus_intensity(threshold[m])

                    stimulus = get_new_stimulus_at(db=threshold[m])
                else:
                    pass  # The algorithm already has another stimulus to test, nothing to do here

        return stimulus, threshold

    @staticmethod
    def get_staircase_stats(sequence, step=None, force_terminate=None, _rev_start=0, _rep_start=0):
        """

        Parameters
        ----------
        sequence
        step

        force_terminate : np.ndarray of booleans of the same length as sequence (optional)
        wherever force_terminate==True, consider this the last element in this staircase, and
        start recounting the staircase from zero starting from the next element

        Returns
        -------
        reversals : np.ndarray
        repeats : np.ndarray

        Return the number of reversals that has already happened in the sequence for each response in sequence
        If sequence is empty, returns [], []
        If step is supplied, then resets the number of reversals after the first staircase is completed.
        For example, for a double staircase (len(step)==2), then reversals will be mapped
        from [0, 1, 2, 2, 3, 4] to [0, 1, 2, 0, 1, 2]
        from [0, 1, 2, 3, 4, 5] to [0, 1, 2, 0, 1, 2] also
        The discrepancy between [..., 2, 3] or [..., 2, 2] will be due to whether the first response of the repetition series is
        the same as the last response in the initial series. However, since the repetition series is treated as
        independent from the initial series, both of these cases should be reset to zero reversals at the start of the
        repetition
        """
        if len(sequence) == 0:
            return np.array([]), np.array([])
        elif len(sequence) == 1:
            return np.array([_rev_start]), np.array([_rep_start])
        else:
            sequence = np.array(sequence, dtype=np.bool)

            if force_terminate is None:
                force_terminate = np.zeros_like(sequence, dtype=np.bool)
            else:
                assert len(force_terminate) == len(sequence)
                force_terminate = np.array(force_terminate, dtype=np.bool)

            # New solution writing this as a recursion problem
            # 0.8322514 sec for 10000 iterations

            # General case
            # This is the key of recursion
            if force_terminate[0] or (step is not None and _rev_start == len(step)):  # Current staircase terminated
                new_rev_start = 0
                new_rep_start = _rep_start + 1
            elif sequence[1] != sequence[0]:  # Next response is different from current -> there is a reversal
                new_rev_start = _rev_start + 1
                new_rep_start = _rep_start
            else:  # Nothing exciting is happening
                new_rev_start = _rev_start
                new_rep_start = _rep_start

            rest_reversals, rest_repeats = DoubleStaircaseStrategy.get_staircase_stats(
                sequence=sequence[1:], step=step, force_terminate=force_terminate[1:],
                _rev_start=new_rev_start, _rep_start=new_rep_start
            )
            reversals = [_rev_start]
            reversals.extend(rest_reversals)
            repeats = [_rep_start]
            repeats.extend(rest_repeats)
            # reversals = np.empty_like(sequence, dtype=np.int32)
            # reversals[0] = _rev_start
            # reversals[1:] = rest_reversals

            return np.array(reversals), np.array(repeats)

    @staticmethod
    def _process_repeated_staircase(x, n):
        """
        Deprecated

        mapping:
        - case A: from [0, 1, 2, 2, 3, 4, 4, 5, 6] to [0, 1, 2, 0, 1, 2, 0, 1, 2]
        - case B: from [0, 1, 2, 3, 4, 5] to [0, 1, 2, 0, 1, 2]

        Parameters
        ----------
        x : np.ndarray
        n : int

        Returns
        -------

        """
        if len(x) < 2:
            return x

        x1 = x == n
        x2 = x >= n
        prod = x1[:-1] & x2[1:]
        idx = np.where(prod)[0]
        while len(idx) > 0:
            i = idx[0] + 1

            x = np.array(x)
            x[i:] -= x[i]

            x1 = x == n
            x2 = x >= n
            prod = x1[:-1] & x2[1:]
            idx = np.where(prod)[0]

        return x

    def get_last_seen_threshold(self, data_m):
        """
        data_m[THRESHOLD] are trial thresholds
        data_m[RESPONSE] are response results

        If all responses are not seen, then return the last threshold.
        If the data_m list is empty, then return nan

        Parameters
        ----------
        data_m : array_like

        Returns
        -------
        threshold : float
            last seen threshold

        """
        if len(data_m) == 0:
            return np.nan

        idx_ = np.nonzero(data_m[RESPONSE] == STIMULUS_SEEN)
        idx_ = idx_[0]
        if len(idx_) == 0:
            idx = -1
        else:
            idx = idx_[-1]
        return data_m[THRESHOLD][idx]

    def get_last_seen_threshold_or_mean(self, data_m):
        # Extract the response sequence column, e.g. [0, 0, 1, 0, 1, 0, 0, 1]
        response_sequence = data_m[RESPONSE]
        # We assume no error in the trial threshold implementation,
        # and only look at the trend of seen/not seen to determine staircase state
        # e.g. reversals_list = [0, 0, 1, 2, 0, 1, 1, 2] where in this example,
        # after the first four responses there is a repeated double determination
        if len(data_m) == 0:
            force_terminate = ()
        else:
            force_terminate = ((data_m[THRESHOLD] >= self.param["max_db"]) &
                               (data_m[RESPONSE] == STIMULUS_SEEN))
            force_terminate |= ((data_m[THRESHOLD] <= self.param["min_db"]) &
                                (data_m[RESPONSE] == STIMULUS_NOT_SEEN))
        reversals_list, repeats_list = DoubleStaircaseStrategy.get_staircase_stats(response_sequence,
                                                                                   step=self.param["step"],
                                                                                   force_terminate=force_terminate)
        thresholds = [self.get_last_seen_threshold(data_m[repeats_list == i]) for i in np.unique(repeats_list)]
        return np.mean(thresholds)
