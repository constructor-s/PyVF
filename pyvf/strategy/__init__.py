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
from operator import mul
from functools import reduce

from pyvf.ext.methodtools import lru_cache
from pyvf.stats import rv_histogram2
from pyvf.stats.pos import pos_ramp
from pyvf.strategy.GrowthPattern import EmptyGrowthPattern

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
PATTERN_DOUBLE = np.array([(3, 3, 0, GOLDMANN_III, 205.37, 0.096369, 1), (3,-3,1,GOLDMANN_III,162.98,-0.091919,1)], dtype=[("xod", np.float32),("yod", np.float32),("loc", np.int32),("size", np.float32),("jmangle", np.float32),("jmslope", np.float32),("region", np.int32)])
PATTERN_P24D2 = _load_saplocmap("saplocmap_p24d2.csv")
PATTERN_P30D2 = _load_saplocmap("saplocmap_p30d2.csv")
PATTERN_P10D2 = _load_saplocmap("saplocmap_p10d2.csv")

PATTERN_P24D2_OS = PATTERN_P24D2.copy(); PATTERN_P24D2_OS["xod"] = -1 * PATTERN_P24D2_OS["xod"]
PATTERN_P30D2_OS = PATTERN_P30D2.copy(); PATTERN_P30D2_OS["xod"] = -1 * PATTERN_P30D2_OS["xod"]
PATTERN_P10D2_OS = PATTERN_P10D2.copy(); PATTERN_P10D2_OS["xod"] = -1 * PATTERN_P10D2_OS["xod"]

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
MULTI = "multi"  # Set to integer more than 1 for Multiple-Stimulus Perimetry
_Stimulus = namedtuple("Stimulus", [XOD,YOD,LOC,SIZE,THRESHOLD,RESPONSE,TSDISP,TSRESP,MULTI],
                       defaults=[np.nan,np.nan,np.nan,GOLDMANN_III,np.nan,STIMULUS_NOT_SEEN,np.nan,np.nan,1])
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
                 (RESPONSE, np.float32), (TSDISP, np.float32), (TSRESP, np.float32), (MULTI, np.int32)]
        return np.array(data, dtype=dtype)


class Strategy:
    """
    Base class for strategy, not meant to be used directly

    Steps for using the strategy class
    1. __init__()
    2. stimulus, center = get_stimulus_threshold(data1)
    2. while stimulus is not None:
    3.     # Test the stimulus and append to data1
    4.     stimulus, center = get_stimulus_threshold(data1)

    data1 : [Stimulus] is a list of Stimuli response records
    stimulus : Stimulus is the next stimulus to be presented to the responder
    center : [float] an array of size M, is the current best estimation of center (M is the number of SAP locations)

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
    def __init__(self, pattern, blindspot, model, rng=None, *args, **kwargs):
        """

        Parameters
        ----------
        pattern
        blindspot
        model
        rng : np.random.RandomState or Object, to initialize the random number generator
        args
        kwargs
        """
        self.args = args
        self.param = kwargs
        self.param['pattern'] = pattern
        self.param['blindspot'] = blindspot
        self.param['model'] = model  # type: Model
        self.param['min_db'] = 0
        self.param['max_db'] = 40

        if isinstance(rng, np.random.RandomState):
            self.rng = rng
        else:
            self.rng = np.random.RandomState(rng)

        self.extra_data = {}

    def get_stimulus_threshold(self, data):
        raise NotImplementedError()

    def clip_stimulus_intensity(self, x):
        return np.clip(x, self.param['min_db'], self.param['max_db'])

    def get_new_stimulus_at(self, db, location):
        return Stimulus(
            xod=location[XOD],
            yod=location[YOD],
            loc=location[LOC],
            size=location[SIZE],
            threshold=self.clip_stimulus_intensity(db),
            response=STIMULUS_NO_RESPONSE,
            tsdisp=STIMULUS_TS_NONE,
            tsresp=STIMULUS_TS_NONE
        )

    @staticmethod
    def trial2pos_ramp(x, center, fn, fp, width, seen, multiplicity=1):
        """

        Parameters
        ----------
        x : array_like
            Evaluation points

        multiplicity : int
            For Single Stimulus Perimetry, this is 1.
            For Multiple Stimulus Perimetry, such as two stimuli presented simultaneously for
            one yes-no responds, this is 2. The PoS is adjusted assuming the PoS curves for the
            two stimuli are independent (i.e. if there is 50% responding to stimulus 1 and 50%
            responding to stimulus 2, then we assume there is 75% responding when both are presented
            simultaneously)


        Returns
        -------
        y : array_like

        """
        if multiplicity == 1:
            if seen == STIMULUS_SEEN:
                yl = fp
                yr = 1 - fn
            else:  # STIMULUS_NOT_SEEN
                yl = 1 - fp
                yr = fn
        elif multiplicity == 2:
            if seen == STIMULUS_SEEN:
                yl = fp + (1 - fn)
                yr = (1 - fn) * 2
            else:  # STIMULUS_NOT_SEEN
                yl = (1 - fp) + fn
                yr = fn + fn
        else:
            raise NotImplementedError()
        return pos_ramp(x=x, center=center, yl=yl, yr=yr, width=width)


class DoubleStaircaseStrategy(Strategy):
    def __init__(self, pattern, blindspot, model, step=(4, 2), threshold_func=None, repeat_threshold=4.0, test_sequence=None, *args, **kwargs):
        """

        Parameters
        ----------
        step
        threshold_func : if not specified, default to DoubleStaircaseStrategy.get_last_seen_threshold or the mean of the two determinations (each determined by the last seen center)
        repeat_threshold : float. This is to implement the behavior of HFA Full Threshold: If the difference between the first determination and the expected normal value is more than 4 dB, repeat the staircase at the new estimate.
        args
        kwargs
        """
        super().__init__(pattern=pattern, blindspot=blindspot, model=model, *args, **kwargs)
        self.param["step"] = step
        self.param["repeat_threshold"] = repeat_threshold

        self.param["test_sequence"] = test_sequence if test_sequence is not None else np.arange(len(self.param['pattern']))
        assert len(self.param["pattern"]) == len(np.unique(self.param["test_sequence"]))

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
        None, center if there is no more to be done
        Stimulus, center if the next stimulus is selected to be stimulus
        center is a list of length M of current center estimates
        """
        data = Stimulus.to_numpy(data)
        stimulus = None  # The next stimulus to be presented
        threshold = np.full(len(self.param['pattern']), np.nan)  # The current best estimate of point thresholds

        model_mean = self.param["model"].get_mean()

        # Use m = 1 ... M to index SAP locations
        # right now there is no randomization... just use the test_sequence to rank stimulus test order
        for m in self.param["test_sequence"]:
            # Convenient lambda function for generating next stimulus
            location = self.param["pattern"][m]

            # Get subset of relevant data_m
            data_m = data[data[LOC] == m]  # type: np.ndarray
            # Data should already be sorted, but just be sure
            data_m = np.sort(data_m, axis=0, order=TSRESP)
            # Extract the response sequence column, e.g. [0, 0, 1, 0, 1, 0, 0, 1]
            response_sequence = data_m[RESPONSE]
            # We assume no error in the trial center implementation,
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
                force_terminate = self.process_additional_termination_rules(term=force_terminate, data_m=data_m, center=model_mean[m])

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

            # Calculate center, and, at the same time, produce a stimulus if necessary
            # If no test has been done yet and we do not already know what to test next (stimulus is None)
            if len(data_m) == 0:
                threshold[m] = model_mean[m]
                if stimulus is None:
                    stimulus = self.get_new_stimulus_at(threshold[m], location=location)
                else:
                    pass

            # Check the last presentation response for ceiling or flooring effects,
            # and this is also termination of the staircase
            elif force_terminate[-1]:
                threshold[m] = self.param["threshold_func"](self, data_m=data_m, term=force_terminate, center=model_mean[m])
                if repeats_list[-1] == 0 and abs(threshold[m] - model_mean[m]) > self.param["repeat_threshold"]:
                    stimulus = self.get_new_stimulus_at(threshold[m], location=location)
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
                threshold[m] = self.param["threshold_func"](self, data_m=data_m, term=force_terminate, center=model_mean[m])
                if repeats_list[-1] == 0 and abs(threshold[m] - model_mean[m]) > self.param["repeat_threshold"]:
                    stimulus = self.get_new_stimulus_at(threshold[m], location=location)
                else:
                    pass  # We are done here

            # We need to continue the staircase
            else:
                # this location still needs testing
                threshold[m] = self.param["threshold_func"](self, data_m=data_m, term=force_terminate, center=model_mean[m])

                # if no next stimulus has been picked yet
                if stimulus is None:
                    # Get last center and apply the right step size
                    direction = +1 if data_m[-1][RESPONSE] == STIMULUS_SEEN else -1

                    threshold[m] = data_m[-1][THRESHOLD] + self.param["step"][reversals_list[-1]] * direction
                    threshold[m] = self.clip_stimulus_intensity(threshold[m])

                    stimulus = self.get_new_stimulus_at(db=threshold[m], location=location)
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
        if len(sequence) >= 2:
            # sequence = np.array(sequence, dtype=np.bool)

            # if force_terminate is None:
            #     force_terminate = np.zeros_like(sequence, dtype=np.bool)
            # else:
                # assert len(force_terminate) == len(sequence)
                # force_terminate = np.array(force_terminate, dtype=np.bool)

            # New solution writing this as a recursion problem
            # 0.8322514 sec for 10000 iterations

            # General case
            # This is the key of recursion
            if (step is not None and _rev_start == len(step)) or (force_terminate is not None and force_terminate[0]):  # Current staircase terminated
                new_rev_start = 0
                new_rep_start = _rep_start + 1
            elif sequence[1] != sequence[0]:  # Next response is different from current -> there is a reversal
                new_rev_start = _rev_start + 1
                new_rep_start = _rep_start
            else:  # Nothing exciting is happening
                new_rev_start = _rev_start
                new_rep_start = _rep_start

            # rest_reversals, rest_repeats = DoubleStaircaseStrategy.get_staircase_stats(
            #     sequence=sequence[1:], step=step, force_terminate=force_terminate[1:],
            #     _rev_start=new_rev_start, _rep_start=new_rep_start
            # )
            # reversals = [_rev_start]
            # reversals.extend(rest_reversals)
            # repeats = [_rep_start]
            # repeats.extend(rest_repeats)
            # return np.array(reversals), np.array(repeats)

            # reversals = np.empty_like(sequence, dtype=np.int32)
            # reversals[0] = _rev_start
            # reversals[1:] = rest_reversals
            # repeats = np.empty_like(sequence, dtype=np.int32)
            # repeats[0] = _rep_start
            # repeats[1:] = rest_repeats
            # return reversals, repeats

            rest = DoubleStaircaseStrategy.get_staircase_stats(
                sequence=sequence[1:], step=step,
                force_terminate=force_terminate[1:] if force_terminate is not None else None,
                _rev_start=new_rev_start, _rep_start=new_rep_start
            )
            ret = np.empty( (2, len(sequence)) , dtype=np.int32)
            ret[0, 0] = _rev_start
            ret[1, 0] = _rep_start
            ret[:, 1:] = rest
            # Faster than concatenate
            # %timeit -n 1000 np.concatenate( (((2,),(2,)), a) , axis=1)
            # -> 4.26 µs ± 1.17 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
            # %timeit -n 10000 b = np.empty((2, 11)); b[0, 0] = 2; b[1, 0] = 2; b[:, 1:] = a
            # -> 2.03 µs ± 566 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)

            return ret
        elif len(sequence) == 1:
            # return np.array([_rev_start]), np.array([_rep_start])  # 1.65 µs ± 48.6 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
            return np.array( ((_rev_start, ), (_rep_start, )) )  # 819 ns ± 21.7 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
        else:  # len(sequence) == 0:
            # return np.array([]), np.array([])  # 1.78 µs ± 261 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
            return np.empty((2, 0), dtype=np.int32)  # 699 ns ± 30.8 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)


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

    def get_last_seen_threshold(self, data_m, *args, **kwargs):
        """
        data_m[THRESHOLD] are trial thresholds
        data_m[RESPONSE] are response results

        If all responses are not seen, then return the last center.
        If the data_m list is empty, then return nan

        Parameters
        ----------
        data_m : array_like

        Returns
        -------
        center : float
            last seen center

        """
        if len(data_m) == 0:
            return np.nan

        idx_ = np.nonzero(data_m[RESPONSE] == STIMULUS_SEEN)
        idx_ = idx_[0]
        if len(idx_) == 0:  # If all stimulus were not seen, return the lowest
            return data_m[THRESHOLD].min()
        else:
            idx = idx_[-1]
            return data_m[THRESHOLD][idx]

    def get_last_seen_threshold_or_mean(self, data_m, *args, **kwargs):
        # Extract the response sequence column, e.g. [0, 0, 1, 0, 1, 0, 0, 1]
        response_sequence = data_m[RESPONSE]
        # We assume no error in the trial center implementation,
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

    def process_additional_termination_rules(self, term, data_m, center):
        return term


class StaircaseQuestStrategy(DoubleStaircaseStrategy):
    def __init__(self, pattern, blindspot, model, repeat_threshold=12.0, term_erf=0.69, *args, **kwargs):
        super().__init__(pattern, blindspot, model, repeat_threshold=repeat_threshold, threshold_func=StaircaseQuestStrategy.get_threshold_from_pdfs, *args, **kwargs)

        self.param["term_erf"] = term_erf
        # Parameters determining shape of probability of seeing curve
        self.param["pos_fp"] = 0.05
        self.param["pos_fn"] = 0.05
        self.param["pos_width"] = 4.0

        # Initialize
        import pyvf.resources.turpin2003 as turpin2003
        with pkg_resources.open_text(turpin2003, "abnormal_pdf.csv") as f:
            abnormal_bins, abnormal_height = np.loadtxt(f, dtype=np.float32, delimiter=",", skiprows=0).T
            abnormal_bins = np.concatenate( (abnormal_bins - 0.5, [abnormal_bins[-1] + 0.5]) )
        with pkg_resources.open_text(turpin2003, "normal_pdf.csv") as f:
            normal_bins, normal_height = np.loadtxt(f, dtype=np.float32, delimiter=",", skiprows=0).T
            normal_bins = np.concatenate((normal_bins - 0.5, [normal_bins[-1] + 0.5]))
        assert np.allclose(abnormal_bins, normal_bins), "The bins of abnormal and normal PDFs must match"
        assert np.allclose(np.sum(abnormal_height), np.sum(normal_height), rtol=0.01, atol=0.01), "Input PDFs must sum to 1"

        # Fields related to probabilistic distribution
        self.refine_n = 10
        self.hist_abnormal = rv_histogram2(histogram=(abnormal_height, abnormal_bins)).refined(self.refine_n)
        self.hist_normal   = rv_histogram2(histogram=(  normal_height,   normal_bins)).refined(self.refine_n)
        self.center_normal = self.hist_normal.mode()
        self.epsilon = self.hist_normal.height.min()  # 1e-3

        self._pdf_updates_cache = {}
        self._pdf_updates_cache_enabled = True
        self._pdf_updates_cache_hits = 0
        self._pdf_updates_cache_misses = 0

    def process_additional_termination_rules(self, term, data_m, center):
        term, threshold, normal, abnormal = self._pdf_updates(term=term, data_m=data_m, center=center)
        return term

    @staticmethod
    def erf(std, mode):
        """
        The SITA algorithm uses the error-related factor (ERF)4 to deter-
        mine whether the variance of either pf is sufficiently narrow to termi- nate the staircase procedure,
        where ERF := 0.19 + sqrt(variance) - 3/70*mode = 0.19 + std - 3/70*mode
        Parameters
        ----------
        std : float | array_like
        mode : float | array_like

        Returns
        -------
        erf : float | array_like
            SITA error-related factor

        References
        --------------
        .. [1] Turpin, A., McKendrick, A. M., Johnson, C. A., & Vingrys, A. J. (2003).
        Properties of Perimetric Threshold Estimates from Full Threshold, ZEST, and SITA-like Strategies,
        as Determined by Computer Simulation. Investigative Ophthalmology and Visual Science, 44(11), 4787–4795.
        https://doi.org/10.1167/iovs.03-0023
        """
        return 0.19 + std - 3.0 / 70.0 * mode

    def get_threshold_from_pdfs(self, data_m, term, center, *args, **kwargs):
        """
        SQ/SITA returned the most likely mode of the two pfs used in the procedure

        The quote above leaves some room for interpretation. So we interpret it as:
        0) Normalize both PDFs (mathematically speaking PDFs are always normalized)
        1) Find the two modes of the two PDFs and the corresponding probability at that dB
        2) Report the mode with the higher height

        Parameters
        ----------
        data_m

        Returns
        -------

        """
        term, threshold, normal, abnormal = self._pdf_updates(term=term, data_m=data_m, center=center)
        return threshold

    def _pdf_updates(self, term, data_m, center, debug=True):
        """



        Parameters
        ----------
        term
        data_m
        center

        Returns
        -------
        term
        threshold
        normal
        abnormal

        Notes
        ----------—
        Benchmark:
        Iterative version: 13864 ms (81.4%)
        Recursive version: 16290 ms (90.6%)
        Recursive cached version: 1501 ms (47.9%)
        """
        if self._pdf_updates_cache_enabled:
            key = (term.tobytes(), data_m.tobytes(), center)
            if key in self._pdf_updates_cache:
                self._pdf_updates_cache_hits += 1
                # _logger.info("Cache hits %d", self._pdf_updates_cache_hits)
                return self._pdf_updates_cache[key]
            else:
                self._pdf_updates_cache_misses += 1
                # _logger.info("Cache miss %d", self._pdf_updates_cache_misses)

        if len(data_m) >= 1:
            term = np.array(term)
            term[:-1], threshold, normal, abnormal = self._pdf_updates(term=term[:-1], data_m=data_m[:-1], center=center)

            pos = StaircaseQuestStrategy.trial2pos_ramp(normal.bins[:-1], data_m[-1][THRESHOLD], self.param["pos_fn"],
                                                        self.param["pos_fp"], self.param["pos_width"], data_m[-1][RESPONSE])
            trial = rv_histogram2(histogram=(pos, normal.bins))

            normal = normal * trial
            abnormal = abnormal * trial

            # Handle a very edge case: e.g. starting from 32.501 with responder at 40.001
            # thresholds: 32.501, 36.501, 40.0 (terminate), -0.45, ...
            # response: 1, 1, 1, 1, ...
            # mode of normal (p): 34.45 (0.24), 36.45 (0.32), 38.45 (0.30) (terminate),
            # mode of abnormal (p): -0.45 (0.31), -0.45 (0.31), -0..45 (0.31) (terminate),
            # Basically this can occur in abnormally high sensitivity, which really only occurs in simulations.
            # Thus pdf_abnormal_threshold is used
            threshold = StaircaseQuestStrategy._get_likely_mode(normal, abnormal, pdf_abnormal_threshold=0.33)

            if debug:
                mode1 = normal.mode()
                p1 = normal.pdf(mode1)
                mode2 = abnormal.mode()
                p2 = abnormal.pdf(mode2)
                _logger.debug("normal(%g)=%g, abnormal(%g)=%g", mode1, p1, mode2, p2)

            # Staircase termination criteria is already included in term
            # Add ERF termination criteria
            if (StaircaseQuestStrategy.erf(normal.std(), normal.mode()) <= self.param["term_erf"] or
                    StaircaseQuestStrategy.erf(abnormal.std(), abnormal.mode()) <= self.param["term_erf"]):
                term[-1] = True

            if term[-1]:
                # If there are any remaining data to be processed, we assume that is from another repeated trial
                # So we have to reset the PDFs here
                normal = self.hist_normal.roll(shift=int(round((threshold - self.center_normal) * self.refine_n)),
                                               fill_value=self.epsilon)
                abnormal = self.hist_abnormal

            if self._pdf_updates_cache_enabled:
                term.setflags(write=False)
                self._pdf_updates_cache[key] = term, threshold, normal, abnormal

            return term, threshold, normal, abnormal

        else:  # len(data_m) == 0
            normal = self.hist_normal.roll(shift=int(round((center - self.center_normal) * self.refine_n)),
                                           fill_value=self.epsilon)
            abnormal = self.hist_abnormal

            if self._pdf_updates_cache_enabled:
                term.setflags(write=False)
                self._pdf_updates_cache[key] = term, center, normal, abnormal

            return term, center, normal, abnormal

    @staticmethod
    def _get_likely_mode(pdf_normal, pdf_abnormal, pdf_abnormal_threshold=0.0):
        """

        Parameters
        ----------
        pdf_normal : rv_histogram2
        pdf_abnormal : rv_histogram2
        pdf_abnormal_threshold : float
            If the abnormal pdf is lower than this number, then reject the abnormal result and always take the normal result

        Returns
        -------

        """
        mode1 = pdf_normal.mode()
        p1 = pdf_normal.pdf(mode1)

        mode2 = pdf_abnormal.mode()
        p2 = pdf_abnormal.pdf(mode2)

        return mode1 if (p1 > p2 or p2 < pdf_abnormal_threshold) else mode2


class ZestStrategy(Strategy):
    def __init__(self, pattern, blindspot, model, term_std, growth_pattern=None, *args, **kwargs):
        """
        Zippy Estimation by Sequential Testing (ZEST)

        Parameters
        ----------
        pattern
        blindspot
        model
        term_std : float, termination cutoff standard deviation
        growth_pattern : GrowthPattern, pre-defined algorithm of inferring thresholds from seed points
        args
        kwargs
        """
        super().__init__(pattern=pattern, blindspot=blindspot, model=model, *args, **kwargs)

        self.growth_pattern = growth_pattern if growth_pattern is not None else EmptyGrowthPattern()

        self.param["term_std"] = term_std
        self.param["normal2abnormal_ratio"] = 4.0 / 1.0

        # Parameters determining shape of probability of seeing curve
        self.param["pos_fp"] = 0.05
        self.param["pos_fn"] = 0.05
        self.param["pos_width"] = 4.0

        # Initialize
        import pyvf.resources.turpin2003 as turpin2003
        with pkg_resources.open_text(turpin2003, "abnormal_pdf.csv") as f:
            abnormal_bins, abnormal_height = np.loadtxt(f, dtype=np.float32, delimiter=",", skiprows=0).T
            abnormal_bins = np.concatenate( (abnormal_bins - 0.5, [abnormal_bins[-1] + 0.5]) )
        with pkg_resources.open_text(turpin2003, "normal_pdf.csv") as f:
            normal_bins, normal_height = np.loadtxt(f, dtype=np.float32, delimiter=",", skiprows=0).T
            normal_bins = np.concatenate((normal_bins - 0.5, [normal_bins[-1] + 0.5]))
        assert np.allclose(abnormal_bins, normal_bins), "The bins of abnormal and normal PDFs must match"
        assert np.allclose(np.sum(abnormal_height), np.sum(normal_height), rtol=0.01, atol=0.01), "Input PDFs must sum to 1"

        # Fields related to probabilistic distribution
        self.refine_n = 10
        self.hist_abnormal = rv_histogram2(histogram=(abnormal_height, abnormal_bins)).refined(self.refine_n)
        self.hist_normal   = rv_histogram2(histogram=(  normal_height,   normal_bins)).refined(self.refine_n)
        self.center_normal = self.hist_normal.mode()
        self.epsilon = self.hist_normal.height.min()  # 1e-3
        self.coef_abnormal = 1 / (1.0 + self.param["normal2abnormal_ratio"])
        self.coef_normal   = 1 - self.coef_abnormal

    # @lru_cache()
    # def get_init_pdf(self, init_mean):
    #     return (self.coef_abnormal * self.hist_abnormal +
    #             self.coef_normal * self.hist_normal.roll(shift=int(round((init_mean - self.center_normal) * self.refine_n)),
    #                                                      fill_value=self.epsilon))

    @lru_cache(maxsize=512)
    def get_current_estimate(self, threshold_sequence, response_sequence, init_mean, multiplicity=1):
        if np.isscalar(multiplicity):
            multiplicity = np.full_like(threshold_sequence, fill_value=multiplicity, dtype=np.int32)
        # ZEST initial PDF is a weighted combination of abnormal sensitivity prior (constant) and
        # normal prior (shifted based on age model and current best prior)
        # The line below is the bottle neck in performance due to slow scipy.stats.rv_histogram.__init__
        # So we will cache the results
        init_pdf = (self.coef_abnormal * self.hist_abnormal +
                    self.coef_normal * self.hist_normal.roll(
                                            shift=int(round((init_mean - self.center_normal) * self.refine_n)),
                                            fill_value=self.epsilon))
        if len(response_sequence) == 0:
            return init_pdf
        else:
            # Calculate the PDF multiplier based on the trial and PoS function
            trial_height = (ZestStrategy.trial2pos_ramp(x=self.hist_normal.bins[:-1],
                                                       center=c,
                                                       fn=self.param["pos_fn"], fp=self.param["pos_fp"],
                                                       width=self.param["pos_width"],
                                                       seen=s, multiplicity=multi)
                                            for c, s, multi in zip(threshold_sequence, response_sequence, multiplicity))
            # Calculate the product
            trial_height = reduce(mul, trial_height)
            # Convert it to a histogram object with the same bins as the prior (hist_normal) object
            trial_pdf = rv_histogram2(histogram=(trial_height, self.hist_normal.bins))
            # Multiply the product of all trials with the prior
            updated_pdf = init_pdf * trial_pdf  # type: rv_histogram2

            return updated_pdf

    def get_stimulus_threshold(self, data):
        M = len(self.param['pattern'])

        data = Stimulus.to_numpy(data)
        self.extra_data["data"] = data

        self.extra_data["pdf"] = [None] * M
        stimuli = [None] * M  # The next stimulus to be presented
        threshold = np.full(M, np.nan)  # The current best estimate of point thresholds
        threshold_determined = threshold.copy()

        model_mean = self.param["model"].get_mean()
        model_std = self.param["model"].get_std()
        # Pass the model to growth_pattern such that now in model_mean_pending
        # all seed points are set to their model value, and
        # other points that should be tested after the seed points are set to nan
        model_mean_seeds, _ = self.growth_pattern.adjust(mean=model_mean, std=model_std, mean_est=threshold_determined)

        while True:
            # Use m = 1 ... M to index SAP locations
            # Originally used a test_sequence list to rank stimulus test order;
            # completely replaced by growth pattern
            # This loop populates stimuli (length M list of next stimulus at each point) and
            # current best estimated threshold based on the trials (length M array)
            for (m,) in zip(*np.where(np.isfinite(model_mean_seeds))):
                # If this point has already been calculated before, then skip it
                if stimuli[m] is not None or np.isfinite(threshold_determined[m]):
                    continue

                # Get parameters of current location m
                location = self.param["pattern"][m]

                # Get subset of relevant data_m
                data_m = data[data[LOC] == m]  # type: np.ndarray

                response_sequence = data_m[RESPONSE]
                threshold_sequence = data_m[THRESHOLD]
                multiplicity_Sequence = data_m[MULTI]

                updated_pdf = self.get_current_estimate(
                    threshold_sequence=tuple(threshold_sequence),
                    response_sequence=tuple(response_sequence),
                    init_mean=model_mean_seeds[m],
                    multiplicity=tuple(multiplicity_Sequence)
                )

                self.extra_data["pdf"][m] = updated_pdf
                updated_pdf_mean = updated_pdf.mean()
                updated_pdf_std = updated_pdf.std()

                # Turpin 2003:
                # As ZEST returned the mean of the final pdf, which provided a less biased estimate than the mode,8
                # a slightly different threshold again was re-
                # turned by ZEST, because of this factor alone.
                # threshold[m] = updated_pdf.mean()
                # Another implementation is trial_pdf.mode() as in
                # Watson, A. B., Pelli, D. G., & others. (1979). The QUEST staircase procedure. Applied Vision Association Newsletter, 14, 6–7.
                # which is not biased by the shape of init_pdf

                if updated_pdf_std < self.param["term_std"]:
                    # Terminated
                    stimuli[m] = None
                    threshold[m] = updated_pdf_mean
                    threshold_determined[m] = updated_pdf_mean
                else:
                    stimuli[m] = self.get_new_stimulus_at(db=updated_pdf_mean, location=location)
                    threshold[m] = updated_pdf_mean
                    threshold_determined[m] = np.nan

            # recalculate growth pattern to see if more points should be included for testing
            model_mean_seeds_new, _ = self.growth_pattern.adjust(mean=model_mean, std=model_std, mean_est=threshold_determined)
            # If any of the nan status has changed - meaning more points can now be tested, we need to repeat
            if np.any(np.isfinite(model_mean_seeds) != np.isfinite(model_mean_seeds_new)):
                model_mean_seeds = model_mean_seeds_new
                continue
            else:
                break

        stimuli_candidates = [s for s in stimuli if s is not None]
        stimulus = self.get_stimulus_from_candidates(stimuli_candidates)

        return stimulus, threshold

    def get_stimulus_from_candidates(self, stimuli_candidates):
        if len(stimuli_candidates) == 0:
            stimulus = None  # All stimulus has been exhausted
        else:
            stimulus = stimuli_candidates[self.rng.randint(0, len(stimuli_candidates))]
        return stimulus


class ZestMSPStrategy(ZestStrategy):
    @staticmethod
    def by_quadrant(stimuli_candidates_all):
        stimuli_candidates_by_quadrant = [[], [], [], []]
        for s in stimuli_candidates_all:
            x, y = s.xod, s.yod
            if x > 0 and y > 0:
                stimuli_candidates_by_quadrant[0].append(s)
            elif x < 0 and y > 0:
                stimuli_candidates_by_quadrant[1].append(s)
            elif x < 0 and y < 0:
                stimuli_candidates_by_quadrant[2].append(s)
            else:
                stimuli_candidates_by_quadrant[3].append(s)
        return stimuli_candidates_by_quadrant

    def get_stimulus_from_candidates(self, stimuli_candidates_all):
        data = self.extra_data["data"]
        pdf = self.extra_data["pdf"]

        if len(stimuli_candidates_all) == 0:
            return None

        # Shallow copy since we will be modifying this list
        stimuli_candidates_all = list(stimuli_candidates_all)
        # First pick a stimulus  at random
        stimulus1 = stimuli_candidates_all.pop(self.rng.randint(0, len(stimuli_candidates_all)))
        # <s>Check if it has ever been seen before</s>
        # New implementation - check if it has ever been presented before
        done_before = (data[LOC] == stimulus1.loc).any()  # (data[RESPONSE][data[LOC] == stimulus1.loc] == STIMULUS_SEEN).any()
        if done_before:
            # Then we will give up MSP and switch to SSP
            return stimulus1

        # Else - this stimulus is not seen before, we can add a second also never-before-seen stimulus for MSP
        stimulus2_candidates = []
        for s in stimuli_candidates_all:
            # They should not be in the same quadrant
            if np.sign(stimulus1.xod) == np.sign(s.xod) and np.sign(stimulus1.yod) == np.sign(s.yod):
                continue
            # If s has been seen before, then no go
            if (data[LOC] == stimulus1.loc).any():  # (data[RESPONSE][data[LOC] == s.loc] == STIMULUS_SEEN).any():
                continue
            # Now this is a good candidate
            stimulus2_candidates.append(s)

        if len(stimulus2_candidates) == 0: # No second stimulus found
            return stimulus1

        stimulus2 = stimulus2_candidates[self.rng.randint(0, len(stimulus2_candidates))]

        # New implementation - test at mode for MSP
        stimulus1 = stimulus1.copy(**{MULTI: 2})
        stimulus2 = stimulus2.copy(**{MULTI: 2})

        stimulus = [stimulus1, stimulus2]  # MSP, return as a list, not tuple, since stimulus is a namedtuple

        return stimulus
