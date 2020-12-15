from functools import reduce
from operator import mul

import numpy as np

from pyvf.stats.pos import pos_ramp
from pyvf.strategy import RESPONSE, STIMULUS_SEEN, TSRESP, STIM_TSDISP, STIMULUS_NOT_SEEN, Stimulus


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
        if stimulus.threshold <= self.threshold[stimulus.loc]:
            # Seen with zero response time
            return stimulus.copy(**{RESPONSE: STIMULUS_SEEN, TSRESP: stimulus[STIM_TSDISP]})
        else:
            # Not seen with zero response time
            return stimulus.copy(**{RESPONSE: STIMULUS_NOT_SEEN, TSRESP: stimulus[STIM_TSDISP]})


class FunctionResponder(Responder):
    """
    Responder that has outputs based on a probability of seeing curve
    """
    def __init__(self, true_threshold, fp, fn, width, seed=None):
        """

        Parameters
        ----------
        true_threshold : array_like
        fp : array_like | float
        fn : array_like | float
        width : array_like | float
        seed
        """
        self.rng = np.random.RandomState(seed=seed)
        self.threshold = np.array(true_threshold)
        self.fp = np.array(fp) if np.ndim(fp) > 0 else np.full_like(self.threshold, fill_value=fp, dtype=np.float32)
        self.fn = np.array(fn) if np.ndim(fn) > 0 else np.full_like(self.threshold, fill_value=fn, dtype=np.float32)
        self.width = np.array(width) if np.ndim(width) > 0 else np.full_like(self.threshold, fill_value=width, dtype=np.float32)

    def get_response(self, stimulus):
        seen = self.rng.random() <= self.probability_of_seeing(stimulus)
        if isinstance(stimulus, Stimulus):  # SSP
            if seen:
                # Seen with zero response time
                return stimulus.copy(**{RESPONSE: STIMULUS_SEEN, TSRESP: stimulus[STIM_TSDISP]})
            else:
                # Not seen with zero response time
                return stimulus.copy(**{RESPONSE: STIMULUS_NOT_SEEN, TSRESP: stimulus[STIM_TSDISP]})
        elif isinstance(stimulus[0], Stimulus):  # MSP
            if seen:
                # Seen with zero response time
                return [s.copy(**{RESPONSE: STIMULUS_SEEN, TSRESP: s[STIM_TSDISP]}) for s in stimulus]
            else:
                # Not seen with zero response time
                return [s.copy(**{RESPONSE: STIMULUS_NOT_SEEN, TSRESP: s[STIM_TSDISP]}) for s in stimulus]
        else:
            raise ValueError()

    def probability_of_seeing(self, stimulus):
        raise NotImplementedError()


class RampResponder(FunctionResponder):
    def __init__(self, true_threshold, fp=0.05, fn=0.05, width=4, seed=None):
        r"""
        Function shape and default values based on Turpin 2003

        Parameters
        ----------
        true_threshold
        fp
        fn
        width : \Delta dB from the start of the ramp to the end of the ramp (full width)

        References
        ----------------
        .. [1] Turpin, A., McKendrick, A. M., Johnson, C. A., & Vingrys, A. J. (2003).
        Properties of Perimetric Threshold Estimates from Full Threshold, ZEST, and SITA-like Strategies,
        as Determined by Computer Simulation. Investigative Ophthalmology and Visual Science, 44(11), 4787–4795.
        https://doi.org/10.1167/iovs.03-0023
        """
        super().__init__(true_threshold=true_threshold, fp=fp, fn=fn, width=width, seed=seed)

    def probability_of_seeing(self, stimulus):
        # left = 1 - self.fn[stimulus.loc]
        # right = self.fp[stimulus.loc]
        #
        # return pos_ramp(stimulus.threshold,
        #                 center=self.threshold[stimulus.loc], yl=left, yr=right, width=self.width[stimulus.loc])

        # If there is no FP or FN, what is the probability of seeing (p_perfect)?
        if isinstance(stimulus, Stimulus):  # SSP
            p_perfect = pos_ramp(stimulus.threshold, center=self.threshold[stimulus.loc], yl=1.0, yr=0.0, width=self.width[stimulus.loc])
            fn = self.fn[stimulus.loc]
            fp = self.fp[stimulus.loc]
        elif isinstance(stimulus[0], Stimulus):  # MSP
            q_perfect = reduce(mul, [
                1 - pos_ramp(s.threshold, center=self.threshold[s.loc], yl=1.0, yr=0.0, width=self.width[s.loc])
                    for s in stimulus
            ])
            p_perfect = 1 - q_perfect

            # In the design of this class it was allowed to have different FN and FP at different locations
            # Usually we really expect it to be equal at all points
            # Here we take the mean
            fn = self.fn[[s.loc for s in stimulus]].mean()
            fp = self.fp[[s.loc for s in stimulus]].mean()
        else:
            raise ValueError()

        p = fp + (1 - fn - fp) * p_perfect
        return p
