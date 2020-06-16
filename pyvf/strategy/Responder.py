import numpy as np

from pyvf.strategy import RESPONSE, STIMULUS_SEEN, TSRESP, STIM_TSDISP, STIMULUS_NOT_SEEN


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