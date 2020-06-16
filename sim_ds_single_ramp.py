"""
Simulation of a RampResponder on one single location with a 4-2 double staircase strategy with retesting

The starting value is specified as a constant using a ConstantModel

Note that this entire simulation is deterministic other than possibly the order in which locations are tested
if the strategy randomly samples the points instead of following a fixed order (important in real life
but not necessarily necessary for simulations)


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

from pyvf.strategy import *
from pyvf.strategy.Model import ConstantModel
from pyvf.strategy.Responder import RampResponder

import logging
_logger = logging.getLogger(__name__)


def simPerfectSingleStaircase(true_threshold, starting_threshold, repeat_threshold=4):
    responder = RampResponder(true_threshold=[true_threshold], fp=0.15, fn=0.15)
    model = ConstantModel(eval_pattern=PATTERN_SINGLE,
                          mean=starting_threshold,
                          std=4)  # std no effect in this case

    strategy = DoubleStaircaseStrategy(
        pattern=PATTERN_SINGLE,
        blindspot=[],
        model=model,
        step=(4, 2),
        threshold_func=DoubleStaircaseStrategy.get_last_seen_threshold_or_mean,
        repeat_threshold=repeat_threshold
    )

    data = []
    stimulus, threshold = strategy.get_stimulus_threshold(data)
    counter = 0
    while stimulus is not None:
        stimulus = stimulus.copy(**{TSDISP: counter})
        stimulus = responder.get_response(stimulus)

        _logger.debug("%3d: %s\t%s", counter, threshold, stimulus)

        data.append(stimulus)
        counter += 1

        stimulus, threshold = strategy.get_stimulus_threshold(data)

    _logger.info("%3d: %s\t%s", counter, threshold, stimulus)
    return data, threshold


def sim_ds_single_offsets():
    true_thresholds = np.array([-0.001, 25, 40.001])
    starting_threshold_offsets = np.arange(-10.5, 11.5, 1.0)
    data_collection = []
    for i, true_threshold in enumerate(true_thresholds):
        data_collection.append([])
        for j, offset in enumerate(starting_threshold_offsets):
            data = simPerfectSingleStaircase(true_threshold=true_threshold, starting_threshold=true_threshold + offset)
            data_collection[i].append(data)
    # Calculate how many presentations did it take for each test condition
    presentations = [[len(x[0]) for x in l] for l in data_collection]
    presentations = np.array(presentations)
    final_estimate = [[x[1][0] for x in l] for l in data_collection]
    final_estimate = np.array(final_estimate)
    # Plotting
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(true_thresholds), len(starting_threshold_offsets),
                           sharex='col', sharey='row', figsize=(21, 9))
    for i in range(ax.shape[0]):
        _logger.info("%d", i)
        for j in range(ax.shape[1]):
            data = data_collection[i][j][0]
            data = Stimulus.to_numpy(data)
            ax[i, j].plot(data[TSDISP], data[THRESHOLD], 'k-',
                          data[TSDISP][data[RESPONSE] == STIMULUS_SEEN],
                          data[THRESHOLD][data[RESPONSE] == STIMULUS_SEEN], 'go',
                          data[TSDISP][data[RESPONSE] == STIMULUS_NOT_SEEN],
                          data[THRESHOLD][data[RESPONSE] == STIMULUS_NOT_SEEN], 'rx',
                          [0, len(data[TSDISP]) - 0.5], [true_thresholds[i], true_thresholds[i]], 'k:'
                          )
            ax[i, j].set_facecolor(plt.get_cmap('Reds', 10)(len(data) - 3))
    fig.savefig("sim_ds_single.pdf")

    return data_collection


def sim_ds_single_turpin_2003_fig5():
    true_thresholds = np.arange(0, 40.1, 1.0)
    starting_thresholds = np.array([30])
    N = 100
    repeat_threshold = 4

    data_collection = []
    for i, true_threshold in enumerate(true_thresholds):
        data_collection.append([])
        for j, starting_threshold in enumerate(starting_thresholds):
            data_collection[i].append([])
            for k in range(N):
                data = simPerfectSingleStaircase(true_threshold=true_threshold, starting_threshold=starting_threshold,
                                             repeat_threshold=repeat_threshold)
                data_collection[i][j].append(data)

    # Calculate how many presentations did it take for each test condition
    presentations = [[[len(x[0]) for x in k] for k in j] for j in data_collection]
    presentations = np.array(presentations)
    final_estimate = [[[x[1][0] for x in k] for k in j] for j in data_collection]
    final_estimate = np.array(final_estimate)

    return locals()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # data_collection = sim_ds_single_offsets()
    sim = sim_ds_single_turpin_2003_fig5()

    # Plotting
    import matplotlib.pyplot as plt

    starting_thresholds = sim["starting_thresholds"]
    true_thresholds = sim["true_thresholds"]
    presentations = sim["presentations"]
    repeat_threshold = sim["repeat_threshold"]
    final_estimate = sim["final_estimate"]
    N = sim["N"]

    for j, starting_threshold in enumerate(starting_thresholds):
        fig, ax = plt.subplots(2, 1, sharex='col', sharey='row', figsize=(8.5, 11))
        ax[0].plot(true_thresholds, presentations[:, j].mean(axis=-1), 'k.-')
        ax[0].set_ylabel("number of presentations")
        ax[0].set_yticks(np.arange(0, 16.1, 2))
        ax[0].grid(True)
        ax[0].legend([f"4-2 staircase, retest threshold = {repeat_threshold} dB"])

        ax[1].plot(true_thresholds, final_estimate[:, j].mean(axis=-1) - true_thresholds, 'k.-')
        ax[1].set_ylabel("error (dB)")
        ax[1].set_yticks(np.arange(-20, 20.1, 5))
        ax[1].set_xlabel("Input threshold (dB)")
        ax[1].set_xticks(np.arange(0, 40.1, 5))
        ax[1].grid(True)
        ax[1].set_aspect('equal', adjustable='datalim')

        fig.suptitle(f"Starting estimate = {starting_threshold}dB, FP = 15%, FN = 15%, N = {N}")
        fig.savefig(f"sim_ds_single_{starting_threshold}.pdf")
