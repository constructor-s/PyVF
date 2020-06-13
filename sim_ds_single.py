"""
Simulation of a PerfectResponder on one single location with a 4-2 double staircase strategy without any retesting

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
import logging

_logger = logging.getLogger(__name__)


def simPerfectSingleStaircase(true_threshold, starting_threshold):
    responder = PerfectResponder(true_threshold=[true_threshold])
    model = ConstantModel(eval_pattern=PATTERN_SINGLE,
                          mean=starting_threshold,
                          std=4)  # std no effect in this case

    strategy = DoubleStaircaseStrategy(
        pattern=PATTERN_SINGLE,
        blindspot=[],
        model=model,
        step=(4, 2),
        threshold_func=DoubleStaircaseStrategy.get_last_seen_threshold
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
    return data


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    data_collection = []
    for i, true_threshold in enumerate(range(40)):
        data_collection.append([])
        for j, starting_threshold in enumerate(range(40)):
            data = simPerfectSingleStaircase(true_threshold=true_threshold, starting_threshold=starting_threshold)
            data_collection[i].append(data)

    presentations = [[len(x) for x in l] for l in data_collection]
    presentations = np.array(presentations)
