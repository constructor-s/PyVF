"""
Simulate 24-2 visual fields

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

from pyvf.strategy import ZestStrategy, PATTERN_P24D2, XOD, YOD, TSDISP
from pyvf.strategy.GrowthPattern import SimpleP24d2QuadrantGrowth
from pyvf.strategy.Model import ConstantModel
from pyvf.strategy.Responder import RampResponder, PerfectResponder
import numpy as np
import logging
_logger = logging.getLogger(__name__)


def pretty_print_vf(arr, pattern=PATTERN_P24D2):
    for i, (val, xod, yod) in enumerate(zip(arr, pattern[XOD], pattern[YOD])):
        if i == 0 or last_yod == yod:
            pass
        else:
            print()
        print(f"{val:6.2f}", end="\t")
        last_yod = yod
    print()


def main():
    logging.basicConfig(level=logging.DEBUG)

    true_thresholds = np.arange(len(PATTERN_P24D2)) * 0.5
    print(true_thresholds)
    pretty_print_vf(true_thresholds, PATTERN_P24D2)

    responder = PerfectResponder(true_threshold=true_thresholds)
    model = ConstantModel(eval_pattern=PATTERN_P24D2,
                          mean=np.full_like(true_thresholds, 30.0),  # Initial guess = population mean
                          std=4)
    strategy = ZestStrategy(
        pattern=PATTERN_P24D2,
        blindspot=[],
        model=model,
        term_std=1.5,
        rng=0,
        growth_pattern=SimpleP24d2QuadrantGrowth()
    )

    counter = 0
    data = []
    while True:
        if counter % 10 == 0:
            print(counter)
        stimulus, threshold = strategy.get_stimulus_threshold(data)
        if stimulus is None:
            break  # Test is finished
        else:
            stimulus = stimulus.copy(**{TSDISP: counter})
            stimulus = responder.get_response(stimulus)
            # _logger.debug("%3d: %s\t%s", counter, threshold, stimulus)
            data.append(stimulus)
            counter += 1

    print(strategy.get_current_estimate.cache_info())
    import matplotlib.pyplot as plt
    plt.plot(true_thresholds)
    plt.plot(threshold)
    plt.show()


if __name__ == '__main__':
    main()
    """
    from line_profiler import LineProfiler

    lp = LineProfiler()
    lp.add_function(ZestStrategy.get_stimulus_threshold)
    lp_wrapper = lp(main)
    lp_wrapper()
    with open("output/sim_field_zest_profile.py", "w") as f:
        lp.print_stats(stream=f, output_unit=1e-3)
    """
