"""

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

from unittest import TestCase

import numpy as np

from pyvf.stats.pos import pos_ramp
from pyvf.strategy import GOLDMANN_III, STIMULUS_NO_RESPONSE, Stimulus, STIMULUS_TS_NONE, STIMULUS_SEEN, \
    STIMULUS_NOT_SEEN
from pyvf.strategy.Responder import PerfectResponder, RampResponder


class Test_POS(TestCase):
    def test_ramp(self):
        self.assertAlmostEqual(pos_ramp(0, 10, 0.1, 0.8, 4), 0.1)
        self.assertAlmostEqual(pos_ramp(10, 10, 0.1, 0.8, 4), 0.45)  # 0.5)
        self.assertAlmostEqual(pos_ramp(40, 10, 0.1, 0.8, 4), 0.8)

        # for x, y in zip([0, 8, 9, 10, 11, 40], [0.1, 0.1, 0.3, 0.5, 0.7, 0.7]):
        for x, y in zip([0, 8.5, 9, 10, 11, 11.5, 40], [0.1, 0.1, 0.2, 0.4, 0.6, 0.7, 0.7]):
            self.assertAlmostEqual(pos_ramp(x, 10, 0.1, 0.7, 3), y)

        # self.assertTrue(np.allclose(pos_ramp([0, 8, 9, 10, 11, 40], 10, 0.1, 0.7, 3),
        #                                      [0.1, 0.1, 0.3, 0.5, 0.7, 0.7]))
        self.assertTrue(np.allclose(pos_ramp([0, 8.5, 9, 10, 11, 11.5, 40], 10, 0.1, 0.7, 3),
                                             [0.1, 0.1, 0.2, 0.4, 0.6, 0.7, 0.7]))

        # import matplotlib.pyplot as plt
        # import numpy as np
        # plt.plot(np.arange(0, 40, 0.1), pos_ramp(np.arange(0, 40, 0.1), 20, 0.1, 1-0.2, 4))
        # plt.grid(True)
        # plt.show()

    def test_ramp_profile(self):
        return
        try:
            from line_profiler import LineProfiler
        except ImportError:
            return
        # Baseline: Total time: 1.93612 s for N = 20000
        # Optimized separate scalar: Total time: 0.318848 s
        # Use try except: Total time: 0.254423 s

        lp = LineProfiler()
        lp.add_function(pos_ramp)
        lp_wrapper = lp(lambda *args, **kwargs: [pos_ramp(*args, **kwargs) for _ in range(20000)])
        lp_wrapper(0, 10, 0.1, 0.8, 0)
        lp_wrapper(10, 10, 0.1, 0.8, 40)
        lp_wrapper(40, 10, 0.1, 0.8, 11)
        with open("test_ramp_profile.py", "w") as f:
            lp.print_stats(stream=f, output_unit=1e-3)

    def test_perfect_responder(self):
        true_threshold = [0, 20, 40]
        responder = PerfectResponder(true_threshold)

        get_stimulus = lambda loc, threshold: Stimulus(
            xod=3.0,
            yod=3.0,
            loc=loc,
            size=GOLDMANN_III,
            threshold=threshold,
            response=STIMULUS_NO_RESPONSE,
            tsdisp=STIMULUS_TS_NONE,
            tsresp=STIMULUS_TS_NONE
        )
        self.assertEqual(responder.get_response(get_stimulus(0, -1.0)).response, STIMULUS_SEEN)
        self.assertEqual(responder.get_response(get_stimulus(0, +1.0)).response, STIMULUS_NOT_SEEN)
        self.assertIn(responder.get_response(get_stimulus(0, 0.0)).response, [STIMULUS_NOT_SEEN, STIMULUS_SEEN])

        self.assertEqual(responder.get_response(get_stimulus(2, 40-1.0)).response, STIMULUS_SEEN)
        self.assertEqual(responder.get_response(get_stimulus(2, 40+1.0)).response, STIMULUS_NOT_SEEN)
        self.assertIn(responder.get_response(get_stimulus(2, 40+0.0)).response, [STIMULUS_NOT_SEEN, STIMULUS_SEEN])

    def test_ramp_responder(self):
        true_threshold = [0, 20, 40]
        responder = RampResponder(true_threshold, fp=0.05, fn=0.05, width=4)

        get_stimulus = lambda loc, threshold: Stimulus(
            xod=3.0,
            yod=3.0,
            loc=loc,
            size=GOLDMANN_III,
            threshold=threshold,
            response=STIMULUS_NO_RESPONSE,
            tsdisp=STIMULUS_TS_NONE,
            tsresp=STIMULUS_TS_NONE
        )

        self.assertAlmostEqual(responder.probability_of_seeing(get_stimulus(loc=1, threshold=15)), 0.95)
        self.assertAlmostEqual(responder.probability_of_seeing(get_stimulus(loc=1, threshold=25)), 0.05)
        self.assertAlmostEqual(responder.probability_of_seeing(get_stimulus(loc=1, threshold=20)), 0.50)

        self.assertAlmostEqual(responder.probability_of_seeing([get_stimulus(loc=1, threshold=0), get_stimulus(loc=2, threshold=0)]), 0.95)
        self.assertAlmostEqual(responder.probability_of_seeing([get_stimulus(loc=1, threshold=50), get_stimulus(loc=2, threshold=50)]), 0.05)
        self.assertAlmostEqual(responder.probability_of_seeing([get_stimulus(loc=1, threshold=20), get_stimulus(loc=2, threshold=40)]), (3-2*0.05)/4.0)
