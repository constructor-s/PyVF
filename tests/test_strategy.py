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

import pyvf.strategy.Model as sModel
from pyvf import strategy
import numpy as np
import timeit


class TestStrategy(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data1 = strategy.Stimulus.to_numpy([
            strategy.Stimulus(3, 3, 0, strategy.GOLDMANN_III, 30, strategy.STIMULUS_NOT_SEEN, 0, 0),
            strategy.Stimulus(3, 3, 0, strategy.GOLDMANN_III, 26, strategy.STIMULUS_SEEN, 0, 0),
            strategy.Stimulus(3, 3, 0, strategy.GOLDMANN_III, 28, strategy.STIMULUS_NOT_SEEN, 0, 0)
        ])
        self.data2 = strategy.Stimulus.to_numpy([
            strategy.Stimulus(3, 3, 0, strategy.GOLDMANN_III, 16, strategy.STIMULUS_NOT_SEEN, 0, 0),
            strategy.Stimulus(3, 3, 0, strategy.GOLDMANN_III, 12, strategy.STIMULUS_NOT_SEEN, 0, 0),
            strategy.Stimulus(3, 3, 0, strategy.GOLDMANN_III, 8, strategy.STIMULUS_NOT_SEEN, 0, 0),
            strategy.Stimulus(3, 3, 0, strategy.GOLDMANN_III, 4, strategy.STIMULUS_NOT_SEEN, 0, 0),
            strategy.Stimulus(3, 3, 0, strategy.GOLDMANN_III, 0, strategy.STIMULUS_NOT_SEEN, 0, 0),
        ])

    def test_constant_model(self):
        model = sModel.ConstantModel(eval_pattern=strategy.PATTERN_SINGLE, mean=30, std=4)
        print("Mean:", model.get_mean())
        print("StD:", model.get_std())
        print(repr(strategy.PATTERN_P24D2))
        print(strategy.PATTERN_P24D2.shape)
        print(strategy.PATTERN_P24D2[0][strategy.XOD])

    def test_heijl1987(self):
        model = sModel.Heijl1987p24d2Model(eval_pattern=strategy.PATTERN_P24D2, age=0)
        print("Mean:", model.get_mean())
        print("StD:", model.get_std())

    def test_stimulus(self):
        stim = strategy.Stimulus(3, 3, 0, strategy.GOLDMANN_III, 30, strategy.STIMULUS_NO_RESPONSE, 0, 0.5)
        stim2 = strategy.Stimulus(3, 3, 0, strategy.GOLDMANN_III, 26, strategy.STIMULUS_SEEN, 1, 1.5)

        print(stim)
        # dtype = [("xod", np.float32), ("yod", np.float32), ("loc", np.int32), ("size", np.float32), ("response", np.float32), ("tsdisp", np.float32), ("tsresp", np.float32)]
        # print(np.array(stim, dtype=dtype)["size"])
        # data1 = np.array([stim] * 2, dtype=dtype)
        # print(data1)
        # print(data1["loc"])
        # print(data1[data1["loc"] == 0])

        stim_arr = strategy.Stimulus.to_numpy([stim2, stim])
        print(stim_arr)
        print(stim_arr[strategy.THRESHOLD][1])
        print(np.sort(stim_arr, axis=0, order=strategy.TSRESP))

    def test_staircase_threshold(self):
        model = sModel.ConstantModel(eval_pattern=strategy.PATTERN_SINGLE,
                                     mean=30,
                                     std=4)  # std no effect in this case
        s = strategy.DoubleStaircaseStrategy(
            pattern=strategy.PATTERN_SINGLE,
            blindspot=[],
            model=model,
            step=(4, 2),
            threshold_func=strategy.DoubleStaircaseStrategy.get_last_seen_threshold_or_mean
        )
        self.assertTrue(np.isnan(s.get_last_seen_threshold([])))
        self.assertEqual(s.get_last_seen_threshold(self.data1), 26)
        self.assertEqual(s.get_last_seen_threshold(self.data2), 0)

    def test_staircase_stats(self):
        sequence = []
        reversals, repeats = strategy.DoubleStaircaseStrategy.get_staircase_stats(sequence)
        self.assertSequenceEqual(reversals, [])
        # self.assertSequenceEqual(direction, [])

        sequence = [0]
        reversals, repeats = strategy.DoubleStaircaseStrategy.get_staircase_stats(sequence)
        self.assertSequenceEqual(reversals, [0])
        # self.assertSequenceEqual(direction, [-1])

        sequence = [1]
        reversals, repeats = strategy.DoubleStaircaseStrategy.get_staircase_stats(sequence)
        self.assertSequenceEqual(reversals, [0])
        # self.assertSequenceEqual(direction, [+1])

        sequence = [0, 0, 1, 1, 1, 0]
        reversals, repeats = strategy.DoubleStaircaseStrategy.get_staircase_stats(sequence)
        self.assertSequenceEqual(reversals.tolist(), [0, 0, 1, 1, 1, 2])
        # self.assertSequenceEqual(direction.tolist(), [-1, -1, +1, +1, +1, -1])

        sequence = [0, 0, 1, 1, 1, 0, 0, 0, 1]
        reversals, repeats = strategy.DoubleStaircaseStrategy.get_staircase_stats(sequence)
        self.assertSequenceEqual(reversals.tolist(), [0, 0, 1, 1, 1, 2, 2, 2, 3])
        # self.assertSequenceEqual(direction.tolist(), [-1, -1, +1, +1, +1, -1, -1, -1, +1])

        sequence = [0, 0, 1, 1, 1, 0, 0, 0, 1, 0]
        reversals, repeats = strategy.DoubleStaircaseStrategy.get_staircase_stats(sequence, step=(4, 2))
        self.assertSequenceEqual(reversals.tolist(), [0, 0, 1, 1, 1, 2, 0, 0, 1, 2])
        self.assertSequenceEqual(repeats.tolist(), [0, 0, 0, 0, 0, 0, 1, 1, 1, 1])

        sequence        = [0, 0, 1, 1, 1, 0, 0, 0, 1, 0]
        force_terminate = [0, 1, 0, 0, 0, 0, 1, 0, 0, 0]
        reversals, repeats = strategy.DoubleStaircaseStrategy.get_staircase_stats(sequence, step=(4, 2), force_terminate=force_terminate)
        self.assertSequenceEqual(reversals.tolist(), [0, 0, 0, 0, 0, 1, 1, 0, 1, 2])
        self.assertSequenceEqual(repeats.tolist(), [0, 0, 1, 1, 1, 1, 1, 2, 2, 2])

        # print(timeit.timeit(lambda: strategy.DoubleStaircaseStrategy.get_staircase_stats(sequence, step=(4, 2)), number=10000))

    def test_process_repeated_staircase(self):
        n = 2
        x = [0, 1, 2, 2, 3, 4]
        y = [0, 1, 2, 0, 1, 2]
        self.assertSequenceEqual(strategy.DoubleStaircaseStrategy._process_repeated_staircase(np.array(x), n).tolist(), y)

        n = 2
        x = [0, 1, 2, 3, 4, 5]
        y = [0, 1, 2, 0, 1, 2]
        self.assertSequenceEqual(strategy.DoubleStaircaseStrategy._process_repeated_staircase(np.array(x), n).tolist(), y)

        n = 2
        x = [0, 1, 2, 2, 3, 4, 4, 5, 6]
        y = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        self.assertSequenceEqual(strategy.DoubleStaircaseStrategy._process_repeated_staircase(np.array(x), n).tolist(), y)

        n = 2
        x = []
        y = []
        self.assertSequenceEqual(strategy.DoubleStaircaseStrategy._process_repeated_staircase(np.array(x), n).tolist(), y)

        n = 2
        x = [0]
        y = [0]
        self.assertSequenceEqual(strategy.DoubleStaircaseStrategy._process_repeated_staircase(np.array(x), n).tolist(), y)

        n = 2
        x = [0, 1]
        y = [0, 1]
        self.assertSequenceEqual(strategy.DoubleStaircaseStrategy._process_repeated_staircase(np.array(x), n).tolist(), y)

        n = 2
        x = [0, 1, 1, 1, 2]
        y = [0, 1, 1, 1, 2]
        self.assertSequenceEqual(strategy.DoubleStaircaseStrategy._process_repeated_staircase(np.array(x), n).tolist(), y)

        print(timeit.timeit(lambda: strategy.DoubleStaircaseStrategy._process_repeated_staircase(np.array(x), n), number=1000))

    def test_goldmann(self):
        sizes = strategy.goldmann_diameter(np.arange(0, 6))
        print(sizes)
        self.assertEqual(np.round(sizes[3], 2), 0.43)
        self.assertEqual(np.round(sizes[5], 2), 1.72)
