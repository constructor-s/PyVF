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
        model = strategy.ConstantModel(eval_pattern=strategy.PATTERN_SINGLE, mean=30, std=4)
        print("Mean:", model.get_mean())
        print("StD:", model.get_std())
        print(repr(strategy.PATTERN_P24D2))
        print(strategy.PATTERN_P24D2.shape)
        print(strategy.PATTERN_P24D2[0][strategy.XOD])

    def test_heijl1987(self):
        model = strategy.Heijl1987p24d2Model(eval_pattern=strategy.PATTERN_P24D2, age=0)
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
        self.assertTrue(np.isnan(strategy.DoubleStaircaseStrategy.get_last_seen_threshold([])))
        self.assertEqual(strategy.DoubleStaircaseStrategy.get_last_seen_threshold(self.data1), 26)
        self.assertEqual(strategy.DoubleStaircaseStrategy.get_last_seen_threshold(self.data2), 0)

    def test_staircase_stats(self):
        sequence = []
        reversals, direction = strategy.DoubleStaircaseStrategy.get_staircase_stats(sequence)
        self.assertEqual(reversals, 0)
        self.assertEqual(direction, 0)

        sequence = [0]
        reversals, direction = strategy.DoubleStaircaseStrategy.get_staircase_stats(sequence)
        self.assertEqual(reversals, 0)
        self.assertEqual(direction, -1)

        sequence = [1]
        reversals, direction = strategy.DoubleStaircaseStrategy.get_staircase_stats(sequence)
        self.assertEqual(reversals, 0)
        self.assertEqual(direction, +1)

        sequence = [0, 0, 1, 1, 1, 0]
        reversals, direction = strategy.DoubleStaircaseStrategy.get_staircase_stats(sequence)
        self.assertEqual(reversals, 2)
        self.assertEqual(direction, -1)

        print(timeit.timeit(lambda: strategy.DoubleStaircaseStrategy.get_staircase_stats(sequence), number=10000))

    def test_goldmann(self):
        sizes = strategy.goldmann_diameter(np.arange(0, 6))
        print(sizes)
        self.assertEqual(np.round(sizes[3], 2), 0.43)
        self.assertEqual(np.round(sizes[5], 2), 1.72)
