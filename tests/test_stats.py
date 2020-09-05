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
from pyvf import stats
import numpy as np

class Testrv_histogram2(TestCase):
    def test_mul(self):
        height = np.array([1.1, 2.2, 4.4, 8.8])
        bins = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        histogram1 = stats.rv_histogram2(histogram=(height, bins))
        print("Original histogram")
        print(histogram1)
        histogram2 = 2 * histogram1
        print("Original histogram * 2")
        print(histogram2)
        self.assertSequenceEqual((height * 2).tolist(),
                                 (histogram2._histogram[0]).tolist())
        self.assertSequenceEqual((bins).tolist(),
                                 (histogram2._histogram[1]).tolist())

        from operator import mul
        from functools import reduce
        height = np.array([1, 1, 0])
        bins = np.array([0, 1, 2, 3])
        hist3 = stats.rv_histogram2(histogram=(height, bins))
        height = np.array([0, 1, 1])
        bins = np.array([0, 1, 2, 3])
        hist4 = stats.rv_histogram2(histogram=(height, bins))
        hist_prod = reduce(mul, [hist3, hist4], 1)
        print(hist_prod)
        self.assertSequenceEqual(hist_prod._histogram[0].tolist(), [0, 1, 0])
        self.assertSequenceEqual(hist_prod._histogram[1].tolist(), [0, 1, 2, 3])

    def test_add(self):
        height = np.array([1.1, 2.2, 4.4, 8.8])
        bins = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        histogram1 = stats.rv_histogram2(histogram=(height, bins))
        print("Original histogram")
        print(histogram1)
        histogram2 = 100 + histogram1
        print("Original histogram + 100")
        print(histogram2)
        self.assertSequenceEqual((height + 100).tolist(),
                                 (histogram2._histogram[0]).tolist())
        self.assertSequenceEqual((bins).tolist(),
                                 (histogram2._histogram[1]).tolist())

    def test_normalize(self):
        height = np.array([1.1, 2.2, 4.4, 8.8])
        bins = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        histogram1 = stats.rv_histogram2(histogram=(height, bins))
        print("Original histogram")
        print(histogram1)
        histogram2 = histogram1.normalized()
        print("Original histogram normalized")
        print(histogram2)
        self.assertSequenceEqual(histogram1.pdf(bins[:-1]).tolist(),
                                 (histogram2._histogram[0]).tolist())
        self.assertSequenceEqual((bins).tolist(),
                                 (histogram2._histogram[1]).tolist())

    def test_stats(self):
        height = np.array([1.0, 1.0])
        bins = np.array([-1.0, 0.0, 1.0])
        histogram1 = stats.rv_histogram2(histogram=(height, bins))
        self.assertSequenceEqual(histogram1.pdf([-0.5, 0.5]).tolist(), [0.5, 0.5])
        self.assertAlmostEqual(histogram1.mean(), 0.0)
        self.assertAlmostEqual(histogram1.median(), 0.0)
        self.assertAlmostEqual(histogram1.var(), 1.0/3.0)
        self.assertAlmostEqual(histogram1.std(), np.sqrt(1.0/3.0))

        self.assertAlmostEqual(histogram1.mode(), -0.5)

        height = np.array([1.0, 2.0])
        bins = np.array([-1.0, 0.0, 1.0])
        histogram2 = stats.rv_histogram2(histogram=(height, bins))
        self.assertAlmostEqual(histogram2.mode(), 0.5)

        histogram3 = histogram2.refined(10)
        self.assertAlmostEqual(histogram3.mode(), 0.05)

    def test_zero(self):
        height = np.array([0.0, 0.0])
        bins = np.array([-1.0, 0.0, 1.0])

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            histogram1 = stats.rv_histogram2(histogram=(height, bins))
        print(histogram1)

    def test_refined(self):
        height = np.array([1.0, 2.0])
        bins = np.array([-1.0, 0.0, 1.0])
        histogram1 = stats.rv_histogram2(histogram=(height, bins))
        histogram2 = histogram1.refined(4)
        self.assertSequenceEqual(histogram2._histogram[0].tolist(),
                                 [0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5])
        self.assertSequenceEqual(histogram2.bins.tolist(),
                                 [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])

    def test_roll(self):
        height = np.arange(0, 40+0.1, 1)
        bins = np.arange(-0.5, 40.5+0.1, 1)
        histogram1 = stats.rv_histogram2(histogram=(height, bins))

        histogram2 = histogram1.roll(shift=2)
        np.testing.assert_equal(
            histogram2._histogram[0],
            [np.nan,np.nan,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38]
        )

        histogram2 = histogram1.roll(shift=-3, fill_value=0.001)
        np.testing.assert_equal(
            histogram2._histogram[0],
            [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,0.001,0.001,0.001]
        )
