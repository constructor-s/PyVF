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

