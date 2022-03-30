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
from pyvf.stats import pdf_stats


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

    def test_bench_histogram(self):
        return
        bins = np.array([  -0.5,  -0.4,  -0.3,  -0.2,  -0.1,     0,   0.1,   0.2,   0.3,   0.4,   0.5,   0.6,   0.7,   0.8,   0.9,     1,   1.1,   1.2,   1.3,   1.4,   1.5,   1.6,   1.7,   1.8,   1.9,     2,   2.1,   2.2,   2.3,   2.4,   2.5,   2.6,   2.7,   2.8,   2.9,     3,   3.1,   3.2,   3.3,   3.4,   3.5,   3.6,   3.7,   3.8,   3.9,     4,   4.1,   4.2,   4.3,   4.4,   4.5,   4.6,   4.7,   4.8,   4.9,     5,   5.1,   5.2,   5.3,   5.4,   5.5,   5.6,   5.7,   5.8,   5.9,     6,   6.1,   6.2,   6.3,   6.4,   6.5,   6.6,   6.7,   6.8,   6.9,     7,   7.1,   7.2,   7.3,   7.4,   7.5,   7.6,   7.7,   7.8,   7.9,     8,   8.1,   8.2,   8.3,   8.4,   8.5,   8.6,   8.7,   8.8,   8.9,     9,   9.1,   9.2,   9.3,   9.4,   9.5,   9.6,   9.7,   9.8,   9.9,    10,  10.1,  10.2,  10.3,  10.4,  10.5,  10.6,  10.7,  10.8,  10.9,    11,  11.1,  11.2,  11.3,  11.4,  11.5,  11.6,  11.7,  11.8,  11.9,    12,  12.1,  12.2,  12.3,  12.4,  12.5,  12.6,  12.7,  12.8,  12.9,    13,  13.1,  13.2,  13.3,  13.4,  13.5,  13.6,  13.7,  13.8,  13.9,    14,  14.1,  14.2,  14.3,  14.4,  14.5,  14.6,  14.7,  14.8,  14.9,    15,  15.1,  15.2,  15.3,  15.4,  15.5,  15.6,  15.7,  15.8,  15.9,    16,  16.1,  16.2,  16.3,  16.4,  16.5,  16.6,  16.7,  16.8,  16.9,    17,  17.1,  17.2,  17.3,  17.4,  17.5,  17.6,  17.7,  17.8,  17.9,    18,  18.1,  18.2,  18.3,  18.4,  18.5,  18.6,  18.7,  18.8,  18.9,    19,  19.1,  19.2,  19.3,  19.4,  19.5,  19.6,  19.7,  19.8,  19.9,    20,  20.1,  20.2,  20.3,  20.4,  20.5,  20.6,  20.7,  20.8,  20.9,    21,  21.1,  21.2,  21.3,  21.4,  21.5,  21.6,  21.7,  21.8,  21.9,    22,  22.1,  22.2,  22.3,  22.4,  22.5,  22.6,  22.7,  22.8,  22.9,    23,  23.1,  23.2,  23.3,  23.4,  23.5,  23.6,  23.7,  23.8,  23.9,    24,  24.1,  24.2,  24.3,  24.4,  24.5,  24.6,  24.7,  24.8,  24.9,    25,  25.1,  25.2,  25.3,  25.4,  25.5,  25.6,  25.7,  25.8,  25.9,    26,  26.1,  26.2,  26.3,  26.4,  26.5,  26.6,  26.7,  26.8,  26.9,    27,  27.1,  27.2,  27.3,  27.4,  27.5,  27.6,  27.7,  27.8,  27.9,    28,  28.1,  28.2,  28.3,  28.4,  28.5,  28.6,  28.7,  28.8,  28.9,    29,  29.1,  29.2,  29.3,  29.4,  29.5,  29.6,  29.7,  29.8,  29.9,    30,  30.1,  30.2,  30.3,  30.4,  30.5,  30.6,  30.7,  30.8,  30.9,    31,  31.1,  31.2,  31.3,  31.4,  31.5,  31.6,  31.7,  31.8,  31.9,    32,  32.1,  32.2,  32.3,  32.4,  32.5,  32.6,  32.7,  32.8,  32.9,    33,  33.1,  33.2,  33.3,  33.4,  33.5,  33.6,  33.7,  33.8,  33.9,    34,  34.1,  34.2,  34.3,  34.4,  34.5,  34.6,  34.7,  34.8,  34.9,    35,  35.1,  35.2,  35.3,  35.4,  35.5,  35.6,  35.7,  35.8,  35.9,    36,  36.1,  36.2,  36.3,  36.4,  36.5,  36.6,  36.7,  36.8,  36.9,    37,  37.1,  37.2,  37.3,  37.4,  37.5,  37.6,  37.7,  37.8,  37.9,    38,  38.1,  38.2,  38.3,  38.4,  38.5,  38.6,  38.7,  38.8,  38.9,    39,  39.1,  39.2,  39.3,  39.4,  39.5,  39.6,  39.7,  39.8,  39.9,    40,  40.1,  40.2,  40.3,  40.4,  40.5, 41.0])
        height = np.array([0.00308,0.00308,0.00308,0.00308,0.00308,0.00308,0.00308,0.00308,0.00308,0.00308,0.000102,0.000102,0.000102,0.000102,0.000102,0.000102,0.000102,0.000102,0.000102,0.000102,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000102,0.000102,0.000102,0.000102,0.000102,0.000102,0.000102,0.000102,0.000102,0.000102,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000102,0.000102,0.000102,0.000102,0.000102,0.000102,0.000102,0.000102,0.000102,0.000102,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000102,0.000102,0.000102,0.000102,0.000102,0.000102,0.000102,0.000102,0.000102,0.000102,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000297,0.000297,0.000297,0.000297,0.000297,0.000297,0.000297,0.000297,0.000297,0.000297,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.0004,0.0004,0.0004,0.0004,0.0004,0.0004,0.0004,0.0004,0.0004,0.0004,0.0004,0.0004,0.0004,0.0004,0.0004,0.0004,0.0004,0.0004,0.0004,0.0004,0.000501,0.000501,0.000501,0.000501,0.000501,0.000501,0.000501,0.000501,0.000501,0.000501,0.0004,0.0004,0.0004,0.0004,0.0004,0.0004,0.0004,0.0004,0.0004,0.0004,0.000501,0.000501,0.000501,0.000501,0.000501,0.000501,0.000501,0.000501,0.000501,0.000501,0.000501,0.000501,0.000501,0.000501,0.000501,0.000501,0.000501,0.000501,0.000501,0.000501,0.000611,0.000611,0.000611,0.000611,0.000611,0.000611,0.000611,0.000611,0.000611,0.000611,0.000304,0.000304,0.000304,0.000304,0.000304,0.000304,0.000304,0.000304,0.000304,0.000304,0.000304,0.000304,0.000304,0.000304,0.000304,0.000304,0.000304,0.000304,0.000304,0.000304,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000199,0.000102,0.000102,0.000102,0.000102,0.000102,0.000102,0.000102,0.000102,0.000102,0.000102,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0])
        def fun():
            hist = stats.rv_histogram2(histogram=(height, bins))
            hist_shifted = hist.roll(shift=-10, fill_value=0.01)
            hist = 0.1 * hist + 0.9 * hist_shifted
            return hist

        from line_profiler import LineProfiler

        lp = LineProfiler()
        lp.add_function(fun)
        lp.add_function(stats.rv_histogram2.__init__)
        lp.add_function(stats.rv_histogram2.__mul__)
        lp.add_function(stats.rv_histogram2._operate)
        lp_wrapper = lp(lambda *args, **kwargs: [fun() for _ in range(1000)])
        lp_wrapper()
        lp.print_stats(output_unit=1e-6)

    def test_variance(self):
        self.assertAlmostEqual(pdf_stats.variance(np.arange(41), np.ones(41)), (41 * 41 - 1) / 12)
