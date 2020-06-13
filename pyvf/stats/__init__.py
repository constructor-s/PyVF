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

import scipy.stats
import numpy as np
import operator


class rv_histogram2(scipy.stats.rv_histogram):
    """
    An extension of the rv_histogram class in scipy.stats to offer useful functionality

    Terminology: A histogram is represented as a two element tuple of arrays (height, bins) height is an array of
    length n. Based on the underlying rv_histogram implementation, height can be int or float.

    bins is an array of length (n+1) - it seems that the underlying rv_histogram assumes an even grid spacing.
    Supplying uneven spacing will not result in any error but the PDF is wrong.

    TODO: Does it make sense to also implement subtraction and division?
    """

    def __mul__(self, rhs):
        """
        Overloading the multiplication operator for nicer syntax for constructing PDF and during Bayesian updates.

        When multiplied by a scalar: Return a stored histogram content that is scaled by the scalar factor. Note that
        by definition this does not affect the PDF, but the internal data1 is modified and can be used for other
        calculations. This is useful for doing a linear combination of different histograms (e.g. initial prior PDF
        combining a normal population with some weight and an abnormal population with some other weight).

        When multiplied by another rv_histogram: Currently the bins need to match between self and rhs. The histogram
        content is element-wise multiplied. This is useful for Bayesian update.

        Parameters
        ----------
        rhs

        Returns
        -------
        Resulting product

        """
        return self._operate(rhs, op=operator.mul)

    def __rmul__(self, lhs):
        return self * lhs

    def __add__(self, rhs):
        """
        When added by a scalar: Increase each height by the rhs - note that this is on the raw histogram,
        not normalized PDF.

        When added by another rv_histogram: Currently the bins need to match between self and rhs. The histogram
        content is element-wise added.

        Parameters
        ----------
        rhs

        Returns
        -------

        """
        return self._operate(rhs, op=operator.add)

    def __radd__(self, lhs):
        return self + lhs

    def _operate(self, rhs, op):
        """
        Generalization of the __mul__ and __add__ methods.

        Parameters
        ----------
        rhs

        Returns
        -------

        """
        # Multiplied by a scalar
        # This is preferred over np.scalar. See https://numpy.org/doc/stable/reference/generated/numpy.isscalar.html
        if np.ndim(rhs) == 0:
            height, bins = self._histogram
            return rv_histogram2(histogram=(op(height, rhs), bins))

        # Multiplied by another histogram
        else:  # Try duck typing
            try:
                # noinspection PyProtectedMember
                r_height, r_bins = rhs._histogram
            except (AttributeError, ValueError) as e:
                raise TypeError(f"unsupported operand type(s) for +: '{type(self)}' and '{type(rhs)}'")

            height, bins = self._histogram

            if not np.allclose(bins, r_bins):  # TODO: Implement multiplication between different bins
                raise ValueError("bins must match between two histograms")

            # Multiply the content of the histogram
            return rv_histogram2(histogram=(op(height, r_height), bins))

    def __str__(self):
        height, bins = self._histogram
        return ("[[" + ",".join(["%6.3g" % i for i in bins])
                + "],\n [" +
                ",".join(["%6.3g" % i for i in height])
                + ",     0],\n [" +
                ",".join(["%6.3g" % i for i in self.pdf(bins)])
                + "]]")

    def normalized(self):
        """

        Returns
        -------
        A new copy of the current histogram with the underlying histogram normalized to the PDF
        """
        height, bins = self._histogram
        # self.pdf(bins[-1]) == 0 based on the base rv_histogram implementation
        height_normalized = height * 1.0 / height.sum()
        return rv_histogram2(histogram=(height_normalized, bins))
