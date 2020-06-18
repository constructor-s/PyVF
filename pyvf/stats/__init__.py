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
import warnings


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
        if np.isscalar(rhs):
            height, bins = self._histogram
            return rv_histogram2(histogram=(op(height, rhs), bins))

        # Multiplied by another histogram
        else:  # Try duck typing
            try:
                # noinspection PyProtectedMember
                r_height, r_bins = rhs._histogram
                r_height = np.asanyarray(r_height)
            except (AttributeError, ValueError) as e:
                raise TypeError(f"unsupported operand type(s) for {op}: '{type(self)}' and '{type(rhs)}'")

            height, bins = self._histogram
            height = np.asanyarray(height)

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

    def refined(self, n):
        """
        Refine the bin spacing by n fold (each interval is uniformed divided into n intervals).
        Heights are correspondingly adjusted by 1/n.

        Parameters
        ----------
        n : int

        Returns
        -------
        new_histogram : rv_histogram2
            A new histogram with the refined spacing
        """
        height, bins = self._histogram

        # Vectorized magic...
        delta = np.diff(bins) * 1.0 / n
        delta = delta.reshape( (-1, 1) )
        grid = np.arange(n).reshape( (1, -1) )
        Delta = delta.dot(grid)
        new_bins = bins[:-1].reshape( (-1, 1) )
        new_bins = new_bins + Delta
        new_bins = new_bins.ravel()
        new_bins = np.concatenate( (new_bins, bins[-1:]) )

        new_height = height * 1.0 / n
        new_height = np.repeat(new_height, n)

        return rv_histogram2(histogram=(new_height, new_bins))

    def roll(self, shift, fill_value=np.nan):
        """
        Shift the heights with respect to the bins

        Parameters
        ----------
        shift : int
            Number of bins to roll over.
        fill_value : float
            Value to fill the new empty slots after shifting

        Notes
        ---------------
        shift is the number of bins to roll over (units of indices), not the unit that bins are in

        The fill_value is used to replace heights, which are not guaranteed to be normalized.
        Get a normalized() version first if that is the expected behavior.

        Returns
        -------
        ret : rv_histogram2
            A new histogram

        See Also
        ------------
        np.roll
        """
        height, bins = self._histogram
        # shift heights, bins remain
        height = np.roll(height, shift=shift)
        # np.roll is circular, we need to reset the circularly shifted values
        if shift > 0:
            height[:shift] = fill_value
        else:
            height[shift:] = fill_value
        return rv_histogram2(histogram=(height, bins))

    def mode(self, warn=False):
        """

        Returns
        -------
        mode : float
            The center of the (first) bin that has the highest height.
            If there are multiple bins, raise a warning.
        """
        height, bins = self._histogram
        if warn:
            i = np.count_nonzero(height == height.max())
            if i > 1:
                warnings.warn(f"{i} > 1 bins have the same height that is the maximum height", RuntimeWarning)
        i = np.argmax(height)
        mode_left = bins[i]
        mode_right = bins[i+1]
        mode = 0.5 * (mode_left + mode_right)
        return mode

    def _construct_doc(self, *args, **kwargs):
        """
        The original rv_generic._construct_doc is too slow and unnecessary. This is to override and bypass.

        Parameters
        ----------
        docdict
        shapes_vals

        Returns
        -------

        """
        pass

    @property
    def height(self):
        return self._histogram[0]

    @property
    def bins(self):
        return self._histogram[1]
