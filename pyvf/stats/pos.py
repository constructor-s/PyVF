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

import numpy as np


def pos_ramp(x, center, yl, yr, width):
    """
    Parameters
    ----------
    x : float | array_like, evaluation points
    center : float, where the probability is 0.5. Note this is not the middle of xl and xr
    yl : float, probability on the left edge
    yr : float, probability on the right edge
    width : float, width of the ramp

    References
    ----------------
    .. [1] Turpin, A., McKendrick, A. M., Johnson, C. A., & Vingrys, A. J. (2003).
    Properties of Perimetric Threshold Estimates from Full Threshold, ZEST, and SITA-like Strategies,
    as Determined by Computer Simulation. Investigative Ophthalmology and Visual Science, 44(11), 4787–4795.
    https://doi.org/10.1167/iovs.03-0023
    """
    # First calculate if fp and fn are both zero
    yc = 0.5  # by definition the center is p = 0.5

    xl = center - 1.0 * (yc - yl) / (yr - yl) * width  # x left of ramp
    xr = xl + width

    # if np.isscalar(x):
    try:
        # Optimized code for scalar
        if x <= xl:
            return yl
        elif x > xr:
            return yr
        else:
            return yl + (x - xl) * (yr - yl) * 1.0 / width
    # else:
    except (ValueError, TypeError):
        # Vector code
        # Coerce x to array type for vector code
        x_ = np.asanyarray(x)
        p = np.empty_like(x_, dtype=np.float32)
        p[x_ <= xl] = yl
        p[(xl < x_) & (x_ <= xr)] = yl + (x_[(xl < x_) & (x_ <= xr)] - xl) * (yr - yl) * 1.0 / width
        p[x_ > xr] = yr

        return p
