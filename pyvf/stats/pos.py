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
    # Coerce x to array type for vector code
    x_ = np.array(x)

    # First calculate if fp and fn are both zero
    yc = 0.5  # by definition the center is p = 0.5

    xl = center - 1.0 * (yc - yl) / (yr - yl) * width  # x left of ramp
    xr = xl + width
    # Vector code
    p = np.empty_like(x_, dtype=np.float32)
    p[x_ <= xl] = yl
    p[(xl < x_) & (x_ <= xr)] = yl + (x_[(xl < x_) & (x_ <= xr)] - xl) * (yr - yl) * 1.0 / width
    p[x_ > xr] = yr

    if type(x) is not type(p) and np.ndim(x) == 0:
        return p.item()
    else:
        return p
