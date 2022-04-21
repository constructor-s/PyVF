from pyvf.strategy import PATTERN_P24D2, PATTERN_P30D2, PATTERN_P10D2, XOD, YOD

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import ScalarFormatter, NullFormatter

import numpy as np
import pandas as pd
import logging

_logger = logging.getLogger(__name__)


class VFPlotManager:
    """
    Create the inset template for a visual field plot. Once parameters are initialized, call create_axes to create the
    overall figure, axis, and the inset subaxes. These are stored in: self.fig, self.ax, self.axins
    """

    def __init__(self):
        """
        Constructor. Parameters can be set after construction.
        """
        self.pattern = PATTERN_P24D2
        self.xlim = (-30, 30)
        self.ylim = (-30, 30)
        self.delta_deg_x = 6
        self.delta_deg_y = 6
        self.x_major_ticks = np.arange(-30, 30.1, 6)
        self.y_major_ticks = np.arange(-30, 30.1, 6)
        self.x_minor_ticks = np.arange(-27, 27.1, 6)
        self.y_minor_ticks = np.arange(-27, 27.1, 6)
        self.figsize = (8.5, 8.5)

        self.ax = None
        self.fig = None

    def create_axes(self):
        """
        Create three objects:

            self.fig : parent figure - created if self.ax is None

            self.ax : parent axis - created if self.ax is None

            self.axins : list of inset axes corresponding to each location in self.patten

        """
        if self.ax is None:
            _logger.info("Creating new figure and axis")
            self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize)

        # Set the major and minor ticks, and
        # turn off the label for the major ticks, and
        # turn on the label for the minor ticks
        # The major ticks outline the grid of the inset axes, and
        # the minor ticks actually corresponds to the visual field locations
        self.ax.xaxis.set_major_locator(plt.FixedLocator(self.x_major_ticks))
        self.ax.yaxis.set_major_locator(plt.FixedLocator(self.y_major_ticks))
        self.ax.xaxis.set_minor_locator(plt.FixedLocator(self.x_minor_ticks))
        self.ax.yaxis.set_minor_locator(plt.FixedLocator(self.y_minor_ticks))
        self.ax.xaxis.set_major_formatter(NullFormatter())
        self.ax.yaxis.set_major_formatter(NullFormatter())
        self.ax.xaxis.set_minor_formatter(ScalarFormatter())
        self.ax.yaxis.set_minor_formatter(ScalarFormatter())

        self.ax.set(xlim=self.xlim, ylim=self.ylim)
        self.ax.grid(True)
        # self.ax.axhline(y=0, color="k", linewidth=2, zorder=10)  # TODO: zorder not working relative to inset
        # self.ax.axvline(x=0, color="k", linewidth=2, zorder=10)

        # Funtions that map absolute x, y axis locations to relative axes coordinates used to specify insets

        dx2axcoord = lambda dx, xlim: dx * 1.0 / (xlim[1] - xlim[0])
        x2axcoord = lambda x, xlim: dx2axcoord(x - xlim[0], xlim)
        xy2bbox = lambda x, y: (
            x2axcoord(x - self.delta_deg_x * 0.5, self.xlim),
            x2axcoord(y - self.delta_deg_y * 0.5, self.ylim),
            dx2axcoord(self.delta_deg_x, self.xlim),
            dx2axcoord(self.delta_deg_y, self.ylim),
        )

        axins = []
        for i in range(self.pattern.shape[0]):
            x = self.pattern[XOD][i]
            y = self.pattern[YOD][i]
            # print(f"{x = }, {y = }, {xy2bbox(x, y) = }")
            axi = inset_axes(self.ax, width="100%", height="100%", loc=3,
                             bbox_to_anchor=xy2bbox(x, y), bbox_transform=self.ax.transAxes,
                             borderpad=0)

            # Turn off axes ticks and labels for clean look
            axi.tick_params(labelleft=False, labelbottom=False)
            axi.set(xticks=(), yticks=())

            axins.append(axi)

        # Final inset to draw the x, y axis on top of everything
        axi = inset_axes(self.ax, width="100%", height="100%", loc=3,
                         bbox_to_anchor=(0, 0, 1, 1), bbox_transform=self.ax.transAxes,
                         borderpad=0, axes_kwargs={"zorder": 10})
        axi.set(xlim=self.xlim, ylim=self.ylim)
        axi.axhline(y=0, color="k", linewidth=2)
        axi.axvline(x=0, color="k", linewidth=2)
        axi.patch.set_alpha(0)
        axi.axis(False)

        self.axins = axins
        self.top_axin = axi


# pretty_print_vf_positions = ((0, 1, 3),(1, 1, 4),(2, 1, 5),(3, 1, 6),(4, 2, 2),(5, 2, 3),(6, 2, 4),(7, 2, 5),(8, 2, 6),(9, 2, 7),(10, 3, 1),(11, 3, 2),(12, 3, 3),(13, 3, 4),(14, 3, 5),(15, 3, 6),(16, 3, 7),(17, 3, 8),(18, 4, 0),(19, 4, 1),(20, 4, 2),(21, 4, 3),(22, 4, 4),(23, 4, 5),(24, 4, 6),(25, 4, 7),(26, 4, 8),(27, 5, 0),(28, 5, 1),(29, 5, 2),(30, 5, 3),(31, 5, 4),(32, 5, 5),(33, 5, 6),(34, 5, 7),(35, 5, 8),(36, 6, 1),(37, 6, 2),(38, 6, 3),(39, 6, 4),(40, 6, 5),(41, 6, 6),(42, 6, 7),(43, 6, 8),(44, 7, 2),(45, 7, 3),(46, 7, 4),(47, 7, 5),(48, 7, 6),(49, 7, 7),(50, 8, 3),(51, 8, 4),(52, 8, 5),(53, 8, 6),)
def get_pretty_print_grid(pattern=PATTERN_P24D2):
    """
    Convert a test pattern into the 2D grid format that pretty_print_vf wants
    """
    # Rudimentary method to detect spacing by using the first two points
    dx = pattern[XOD][1] - pattern[XOD][0]
    assert dx > 0, "Could not detect grid spacing"
    # Assume that...
    dy = dx

    left = pattern[XOD].min()
    top = pattern[YOD].max()
    # Make grid symmetrical
    left = -max(abs(left), abs(top))
    top = -left

    return [
        (index, int((top - loc[1]) // dy), int((loc[0] - left) // dx))
        for index, loc in enumerate(pattern)
    ], left, top, dx, dy


def pretty_print_vf(x, pattern=None, fmt=False, apply_style=False, outside_val=np.inf, vmin=0, vmax=28):
    """
    Requires pandas 1.3.0

    Parameters
    ------------
    apply_style: Style the output dataframe for use in notebooks
    fmt: Floating point format string
    """
    if len(x) == 52:
        # Insert OD blind spot locations
        x = np.insert(x, [25, 33], np.nan)
    elif len(x) == 74:
        # Insert OD blind spot locations
        x = np.insert(x, [35, 45], np.nan)

    if pattern is None:
        for p in (PATTERN_P24D2, PATTERN_P30D2, PATTERN_P10D2):
            if len(p) == len(x):
                pattern = p
    pretty_print_vf_positions, left, top, dx, dy = get_pretty_print_grid(pattern)

    data = [[outside_val for _ in range(int(-left*2/dx+1))] for _ in range(int(top*2/dy+1))]  # TODO: Remove hard coded table size of 10 x 10
    for idx, i, j in pretty_print_vf_positions:
        data[i][j] = x[idx]
    df = pd.DataFrame(data, columns=np.arange(left, -left+1, dx), index=np.arange(top, -top-1, -dy))
    if not apply_style and not fmt:
        return df
    else:
        styled = df.style
        if apply_style:
            styled = df.style.background_gradient(axis=None, vmin=vmin, vmax=vmax, cmap='gray')
        if fmt:
            styled = styled.format(lambda v: (fmt % v) if np.isfinite(v) else "", na_rep='?')
        return styled
