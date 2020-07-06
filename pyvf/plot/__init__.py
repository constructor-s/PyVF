from pyvf.strategy import PATTERN_P24D2, XOD, YOD

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import ScalarFormatter, NullFormatter

import numpy as np
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
