import json
import time
from matplotlib.ticker import ScalarFormatter, NullFormatter

import pyvf.strategy.Model
import pyvf.plot
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from datetime import datetime
from pytz import timezone
from collections import namedtuple
from pathlib import Path
from argparse import ArgumentParser
from pandas.api.types import CategoricalDtype

parser = ArgumentParser()
parser.add_argument("-i", "--input", required=True, type=str, help="Output from part 1")
parser.add_argument("-o", "--output-folder", required=True, type=str, help="Figure export folder")
args = parser.parse_args()

df = pd.read_hdf(args.input, key="df")  # type: pd.DataFrame

tests_start = df[df["comment"]!="invalid"].groupby(["id", "eye"])["timestamp"].min()
tests_end = df[df["comment"]!="invalid"].groupby(["id", "eye"])["timestamp"].max()
tests_duration = tests_end - tests_start
tests_duration_weeks = tests_duration.apply(lambda x: x.days/7.0)
tests_n = df[df["comment"]!="invalid"].groupby(["id", "eye"])["timestamp"].count()

#%%
for name, group in df[(df["comment"]!="invalid") & (df["comment"]!="exclude")].groupby(["id", "eye"]):
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 8.5))
    plotter = pyvf.plot.VFPlotManager()
    if name[1].upper() == "OD":
        plotter.pattern = pyvf.strategy.PATTERN_P24D2
    elif name[1].upper() == "OS":
        plotter.pattern = pyvf.strategy.PATTERN_P24D2_OS
    else:
        raise ValueError(f"{name} is invalid")
    plotter.ax = ax
    plotter.create_axes()

    plotter.ax.set_title("-".join(name))
    plotter.ax.xaxis.set_minor_formatter(NullFormatter())
    plotter.ax.yaxis.set_minor_formatter(NullFormatter())
    for i, axin in enumerate(plotter.axins):
        if i in (25, 34):
            continue
        ax2 = axin.twiny()
        ax2.hist(group[f"L{i}"], bins=np.arange(-4, 40.1, 4), orientation='horizontal', color=plt.cm.gray(mpl.colors.Normalize(-35, 45)(group[f"L{i}"].mean())), zorder=-10)
        ax2.set(xticks=())
        ax2.set_ylim([-1, 40])
        axin.scatter(group["timestamp"], group[f"L{i}"], s=5, c=group[f"L{i}"], cmap=plt.cm.viridis, norm=mpl.colors.Normalize(0, 60), zorder=0)
        lfit = np.poly1d(np.polyfit(pd.to_numeric(group["timestamp"]), group[f"L{i}"], 1))
        axin.plot(group["timestamp"], lfit(pd.to_numeric(group["timestamp"])), 'k--', linewidth=1, zorder=10)
        axin.set_ylim([-1, 40])

    if name[1].upper() == "OD":
        top_left_i = 0
    elif name[1].upper() == "OS":
        top_left_i = 3
    else:
        raise ValueError(f"{name} is invalid")
    plotter.axins[top_left_i].tick_params(labelleft=True, labeltop=True)
    plotter.axins[top_left_i].set(xticks=group["timestamp"].iloc[[0, -1]], yticks=np.arange(0, 40.1, 10))
    plotter.axins[top_left_i].set_xlabel("Threshold (dB)")
    plt.setp(plotter.axins[top_left_i].get_xticklabels(), rotation=90)

    axi = inset_axes(plotter.ax, width="100%", height="100%", loc=3,
                     bbox_to_anchor=(0.03, 0.02, 0.26, 0.17), bbox_transform=plotter.ax.transAxes,
                     borderpad=0, axes_kwargs={"zorder": 10})
    axi.scatter(group["timestamp"], group["md"], s=5, c=group["md"], marker="s", cmap=plt.cm.viridis, norm=mpl.colors.Normalize(-30, 30), zorder=0)
    axi.legend(fontsize="small", facecolor="None", edgecolor="None", title="MD")
    lfit = np.poly1d(np.polyfit(pd.to_numeric(group["timestamp"]), group["md"], 1))
    axi.plot(group["timestamp"], lfit(pd.to_numeric(group["timestamp"])), 'k--', linewidth=1, zorder=+20)
    ylim = axi.get_ylim()
    axi.set_xticks(group["timestamp"].iloc[[0, -1]])
    plt.setp(axi.get_xticklabels(), rotation=90)
    ax2 = axi.twiny()
    ax2.hist(group["md"],
             bins=np.arange(min(-30, np.floor(group["md"].min())), max(0, np.ceil(group["md"].max()))+2, 1.0),
             orientation='horizontal', color=plt.cm.gray(mpl.colors.Normalize(-45, 15)(group["md"].mean())),
             zorder=-10)
    ax2.set(xticks=())
    ax2.grid(axis="y", linestyle="dashed")

    axi = inset_axes(plotter.ax, width="100%", height="100%", loc=3,
                     bbox_to_anchor=(0.73, 0.02, 0.26, 0.17), bbox_transform=plotter.ax.transAxes,
                     borderpad=0, axes_kwargs={"zorder": 10})
    axi.scatter(group["timestamp"], group["psd"], s=5, c=group["psd"], marker="s", cmap=plt.cm.viridis_r,
                norm=mpl.colors.Normalize(-20, 20), zorder=0)
    axi.legend(fontsize="small", facecolor="None", edgecolor="None", title="PSD")
    lfit = np.poly1d(np.polyfit(pd.to_numeric(group["timestamp"]), group["psd"], 1))
    axi.plot(group["timestamp"], lfit(pd.to_numeric(group["timestamp"])), 'k--', linewidth=1, zorder=+20)
    ylim = axi.get_ylim()
    axi.set_xticks(group["timestamp"].iloc[[0, -1]])
    plt.setp(axi.get_xticklabels(), rotation=90)
    ax2 = axi.twiny()
    ax2.hist(group["psd"],
             bins=np.arange(min(0, np.floor(group["psd"].min())), max(20, np.ceil(group["psd"].max())) + 2, 1.0),
             orientation='horizontal', color=plt.cm.gray_r(mpl.colors.Normalize(-10, 30)(group["psd"].mean())),
             zorder=-10)
    ax2.set(xticks=())
    ax2.grid(axis="y", linestyle="dashed")

    fig.show()
    # fig.savefig(Path(args.output_folder) / ("-".join(name)+".pdf"))
    # fig.savefig(Path(args.output_folder) / ("-".join(name)+".png"), dpi=300)
    plt.close(fig)

    break
