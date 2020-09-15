"""
Some calculations around the SORS meta-strategy [1] using the Rotterdam2013 dataset

[1] Kucur, Ş. S., & Sznitman, R. (2017). Sequentially optimized reconstruction strategy:
A meta-strategy for perimetry testing. PLoS ONE, 12(10).
https://doi.org/10.1371/journal.pone.0185049


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
from pyvf.strategy import PATTERN_P24D2
from pyvf.plot import VFPlotManager

from sklearn.model_selection import GroupShuffleSplit, PredefinedSplit
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np

import logging
import argparse

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='Train and plot the results of SORS training process')
parser.add_argument('--load', type=str, help='Load a previously trained session using dill.load_session instead of training')
parser.add_argument('--md-upper', type=float, help='Filter fields with an MD upper bound')
parser.add_argument('--md-lower', type=float, help='Filter fields with an MD lower bound')
parser.add_argument('--split-by', type=str, default=None, help='Split test train based on predefined sorted sequence of variable')
parser.add_argument('--suffix', type=str, default="''", help='Suffix to plot titles')
args = parser.parse_args()


if not args.load:
    from pyvf.resources.rotterdam2013 import VF_THRESHOLD, VF_BLINDSPOTS, VF_THRESHOLD_INFO
    mask = np.isfinite(VF_THRESHOLD_INFO["MD"])
    if args.md_upper is not None:
        mask &= VF_THRESHOLD_INFO["MD"] < args.md_upper
    if args.md_lower is not None:
        mask &= VF_THRESHOLD_INFO["MD"] > args.md_lower
    VF_THRESHOLD = VF_THRESHOLD[mask]
    VF_THRESHOLD_INFO = VF_THRESHOLD_INFO[mask]
    _logger.info("VF_THRESHOLD.shape = %s", VF_THRESHOLD.shape)

    Omega_train_all = []
    D_train_all = []
    n_splits = 10
    if args.split_by is None:
        random_state = 0
        train_test_splits = tuple(GroupShuffleSplit(n_splits=n_splits, random_state=random_state)
                                  .split(X=VF_THRESHOLD, y=None, groups=VF_THRESHOLD_INFO["STUDY_SITE_ID"]))
    else:
        val = VF_THRESHOLD_INFO[args.split_by].values
        val_rank = scipy.stats.rankdata(val)
        group_size = np.ceil(len(val_rank) * 1.0 / n_splits)
        test_fold = np.floor(val_rank / group_size)
        train_test_splits = tuple(PredefinedSplit(test_fold=test_fold)
                                  .split(X=VF_THRESHOLD, y=None))

    for train_i, test_i in train_test_splits:
        print(f"{len(train_i) = }, {len(test_i) = })")
        print(f"{VF_THRESHOLD_INFO[['SITE', 'MD', 'AGE', 'IOP']].iloc[train_i].describe() = }")
        print(f"{VF_THRESHOLD[train_i].mean() = }")
        print(f"{VF_THRESHOLD_INFO[['SITE', 'MD', 'AGE', 'IOP']].iloc[test_i].describe() = }")
        print(f"{VF_THRESHOLD[test_i].mean() = }")

        # Implementation of SORS training algorithm in "Algorithm 1"
        # Designed to be an accurate line by line translation, not for efficiency
        M = VF_THRESHOLD.shape[1]
        N = len(train_i)
        X = VF_THRESHOLD[train_i].T  # Training data
        assert X.shape == (M, N)
        Omega = set(range(0, M))  # location set
        S = M

        Omega_star_S = []
        D_star = []

        for k in range(1, S+1):
            error = {}
            D_k = {}

            for l in Omega - set(Omega_star_S):
                Omega_km1_l = Omega_star_S[:k-1] + [l]
                Y_Omega_km1_l = X[Omega_km1_l, :]
                D_l_k = X.dot(Y_Omega_km1_l.T.dot(np.linalg.inv(Y_Omega_km1_l.dot(Y_Omega_km1_l.T))))
                X_hat = D_l_k.dot(Y_Omega_km1_l)
                error[l] = np.linalg.norm(X - X_hat)
                D_k[l] = D_l_k

            l_star_k = min(error, key=error.get)  # Equivalent to: min(d.keys(), key=lambda x: d[x])
            Omega_star_S.append(l_star_k)
            D_star.append(D_k[l_star_k])
            print(f"{len(Omega_star_S) = :2d}, {error[l_star_k] = :6.1f}: {Omega_star_S[-1] = }")

        Omega_train_all.append(Omega_star_S)
        D_train_all.append(D_star)

    try:
        import dill
        import datetime
        dill.dump_session(datetime.datetime.now().strftime("%Y%m%d%H%M%S.pkl"))
    except Exception as e:
        _logger.error(e)

else:
    import dill
    old_args = args
    dill.load_session(args.load)
    args = old_args


def plot_sequence(Omega_train_all, D_train_all):
    n_splits = len(Omega_train_all)
    rows = int(np.floor(np.sqrt(n_splits)))
    cols = int(np.ceil(n_splits * 1.0 / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 10, rows * 10))

    colors = plt.get_cmap("Reds")(np.linspace(0.0, 1.0, len(Omega_train_all[0])))

    for ax, omega, d in zip(axes.ravel(), Omega_train_all, D_train_all):
        plotter = VFPlotManager()
        plotter.pattern = PATTERN_P24D2
        plotter.ax = ax
        plotter.create_axes()
        for i, loc in enumerate(omega):
            plotter.axins[loc].text(0, 0, "%2d" % i, horizontalalignment="center", verticalalignment="center")
            plotter.axins[loc].set(xlim=[-1, 1], ylim=[-1, 1])
            plotter.axins[loc].set_facecolor(colors[i])

    return fig, axes


fig, axes = plot_sequence(Omega_train_all, D_train_all)
fig.savefig("SORS.pdf")


def plot_performance():
    rows = 2
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(11, 8.5), sharex=True, sharey=True)

    train_perf_all = []
    train_bias_all = []
    test_perf_all = []
    test_bias_all = []
    for (train_i, test_i), omega, d in zip(train_test_splits, Omega_train_all, D_train_all):
        X_train = VF_THRESHOLD[train_i].T
        X_test = VF_THRESHOLD[test_i].T

        train_perf = []
        train_bias = []
        test_perf = []
        test_bias = []

        for k in range(1, S+1):  # Using k points to reconstruct
            omega_k = omega[:k]
            D = d[k-1]

            def reconstruct_error(X):
                Y = X[omega_k, :]
                X_hat = D.dot(Y)
                error = X_hat - X
                return error

            def reconstruct_rmse(X):
                return np.sqrt(np.mean(reconstruct_error(X) ** 2.0))

            rmse_train = reconstruct_rmse(X_train)
            rmse_test = reconstruct_rmse(X_test)

            train_perf.append(rmse_train)
            train_bias.append(np.mean(reconstruct_error(X_train)))
            test_perf.append(rmse_test)
            test_bias.append(np.mean(reconstruct_error(X_test)))

        train_perf_all.append(train_perf)
        train_bias_all.append(train_bias)
        test_perf_all.append(test_perf)
        test_bias_all.append(test_bias)

    for ax, perf_all, tit, xlab, ylab in zip(axes.ravel(), (train_perf_all, train_bias_all, test_perf_all, test_bias_all),
                                       ("Training", "Training", "Testing", "Training"),
                                       ("", "", "Number of locations used as reconstruction input", "Number of locations used as reconstruction input"),
                                       ("RMSE", "mean(Predict - Actual)", "RMSE", "mean(Predict - Actual)")):
        xx = np.arange(1, S+1)
        ax.plot(xx, np.array(perf_all).T, '--', alpha=0.5)
        ax.grid(True)
        ax.set_xticks(np.arange(0, 54.1, 6))
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(tit + eval(args.suffix))
    axes.ravel()[-1].legend([str(i) for i in range(len(perf_all))], loc="upper right")

    return fig, axes


fig, axes = plot_performance()
fig.savefig("SORS_performance.pdf")



