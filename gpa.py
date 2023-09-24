from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvf.stats.regression

parser = ArgumentParser(description="Linear regression from summary table")
parser.add_argument("-i", "--input", type=str, required=True, help="Summary CSV file")
args = parser.parse_args()

df = pd.read_csv(args.input)
re = df[(df.eye == "R") & (df.age > 27087)].iloc[::2]  # 27087)]
le = df[(df.eye == "L") & (df.age > 27087)].iloc[::2]  # 27087)]

for eyedata, tit in zip((re, le), ("Right Eye", "Left Eye")):
    x = eyedata.age.values / 365.25
    y = eyedata.md.values

    model = pyvf.stats.regression.BayesianLinearRegression(measurement_std=np.std(y), slope_std=1.0, intercept_std=2.0)
    model.fit(x, y)

    xpred = np.linspace(x[0], x[-1] + 5, 100)
    xext = np.linspace(x[-1], x[-1] + 5, 100)
    ypred, ypred_std = model.predict(xpred)

    fig, ax = plt.subplots(figsize=(8.5, 2.5), dpi=150)
    ax.plot(x, y, "ks", markersize=3)
    ax.plot(xpred, ypred, "b--",
            xext, np.full_like(xext, fill_value=y[-1]), "k--")
    ax.plot(xpred[xpred<=x[-1]], (ypred+1.96*ypred_std)[xpred<=x[-1]], "-",
            xpred[xpred<=x[-1]], (ypred-1.96*ypred_std)[xpred<=x[-1]], "-", color="gray")
    ax.set_ylim([-30, 5])
    ax.set_title(fr"{tit}: MD slope = ${model.Mu[0]:.2f} \pm ${1.96*np.sqrt(model.Sigma[0, 0]):.2f} dB/yr (95% CI)")
    ax.set_xticks(np.arange(np.floor(xpred[0]), np.ceil(xpred[-1]), 1))
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("MD (dB)")
    ax.grid(True)
    fig.show()

    print(len(eyedata))
    print(eyedata.age.values[-1] - eyedata.age.values[0])
    print(eyedata.md.std())
