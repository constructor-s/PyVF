from pyvf.resources.rotterdam2013 import *
import pandas as pd
from scipy.stats import *
import matplotlib.pyplot as plt

ssp = pd.read_csv("zest_simulate_rotterdam_ZestStrategy.csv").merge(VF_THRESHOLD_INFO, how="left", on="FIELD_ID")
msp = pd.read_csv("zest_simulate_rotterdam_ZestMSPStrategy.csv").merge(VF_THRESHOLD_INFO, how="left", on="FIELD_ID")
assert (ssp["FIELD_ID"] == msp["FIELD_ID"]).all()

threshold_truth = VF_THRESHOLD[ssp["FIELD_ID"]]
threshold_ssp = ssp.loc[:, "L0":"L53"].values
threshold_msp = msp.loc[:, "L0":"L53"].values

print(np.mean(threshold_msp - threshold_truth))
print(np.mean(threshold_ssp - threshold_truth))
print(np.sqrt(np.mean((threshold_msp - threshold_truth) ** 2)))
print(np.sqrt(np.mean((threshold_ssp - threshold_truth) ** 2)))
print((msp["PRESENTATIONS"] - ssp["PRESENTATIONS"]).describe())

print(normaltest(msp["PRESENTATIONS"] - ssp["PRESENTATIONS"]))
print(wilcoxon(msp["PRESENTATIONS"], ssp["PRESENTATIONS"]))

print(normaltest((threshold_msp - threshold_truth).ravel()))
print(normaltest((threshold_ssp - threshold_truth).ravel()))
print(wilcoxon(threshold_msp.ravel(), threshold_ssp.ravel()))


fig, ax = plt.subplots()
ax.hist(msp["PRESENTATIONS"], alpha=0.5, label="MSP")
ax.hist(ssp["PRESENTATIONS"], alpha=0.5, label="SSP")
ax.legend()
fig.show()

fig, ax = plt.subplots()
ax.hist(msp["PRESENTATIONS"] - ssp["PRESENTATIONS"], alpha=0.5, label="MSP")
ax.legend()
fig.show()

fig, ax = plt.subplots()
ax.scatter("MD", "PRESENTATIONS", data=msp, label="MSP")
ax.scatter("MD", "PRESENTATIONS", data=ssp, label="SSP")
ax.legend()
ax.set_xlabel("MD (dB)")
ax.set_ylabel("Number of presentations")
fig.show()

fig, ax = plt.subplots()
ax.scatter(msp["MD"], np.sqrt(np.mean((threshold_msp - threshold_truth) ** 2, axis=1)), label="MSP")
ax.scatter(ssp["MD"], np.sqrt(np.mean((threshold_ssp - threshold_truth) ** 2, axis=1)), label="SSP")
ax.legend()
ax.set_xlabel("MD (dB)")
ax.set_ylabel("RMSE (dB)")
fig.show()
