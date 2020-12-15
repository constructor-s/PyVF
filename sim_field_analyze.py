from pyvf.resources.rotterdam2013 import *
import pandas as pd

ssp = pd.read_csv("zest_simulate_rotterdam_20201214.csv")
msp = pd.read_csv("zest_simulate_rotterdam_MSPv1.csv")
assert (ssp["FIELD_ID"] == msp["FIELD_ID"]).all()

threshold_truth = VF_THRESHOLD[ssp["FIELD_ID"]]
threshold_ssp = ssp.loc[:, "L0":"L53"].values
threshold_msp = msp.loc[:, "L0":"L53"].values

print((np.mean((threshold_ssp - threshold_truth) ** 1)))
print((np.mean((threshold_msp - threshold_truth) ** 1)))
print(np.sqrt(np.mean((threshold_ssp - threshold_truth) ** 2)))
print(np.sqrt(np.mean((threshold_msp - threshold_truth) ** 2)))
print((msp["PRESENTATIONS"] - ssp["PRESENTATIONS"]).describe())

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.hist(msp["PRESENTATIONS"], alpha=0.5)
ax.hist(ssp["PRESENTATIONS"], alpha=0.5)
fig.show()
