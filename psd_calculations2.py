import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pyvf.strategy.Model
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, required=True, help="CSV data file")
args = parser.parse_args()

df = pd.read_csv(args.input)

psd_reference = pd.DataFrame(df[["STUDY_ID", "SIDE", "PSD"]])
psd_reference["PSD_HFA"] = df.PSD.apply(lambda x: float(x.strip("dB").strip()))

cols = ["STUDY_ID", "SIDE", "Age", "PSD_TPP"]
cols.extend(f"L{i}" for i in range(1, 54+1))
psd_tpp_calc = pd.DataFrame(df[cols])

# Preallocate space
psd_tpp_calc["PSD_TPP_Turpin"] = np.nan
psd_tpp_calc["PSD_TPP_Heijl"] = np.nan
psd_tpp_calc["PSD_TPP_NW"] = np.nan

for _, row in enumerate(psd_tpp_calc.itertuples()):
    model = pyvf.strategy.Model.Heijl1987p24d2Model(
        age=row.Age,
        eval_pattern=pyvf.strategy.PATTERN_P24D2
    )

    vf = [row.__getattribute__(f"L{i}") for i in range(1, 54+1)]
    vf = np.array(vf)

    i = row.Index

    psd_tpp_calc.loc[i, "PSD_TPP_Turpin"] = model.get_psd(vf, method="turpin")
    psd_tpp_calc.loc[i, "PSD_TPP_Heijl"] = model.get_psd(vf, method="heijl")
    psd_tpp_calc.loc[i, "PSD_TPP_NW"] = model.get_psd(vf, method="non-weighted")

psd_hfa_tpp_summary = pd.merge(psd_reference[["STUDY_ID", "SIDE", "PSD_HFA"]],
                               psd_tpp_calc[["STUDY_ID", "SIDE", "PSD_TPP", "PSD_TPP_Turpin", "PSD_TPP_Heijl", "PSD_TPP_NW"]],
                               on=["STUDY_ID", "SIDE"])
# psd_hfa_tpp_summary = psd_hfa_tpp_summary[psd_hfa_tpp_summary["PSD_HFA"] > 4.0]

#%%
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
for left, right in (("PSD_TPP_Heijl", "PSD_HFA"),
                    ("PSD_TPP_Turpin", "PSD_HFA"),
                    ("PSD_TPP_NW", "PSD_HFA"),):
    x = psd_hfa_tpp_summary["PSD_HFA"]
    y = psd_hfa_tpp_summary[left] - psd_hfa_tpp_summary[right]
    ax.plot(x, y, ".", markersize=5, alpha=0.7,
            label=f"{left}-{right}\n" + fr"$\mu_\Delta={y.mean():.1f}$ dB, $\sigma_\Delta={y.std():.1f}$ dB")

ax.set_title(f"N={len(psd_hfa_tpp_summary)} eyes ({pathlib.Path(args.input).name})")
ax.legend(loc="lower left")
ax.grid()
ax.axis('equal')
ax.set_xticks(np.arange(0.0, 20.1, 2.0))
ax.set_yticks(np.arange(-10.0, +10.1, 2.0))
ax.set_xlabel("Reference PSD reported on HFA thresholds (dB)")
ax.set_ylabel(r"$\Delta$ between calculated PSD from TPP thresholds (dB)")
fig.show()
fig.savefig("PSD_calculations_EV.pdf")
fig.savefig("PSD_calculations_EV.png", dpi=300)

