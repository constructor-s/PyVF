from argparse import ArgumentParser
from collections import namedtuple
from datetime import datetime

import pyvf.strategy
import pyvf.strategy.Model
import pyvf.parse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import itertools
import pathlib
import json

parser = ArgumentParser()
parser.add_argument("files", nargs="+")
parser.add_argument("--mode", type=str, default="hfa")
parser.add_argument("--meta", type=str, default=None)
args = parser.parse_args()

Entry = namedtuple("Entry", field_names=("id", "eye", "PSD_HFA", "PSD_NW", "PSD_Turpin", "PSD_Heijl", "path"))
entries = []

if pathlib.Path(args.meta).exists():
    with open(args.meta) as f:
        meta = json.load(f)
else:
    meta = {}

for f in itertools.chain(*(glob.glob(i) for i in args.files)):
    if args.mode.lower() == "hfa":
        parser = pyvf.parse.parse(f)

        model = pyvf.strategy.Model.Heijl1987p24d2Model(
            age=(parser.datetime - parser.dob).days / 365.25,
            eval_pattern=pyvf.strategy.PATTERN_P24D2
        )

        td_nbs = np.array(parser.pdf_parser.td)
        md = parser.md
        vf = np.array(parser.pdf_parser.vf)

        psd_reference = parser.psd
        id = parser.id
        eye = parser.laterality

    elif args.mode.lower() == "tpp":
        psd_reference = np.nan
        with open(f) as fp:
            data = json.load(fp)

        if pathlib.Path(f).parent.name in meta.get("categories", {}).get("invalid", []):
            continue

        dob = meta.get("user", {}).get("dob", None)
        if dob is None:
            dob = data["user"]["dob"]
        dob = datetime.strptime(str(dob), "%Y%m%d").date()
        test_start = next(i for i in data["data"] if i.get("message", None) == "Test start")
        test_datetime = datetime.fromtimestamp(float(test_start["timestamp"]) / 1000.0)


        model = pyvf.strategy.Model.Heijl1987p24d2Model(
            age=(test_datetime-datetime(dob.year, dob.month, dob.day)).days/365.25,
            eval_pattern=pyvf.strategy.PATTERN_P24D2
        )

        vf = np.array([i[3] for i in data["locations"]])
        id = str(data["user"]["id"])
        eye = data["user"]["side"].split()[0].upper()
        if eye == "RIGHT":
            eye = "OD"
        elif eye == "LEFT":
            eye = "OS"

    else:
        raise ValueError()

    ent = Entry(path=f,
                PSD_HFA=psd_reference,
                PSD_NW=model.get_psd(vf, method="non-weighted"),
                PSD_Turpin=model.get_psd(vf, method="turpin"),
                PSD_Heijl=model.get_psd(vf, method="heijl"),
                id=id, eye=eye
                )
    entries.append(ent)


df = pd.DataFrame(entries)
df.index.name = "index"
df.to_csv("psd_calculations.csv")

#%%
fig, ax = plt.subplots(1, 1)
ax.plot(df.PSD_HFA, df.PSD_NW-df.PSD_HFA, "x")
ax.plot(df.PSD_HFA, df.PSD_Turpin-df.PSD_HFA, "x")
ax.plot(df.PSD_HFA, df.PSD_Heijl-df.PSD_HFA, "x")
ax.legend(("NW", "Turpin", "Heijl"))
ax.grid(True)
fig.show()
