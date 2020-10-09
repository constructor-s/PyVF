import json
import time
import pyvf.strategy.Model
import pandas as pd
import numpy as np
from datetime import datetime
from pytz import timezone
from collections import namedtuple
from pathlib import Path
from argparse import ArgumentParser
from pandas.api.types import CategoricalDtype

parser = ArgumentParser()
parser.add_argument("-i", "--input-folders", required=True, nargs="+", type=str, help="Folder containing all subject folders each containing test folders")
parser.add_argument("-o", "--output", required=True, type=str, help="CSV output")
args = parser.parse_args()

field_names = ["id", "comment", "timestamp", "dob", "duration", "eye", "pattern", "strategy", "md", "psd", "vfi", "ght", "path"]
field_names.extend([f"L{i}" for i in range(54)])
defaults = ["", "", np.nan, np.nan, np.nan, "", "", "", np.nan, np.nan, np.nan, "", ""]
defaults.extend([np.nan for _ in range(54)])
SummaryEntry = namedtuple("SummaryEntry", field_names=field_names, defaults=defaults)

df_entries = []
for d in args.input_folders:
    p = Path(d)

    meta_file = p / "meta.json"
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
    else:
        meta = {}

    for test_data_file in p.glob("*/data.json"):
        comments = []
        for k, v in meta.get("categories", {}).items():
            if test_data_file.parent.name in v:
                comments.append(k)

        with open(test_data_file) as f:
            data = json.load(f)

        test_start = next(i for i in data["data"] if i.get("message", None) == "Test start")
        test_stop = next(i for i in reversed(data["data"]) if i.get("message", None) == "Test stop")

        dob = meta.get("user", {}).get("dob", None)
        if dob is None:
            dob = data["user"]["dob"]
        dob = datetime.strptime(str(dob), "%Y%m%d").date()

        eye = data["user"]["side"].split()[0].upper()
        if eye == "RIGHT":
            eye = "OD"
        elif eye == "LEFT":
            eye = "OS"

        test_datetime = datetime.fromtimestamp(float(test_start["timestamp"]) / 1000.0)
        model = pyvf.strategy.Model.Heijl1987p24d2Model(
            age=(test_datetime-datetime(dob.year, dob.month, dob.day)).days/365.25,
            eval_pattern=pyvf.strategy.PATTERN_P24D2
        )

        thresholds = tuple(float(loc[3]) for loc in data["locations"])
        se = SummaryEntry(id=p.name,
                          comment="/".join(comments),
                          timestamp=test_datetime,
                          dob=dob,
                          duration=(float(test_stop["timestamp"])-float(test_start["timestamp"])) / 1000.0,
                          eye=eye,
                          path=str(test_data_file),
                          md=model.get_md(thresholds),
                          psd=model.get_psd(thresholds),
                          **{
                              f"L{i}": v for i, v in enumerate(thresholds)
                          }
                          )
        df_entries.append(se)

df = pd.DataFrame(df_entries)
for col in ('id', 'comment', 'eye', 'pattern', 'strategy', 'ght'):
    df[col] = df[col].astype('category')
df["dob"] = pd.to_datetime(df["dob"])
df = df.sort_values(["id", "eye", "timestamp"])
df.to_csv(args.output, index=False)
df.to_hdf(args.output+".h5", key='df', mode='w', format="table")
