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
import scipy.stats
import bs4

parser = ArgumentParser()
parser.add_argument("-i", "--input-folders", required=True, nargs="+", type=str, help="Folder containing all subject folders each containing test folders")
parser.add_argument("-o", "--output", required=True, type=str, help="CSV output")
args = parser.parse_args()

field_names = ["id", "comment", "timestamp", "dob", "routine", "duration", "eye", "pattern", "strategy",
               "fl_error", "fl_total", "fp_error", "fp_total", "fn_error", "fn_total",
               "md", "psd", "vfi", "ght", "path"]
field_names.extend([f"L{i}" for i in range(68)])
field_names.extend([f"TD{i}" for i in range(68)])
field_names.extend([f"PD{i}" for i in range(68)])
field_names.extend([f"TDp{i}" for i in range(68)])
field_names.extend([f"PDp{i}" for i in range(68)])
defaults = ["", "", np.nan, np.nan, np.nan, "", "", "", np.nan, np.nan, np.nan, "", ""]  # This needs to be updated...
defaults.extend([np.nan for _ in range(68)])
defaults.extend([np.nan for _ in range(68)])
defaults.extend([np.nan for _ in range(68)])
defaults.extend([np.nan for _ in range(68)])
defaults.extend([np.nan for _ in range(68)])
SummaryEntry = namedtuple("SummaryEntry", field_names=field_names, defaults=defaults)

df_entries = []
for d in args.input_folders:
    print(".", end="")
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

        if (test_data_file.parent / "index.html").exists():
            with open(test_data_file.parent / "index.html") as f:
                soup = bs4.BeautifulSoup(f.read(), "html.parser")
            vfi_str = soup.find(id="vfi_val").string
            vfi = float(vfi_str.strip("%") if vfi_str.upper() != "N/A" else "nan")
            ght = soup.find(id="ght_val").string
            ght = ght if ght.upper() != "N/A" else ""
        else:
            vfi = np.nan
            ght = ""

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

        routine = data["user"]["routine"]

        if routine.upper().startswith("TPP242"):

            model = pyvf.strategy.Model.Model(  # .HFASitaStandardp24d2Model(  # pyvf.strategy.Model.Heijl1987p24d2Model(
                age=(test_datetime-datetime(dob.year, dob.month, dob.day)).days/365.25,
                # eval_pattern=pyvf.strategy.PATTERN_P24D2
                **pyvf.strategy.ModelDefaults.SITAS_P24D2_PARAMETERS
            )
            OFFSET = +1.4

        elif routine.upper().startswith("TPP102"):
            # model = pyvf.strategy.Model.TPP2020p24d2Model(  # pyvf.strategy.Model.Heijl1987p24d2Model(
            #     age=(test_datetime - datetime(dob.year, dob.month, dob.day)).days / 365.25,
            #     eval_pattern=pyvf.strategy.PATTERN_P10D2
            # )
            model = None
        else:
            raise NotImplementedError(f"routine = {routine} is not yet implemented")

        thresholds = np.array(data["locations"])[:, 3]  # tuple(float(loc[3]) for loc in data["locations"])
        kwargs = {f"L{i}": v for i, v in enumerate(thresholds)}
        vf_stats_dict = {}
        if model is not None:
            vf_stats = model.get_vf_stats(thresholds, psd_method="heijl")
            assert vf_stats.shape[0] == 1
            vf_stats_dict = vf_stats.iloc[0].to_dict()
            # kwargs.update({f"TDp{i}": v for i, v in enumerate(scipy.stats.norm.cdf(model.get_td(thresholds) * 1.0 / model.get_std()))})
            # kwargs.update({f"PDp{i}": v for i, v in enumerate(scipy.stats.norm.cdf(model.get_pd(thresholds) * 1.0 / model.get_std()))})
        kwargs.update({f"TD{i}": vf_stats_dict.get(f"TD{i}", np.nan) for i in range(len(thresholds))})
        kwargs.update({f"PD{i}": vf_stats_dict.get(f"PD{i}", np.nan) for i in range(len(thresholds))})
        kwargs.update({f"TDp{i}": vf_stats_dict.get(f"TDp{i}", np.nan) for i in range(len(thresholds))})
        kwargs.update({f"PDp{i}": vf_stats_dict.get(f"PDp{i}", np.nan) for i in range(len(thresholds))})
        se = SummaryEntry(id=p.name,
                          comment="/".join(comments),
                          timestamp=test_datetime,
                          dob=dob,
                          routine=routine,
                          duration=(float(test_stop["timestamp"])-float(test_start["timestamp"])) / 1000.0,
                          eye=eye,
                          path=str(test_data_file),
                          md=vf_stats_dict.get(f"md", np.nan),
                          psd=vf_stats_dict.get(f"psd", np.nan),
                          vfi=vf_stats_dict.get(f"vfi", np.nan),
                          ght=vf_stats_dict.get(f"ght", np.nan),
                          pattern=routine,
                          strategy=routine,
                          fl_error=int(data["reliability"]["fixationLossCatch"]),
                          fl_total=int(data["reliability"]["fixationLossTotal"]),
                          fp_error=int(data["reliability"]["falsePositiveCatch"]),
                          fp_total=int(data["reliability"]["falsePositiveTotal"]),
                          fn_error=int(data["reliability"]["falseNegativeCatch"]),
                          fn_total=int(data["reliability"]["falseNegativeTotal"]),
                          **kwargs)
        df_entries.append(se)

df = pd.DataFrame(df_entries)
for col in ('id', 'comment', 'eye', 'pattern', 'strategy', 'ght'):
    df[col] = df[col].astype('category')
df["dob"] = pd.to_datetime(df["dob"])
df = df.sort_values(["id", "eye", "timestamp"])
try:
    df.to_csv(args.output, index=False, float_format="%.6g")
except PermissionError as e:
    print(e)
    input(f"Please close the following file and then press enter to continue: {args.output}")
    df.to_csv(args.output, index=False, float_format="%.6g")
# df.to_hdf(args.output+".h5", key='df', mode='w', format="table")
