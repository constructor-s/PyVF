from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvf.stats.regression
import pyvf.strategy.Model
import pyvf.strategy
import glob
import datetime
from collections import namedtuple
import json

pattern = r"E:\GoogleDrive\mobile_perimeter_documents\2020_TPP_Data\Home_Monitoring\tpp_byid\01\*\data.json"
output_csv = r"E:\GoogleDrive\mobile_perimeter_documents\2020_TPP_Data\Home_Monitoring\tpp_byid\01\summary_01_tpp.csv"
dob_txt = r"E:\GoogleDrive\mobile_perimeter_documents\2020_TPP_Data\Home_Monitoring\tpp_byid\01\dob.txt"
files = glob.glob(pattern)
VFResult = namedtuple("VFResult", ["id", "age", "eye", "md", "psd", "vfi", "ght"])

with open(dob_txt) as f:
    dob = datetime.datetime.strptime(f.readline(), "%Y%m%d")

print(len(files))
print(files)

results = []
for f in sorted(files):
    with open(f) as f:
        j = json.load(f)

    acquisition = datetime.datetime.fromtimestamp(j["data"][0]["timestamp"] / 1000.0)
    vf = np.array([i[3] for i in j["locations"]])

    age_days = (acquisition - dob).days
    model = pyvf.strategy.Model.Heijl1987p24d2Model(eval_pattern=pyvf.strategy.PATTERN_P24D2, age=age_days / 365.25)

    result = VFResult(
        id=j["user"]["id"],
        age=age_days,
        eye=j["user"]["side"][0].upper(),
        md=model.get_md(vf),
        psd=0,
        vfi=0,
        ght=""
    )
    results.append(result)

df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
