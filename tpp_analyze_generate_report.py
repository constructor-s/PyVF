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
from pathlib import Path
import bs4
import copy
from PIL import Image
from collections import OrderedDict


def vfarray2matrix(vf, pattern=pyvf.strategy.PATTERN_P24D2, xspacing=6, xmin=-27, xmax=+27, yspacing=6, ymin=-27, ymax=+27, fill_value=np.nan):
    """
    Convert a 1D array of visual field values to a 2D matrix according to the pattern with padding at the corners

    Parameters
    ----------
    vf : 1d array of visual field values

    Returns
    -------
    1d array converted to 2d matrix according to the matrix for visual representation
    """
    xsize = int((xmax - xmin) / xspacing + 1)
    ysize = int((ymax - ymin) / yspacing + 1)

    def loc2ind(x, y):
        xi = int((x - xmin) / xspacing)
        yi = int((ymax - y) / yspacing)

        return yi, xi  # Numpy array index is (y, x)

    ret = np.full((xsize, ysize), fill_value=fill_value)

    for vf_loc, pattern_loc in zip(vf, pattern):
        ind = loc2ind(pattern_loc["xod"], pattern_loc["yod"])
        ret[ind] = vf_loc

    return ret


def grouper(iterable, n, fillvalue=None):
    """Collect data into fixed-length chunks or blocks
    https://docs.python.org/3/library/itertools.html#itertools-recipes"""
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    from itertools import zip_longest
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

parser = ArgumentParser()
parser.add_argument("-i", "--input", required=True, type=str, help="Output from part 1")
parser.add_argument("-o", "--output-dir", required=True, type=str, help="HTML output directory")
parser.add_argument("--id", required=True, type=int, help="Subject ID")
parser.add_argument("--template", required=True, type=str, help="")
args = parser.parse_args()

with open(args.template) as f:
    soup = bs4.BeautifulSoup(f.read(), "html.parser")
stylesheets = soup.find_all("link", {"rel": "stylesheet"})
for s in stylesheets:
    t = soup.new_tag('style')
    with open(Path(args.template).parent / s["href"]) as f:
        c = bs4.element.NavigableString(f.read())
    t.insert(0, c)
    # t['type'] = 'text/css'
    s.replaceWith(t)
# with open("output.html", "wb") as f:
#     f.write(soup.encode("utf-8"))

template_soup = soup  # rename the variable for clarify # TODO: Refactor

df = pd.read_csv(args.input) # , keep_default_na=False)
df = df[df["id"] == args.id]
df = df[df["comment"] != "invalid"]
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["dob"] = pd.to_datetime(df["dob"])
for i in range(54):
    df[f"L{i}"] = pd.to_numeric(df[f"L{i}"])
    df[f"TDp{i}"] = pd.to_numeric(df[f"TDp{i}"])
    df[f"PDp{i}"] = pd.to_numeric(df[f"PDp{i}"])

pages = []

for group_name, group in df.groupby(["id", "eye"]):
    group = group.sort_values("timestamp")
    n_tests = len(group)

    for page_i, test_group in enumerate(grouper(group.itertuples(), n=3)):
        curr_soup = copy.copy(template_soup)

        for i, item in enumerate(test_group):
            if item is None:
                curr_soup.select("#overview-container .vf-overview")[i]["style"] = "visibility: hidden;"
                continue

            if i == 0:
                curr_soup.select_one("#value-name").string = "Anonymous, Anonymous"  # TODO: Remove hard code
                curr_soup.select_one("#value-dob").string = item.dob.strftime("%b %e, %Y")
                curr_soup.select_one("#value-gender").string = ""
                curr_soup.select_one("#value-id").string = str(item.id)
                curr_soup.select_one("#value-side").string = item.eye
                curr_soup.select_one("#value-test-type").string = "Central 24-2 Threshold Test"  # TODO: Remove hard code
                curr_soup.select_one("#value-tpp-version").string = datetime.now().strftime("Report generated with PyVF %b %e, %Y - Authorized research use only")

            curr_soup.select("#overview-container .value-date")[i].string = item.timestamp.strftime("%b %e, %Y")
            curr_soup.select("#overview-container .value-strategy")[i].string = item.strategy

            if item.comment:  # There is already a comment
                processed_comment = item.comment
            else:
                hfl = item.fl_error/item.fl_total > 0.33
                hfp = item.fp_error/item.fp_total > 0.20
                hfn = item.fn_error/item.fn_total > 0.20
                if (hfl & hfp) or (hfl & hfn) or (hfp & hfn):
                    processed_comment = "Poor reliability"
                elif hfl:
                    processed_comment = "High FL"
                elif hfp:
                    processed_comment = "High FP"
                elif hfn:
                    processed_comment = "High FN"
                else:
                    processed_comment = ""
            curr_soup.select("#overview-container .value-comment")[i].string = processed_comment if not pd.isna(processed_comment) else "NIL"

            curr_soup.select("#overview-container .value-fl")[i].string = f"{item.fl_error}/{item.fl_total}={(item.fl_error*100.0/item.fl_total) if item.fl_total else 0.0:.0f}%"
            curr_soup.select("#overview-container .value-fn")[i].string = f"{item.fn_error}/{item.fn_total}={(item.fn_error*100.0/item.fn_total) if item.fn_total else 0.0:.0f}%"
            curr_soup.select("#overview-container .value-fp")[i].string = f"{item.fp_error}/{item.fp_total}={(item.fp_error*100.0/item.fp_total) if item.fp_total else 0.0:.0f}%"
            curr_soup.select("#overview-container .value-ght")[i].string = f"{item.ght}"
            curr_soup.select("#overview-container .value-vfi")[i].string = f"{item.vfi:.0f}%"
            curr_soup.select("#overview-container .value-md")[i].string = f"{item.md:.1f} dB"
            curr_soup.select("#overview-container .value-psd")[i].string = f"{item.psd:.1f} dB"

            if item.eye == "OD":
                vf_matrix = vfarray2matrix(vf=[item.__getattribute__(f"L{i}") for i in range(54)], pattern=pyvf.strategy.PATTERN_P24D2)
                tdp_matrix = vfarray2matrix(vf=[item.__getattribute__(f"TDp{i}") for i in range(54)], pattern=pyvf.strategy.PATTERN_P24D2)
                pdp_matrix = vfarray2matrix(vf=[item.__getattribute__(f"PDp{i}") for i in range(54)], pattern=pyvf.strategy.PATTERN_P24D2)
            elif item.eye == "OS":
                vf_matrix = vfarray2matrix(vf=[item.__getattribute__(f"L{i}") for i in range(54)], pattern=pyvf.strategy.PATTERN_P24D2_OS)
                tdp_matrix = vfarray2matrix(vf=[item.__getattribute__(f"TDp{i}") for i in range(54)], pattern=pyvf.strategy.PATTERN_P24D2_OS)
                pdp_matrix = vfarray2matrix(vf=[item.__getattribute__(f"PDp{i}") for i in range(54)], pattern=pyvf.strategy.PATTERN_P24D2_OS)
            else:
                raise ValueError()

            vf_num_ele = curr_soup.select("#overview-container .vf-num")[i]
            vf_vis_ele = curr_soup.select("#overview-container .vf-vis")[i]
            td_sig_ele = curr_soup.select("#overview-container .td-sig")[i]
            pd_sig_ele = curr_soup.select("#overview-container .pd-sig")[i]
            td_legend = OrderedDict(sorted(((float(k), v) for k, v in json.loads(td_sig_ele["data-legend"])["char"].items()), reverse=False))
            pd_legend = OrderedDict(sorted(((float(k), v) for k, v in json.loads(pd_sig_ele["data-legend"])["char"].items()), reverse=False))

            for r in range(10):
                for c in range(10):
                    if np.isfinite(vf_matrix[r, c]):
                        vf_num_ele.select_one(f".vf10x10-{r}-{c}").string = f"{vf_matrix[r, c]:.0f}"
                    else:
                        vf_num_ele.select_one(f".vf10x10-{r}-{c}").string = ""

                    for ele, mat, leg in zip((td_sig_ele, pd_sig_ele),
                                             (tdp_matrix, pdp_matrix),
                                             (td_legend, pd_legend)):
                        if np.isfinite(mat[r, c]):
                            for k, v in leg.items():  # k are p values sorted in ascending order, v is the legend character
                                if mat[r, c] <= k / 100.0:
                                    ele.select_one(f".vf10x10-{r}-{c}").string = v
                                    break
                        else:
                            ele.select_one(f".vf10x10-{r}-{c}").string = ""

            vf_matrix_20x20 = np.array(Image.fromarray(vf_matrix).resize((20, 20), Image.BILINEAR))
            for r in range(20):
                for c in range(20):
                    ele = vf_vis_ele.select_one(f".vf20x20-{r}-{c}")
                    if np.isfinite(vf_matrix_20x20[r, c]):
                        ele["class"].append(f"v{round(vf_matrix_20x20[r, c])}")

        with open(Path(args.output_dir) / f"output_{group_name[0]}_{group_name[1]}_{page_i}.html", "wb") as f:
            f.write(curr_soup.encode("utf-8"))



