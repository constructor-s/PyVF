"""
Simulate 24-2 visual fields

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

from pyvf.strategy import ZestStrategy, PATTERN_P24D2, XOD, YOD, TSDISP, Stimulus, STIMULUS_SEEN, STIMULUS_NOT_SEEN, \
    RESPONSE, ZestMSPStrategy, Strategy
from pyvf.strategy.GrowthPattern import SimpleP24d2QuadrantGrowth, SimpleGrowthPattern
from pyvf.strategy.Model import ConstantModel, AgeLinearModel, Heijl1987p24d2Model
from pyvf.strategy.Responder import RampResponder, PerfectResponder
from pyvf.resources.rotterdam2013 import VF_THRESHOLD, VF_THRESHOLD_INFO
import numpy as np
import pandas as pd
from collections import namedtuple
import logging
_logger = logging.getLogger(__name__)


def pretty_print_vf(arr, pattern=PATTERN_P24D2):
    for i, (val, xod, yod) in enumerate(zip(arr, pattern[XOD], pattern[YOD])):
        if i == 0 or last_yod == yod:
            pass
        else:
            print()
        print(f"{val:6.2f}", end="\t")
        last_yod = yod
    print()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    suffix = "_20201221_e0"

    # Randomly sample visual field for each eye
    rng = np.random.RandomState(0)
    sample_field_info = VF_THRESHOLD_INFO.groupby("STUDY_SITE_ID").apply(lambda x: x.sample(1, random_state=rng))

    field_names = ["FIELD_ID", "REPEAT", "PRESENTATIONS"]
    field_names.extend([f"L{i}" for i in range(54)])
    FieldSimulationResult = namedtuple("FieldSimulationResult", field_names)

    results = []

    repeats = 2
    import time
    import sys
    import datetime
    print(datetime.datetime.now())
    tic = time.perf_counter()
    true_threshold = pd.DataFrame(VF_THRESHOLD,
                                  index=np.arange(VF_THRESHOLD_INFO["FIELD_ID"].min(),
                                                  VF_THRESHOLD_INFO["FIELD_ID"].max() + 1),
                                  columns=[f"L{i}" for i in range(VF_THRESHOLD.shape[1])])
    for i, info in enumerate(sample_field_info.itertuples()):
        n = len(sample_field_info)
        elapsed = time.perf_counter() - tic
        remaining = 0.0 if i == 0 else elapsed * 1.0 / i * (n-i)
        print(f"{i}/{n}, e: {elapsed:.0f} sec, r: {remaining:.0f} sec", end="        \r")
        field_id = info.FIELD_ID
        true_thresholds = true_threshold.loc[field_id].values  # THIS WAS WRONG! FIELD_ID IS FROM 1, NOT 0
        for rep in range(repeats):
            responder = RampResponder(true_threshold=true_thresholds, fp=0.0, fn=0.0, width=4, seed=i*rep)
            model = Heijl1987p24d2Model(eval_pattern=PATTERN_P24D2, age=info.AGE)
            strategy = ZestStrategy(
                pattern=PATTERN_P24D2,
                blindspot=[25, 34],
                model=model,
                term_std=1.5,
                rng=0,
                growth_pattern=SimpleP24d2QuadrantGrowth()
            )

            counter = 0
            data = []
            while True:
                stimulus, threshold = strategy.get_stimulus_threshold(data)
                if stimulus is None:
                    break  # Test is finished
                else:  # isinstance(stimulus, Stimulus):  # Single stimulus perimetry
                    # _logger.debug("%3d: %s\t%s", counter, threshold, stimulus)
                    if isinstance(stimulus, Stimulus):
                        # sys.stdout.write("S")
                        stimulus = stimulus.copy(**{TSDISP: counter})
                        stimulus = responder.get_response(stimulus)
                        data.append(stimulus)
                    elif isinstance(stimulus[0], Stimulus):
                        # sys.stdout.write("M")
                        stimulus = [s.copy(**{TSDISP: counter}) for s in stimulus]
                        stimulus = responder.get_response(stimulus)
                        data.extend(stimulus)
                    else:
                        raise ValueError(f"Invalid stimulus object or list of stimuli: {stimulus}")
                # else:  # Multiple stimulus perimetry
                #     total_response = STIMULUS_NOT_SEEN
                #     individual_responded = []
                #     for s in stimulus:
                #         s = s.copy(**{TSDISP: counter})
                #         s = responder.get_response(s)
                #         total_response |= s.response  # Consider total response as an "OR" on individual stimulus
                #         individual_responded.append(s)
                #     for s in individual_responded:
                #         data.append(s.copy(**{RESPONSE: total_response}))
                counter += 1
            # print()

            res = [field_id, rep, sum([1.0/s.multi for s in data])]
            res.extend(threshold)
            res = FieldSimulationResult(*res)
            results.append(res)

    output_df = pd.DataFrame(results)
    output_df["PRESENTATIONS"] = output_df["PRESENTATIONS"].astype(int)
    output_df.to_csv(f"zest_simulate_rotterdam_{strategy.__class__.__name__}_{suffix}.csv", index=False, float_format="%.6f")

# main()
# from line_profiler import LineProfiler
# lp = LineProfiler()
# lp.add_function(ZestStrategy.get_current_estimate)
# lp.add_function(ZestStrategy.get_new_stimulus_at)
# lp.add_function(Strategy.clip_stimulus_intensity)
# lp.add_function(ZestStrategy.get_stimulus_threshold)
# lp.add_function(SimpleGrowthPattern.adjust)
# lp_wrapper = lp(main)
# lp_wrapper()
# with open("output/profile6.py", "w") as f:
#     lp.print_stats(stream=f, output_unit=1e-6)
