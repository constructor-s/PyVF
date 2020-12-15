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

from pyvf.strategy import ZestStrategy, PATTERN_P24D2, XOD, YOD, TSDISP
from pyvf.strategy.GrowthPattern import SimpleP24d2QuadrantGrowth
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

    # Randomly sample visual field for each eye
    rng = np.random.RandomState(0)
    sample_field_info = VF_THRESHOLD_INFO.groupby("STUDY_SITE_ID").apply(lambda x: x.sample(1, random_state=rng))

    field_names = ["FIELD_ID", "REPEAT", "PRESENTATIONS"]
    field_names.extend([f"L{i}" for i in range(54)])
    FieldSimulationResult = namedtuple("FieldSimulationResult", field_names)

    results = []

    repeats = 1
    import time
    tic = time.perf_counter()
    for i, info in enumerate(sample_field_info.itertuples()):
        n = len(sample_field_info)
        elapsed = time.perf_counter() - tic
        remaining = 0.0 if i == 0 else elapsed * 1.0 / i * (n-i)
        print(f"{i}/{n}, e: {elapsed:.0f} sec, r: {remaining:.0f} sec", end="        \r")
        field_id = info.FIELD_ID
        true_thresholds = VF_THRESHOLD[field_id]
        for rep in range(repeats):
            responder = RampResponder(true_threshold=true_thresholds, fp=0.15, fn=0.15, width=4, seed=i*rep)
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
                else:
                    stimulus = stimulus.copy(**{TSDISP: counter})
                    stimulus = responder.get_response(stimulus)
                    # _logger.debug("%3d: %s\t%s", counter, threshold, stimulus)
                    data.append(stimulus)
                    counter += 1

            res = [field_id, rep, len(data)]
            res.extend(threshold)
            res = FieldSimulationResult(*res)
            results.append(res)

    output_df = pd.DataFrame(results)
    output_df.to_csv("zest_simulate_rotterdam.csv", index=False)
