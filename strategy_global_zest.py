from pyvf.stats.pdf_stats import variance
from pyvf.strategy import *
from pyvf.resources import rotterdam2013
from pyvf.strategy.GrowthPattern import SimpleP24d2QuadrantGrowth
from pyvf.strategy.Model import Model
import numpy as np
import pandas as pd
import scipy.stats
from pyvf.strategy.Responder import RampResponder

#%%
class ZestMultiStrategy(Strategy):
    def __init__(self, pattern, blindspot, model, database, eyes, *args, **kwargs):
        super().__init__(pattern=pattern, blindspot=blindspot, model=model, *args, **kwargs)

        # Parameters determining shape of probability of seeing curve
        self.param["pos_fp"] = 0.05
        self.param["pos_fn"] = 0.05
        self.param["pos_width"] = 4.0
        self.param["entropy_ratio_threshold"] = kwargs.get("entropy_ratio_threshold", 0.1)

        self.database = database

        self.param["bins"] = np.arange(0, 42.1, 2)
        self.param["bins_mid"] = self.param["bins"][:-1] + 0.5
        # self.x = self.bins[:-1] # left, inclusive edges, and assume the bin counts correspond to these

        # Weights = 1 / multiplicity of the eye in database
        counts = pd.value_counts(eyes, sort=False)
        weights = 1.0 / counts / len(counts)
        self.database_p0 = weights[eyes].values
        self.database_p0_entropy = scipy.stats.entropy(self.database_p0)
        assert np.allclose(self.database_p0.sum(), 1.0), "Sum of weights all visual field entries in database must equal to one"

        self.deg2index = {(x, y): i for i, (x, y) in enumerate(PATTERN_P24D2[[XOD, YOD]])}

    def get_stimulus_threshold(self, data):
        M = len(self.param['pattern'])
        data = Stimulus.to_numpy(data)
        # points_not_tested_yet = list(set(range(M)) - set(Stimulus.to_numpy(data)[LOC]))

        if len(data) == 0:
            # Special case at the beginning
            log_p = np.log(self.database_p0)
        else:
            log_p_updates = np.ones((len(data), len(self.database_p0)), dtype=np.float64)
            for trial_i in range(len(data)):
                loc_i = self.deg2index[tuple(data[[XOD, YOD]][trial_i])]
                response = data[RESPONSE][trial_i]
                threshold = data[THRESHOLD][trial_i]

                # If the response is true...
                p_update = pos_ramp(
                    x=self.database[:, loc_i],
                    center=threshold,
                    yl=self.param["pos_fp"],
                    yr=1-self.param["pos_fn"],
                    width=self.param["pos_width"]
                )
                # otherwise...
                if not response:
                    p_update = 1 - p_update
                log_p_updates[trial_i, :] = np.log(p_update)

            # self.extra_data["log_p_updates"] = log_p_updates
            log_p = np.log(self.database_p0) + log_p_updates.sum(axis=0)

        p = np.exp(log_p - log_p.max())  # Prevent underflowing
        p = p * 1.0 / p.sum()

        hist2d = self.get_hist2d(self.database, p, bins=self.param["bins"])
        mean_per_loc = p @ self.database
        var_per_loc = np.array([
            variance(self.param["bins_mid"], counts) for counts in hist2d.T
        ]).T
        # var_per_loc = np.array([
        #     self.get_var(self.database[:, loc_i], p, bins=self.param["bins"]) for loc_i in range(M)
        # ])
        # self.extra_data["var_per_loc"] = var_per_loc

        # if np.all(var_per_loc < 0.01):
        if scipy.stats.entropy(p) / self.database_p0_entropy < self.param["entropy_ratio_threshold"]:
            stimulus = None
        else:
            # if points_not_tested_yet:
            #     loc_i = points_not_tested_yet[var_per_loc[points_not_tested_yet].argmax()]
            #     print(f"Testing {loc_i}")
            # else:
            loc_i = var_per_loc.argmax()
            stimulus = self.get_new_stimulus_at(db=mean_per_loc[loc_i], location=self.param["pattern"][loc_i])

        self.extra_data = locals()

        return stimulus, mean_per_loc

    @staticmethod
    def get_hist2d(database, p, bins):
        M = database.shape[1]
        result = np.full((len(bins)-1, M), np.nan, dtype=np.float64)
        for m in range(M):
            result[:, m], _ = np.histogram(database[:, m], weights=p, bins=bins, density=True)
        return result

    # @staticmethod
    # def get_var(observations, weights, bins):
    #     p, _ = np.histogram(observations, weights=weights, bins=bins, density=True)
    #     x = bins[:-1]
    #     return variance(x, p)


#%%
rng = np.random.default_rng(0)

strategy_perf_all = []

for true_index in range(len(rotterdam2013.VF_THRESHOLD)):
    mask = rotterdam2013.VF_THRESHOLD_SITES != rotterdam2013.VF_THRESHOLD_SITES[true_index]
    database = np.clip(rotterdam2013.VF_THRESHOLD, 0, 33)[mask, :]
    eyes = rotterdam2013.VF_THRESHOLD_SITES[mask]
    # gb = pd.DataFrame(database).groupby(eyes).median()
    strategy = ZestMultiStrategy(
        pattern=PATTERN_P24D2,
        blindspot=(25, 34),
        model=Model(eval_pattern=PATTERN_P24D2, age=65),
        database=database,
        eyes=eyes,
        term_std=1.5,
        rng=0,
        growth_pattern=SimpleP24d2QuadrantGrowth(),
        entropy_ratio_threshold=0
    )
    truth_clipped = rotterdam2013.VF_THRESHOLD[true_index].clip(0, 33)
    responder = RampResponder(true_threshold=np.where(rotterdam2013.VF_THRESHOLD[true_index] < 0, -10, truth_clipped), fp=0.05, fn=0.05, width=4, seed=int(rng.random() * 256 * 256))

    counter = 0
    data = []
    while counter < 60:
        stimulus, threshold = strategy.get_stimulus_threshold(data)
        strategy_perf = ({
            "field_id": true_index,
            "site_id": rotterdam2013.VF_THRESHOLD_SITES[true_index],
            "trials": counter,
            "rmse": np.sqrt(np.mean((threshold.clip(0, 33) - truth_clipped) ** 2)),
            "mae": np.mean(np.abs(threshold.clip(0, 33) - truth_clipped)),
            "sampled": Stimulus.to_numpy(data)[LOC]
        })
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
        # print(counter, np.sqrt(np.mean(((threshold.clip(0, 33) - responder.threshold.clip(0, 33)) ** 2))))
        strategy_perf_all.append(strategy_perf)
    print(true_index)

    if true_index > 500:
        break
#%%
truth = responder.threshold.clip(0, 33)
pred = threshold
df = pd.DataFrame(data)
df["truth"] = truth[df["loc"]]
df["pred"] = pred[df["loc"]]

from pyvf.plot import pretty_print_vf

print(pretty_print_vf(truth, fmt="%.0f"))

print(pretty_print_vf(pred, fmt="%.0f"))

print(pretty_print_vf(pred-truth, fmt="%.0f"))

import matplotlib.pyplot as plt
# plt.plot(np.array(strategy.extra_data["p_history"]).max(axis=1))
# plt.plot(scipy.stats.entropy(np.array(strategy.extra_data["p_history"]), axis=1))
# plt.show()
