"""
Interface and implementations for normal population average values for use by perimetry strategies as starting values,
also for analysis of results against this baseline.
Companion GrowthPattern classes codes for interface and implementations for during-test adjustments of the starting values.


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

import numpy as np
import pandas  # Do not import as pd to avoid collision with pattern deviation
import warnings
from pyvf.strategy import XOD, YOD, PATTERN_P24D2
from .ModelDefaults import DEFAULT_PARAMETERS
import logging
_logger = logging.getLogger(__name__)

nan = np.nan

def wtd_var(x, weights=None, normwt=False, method="unbiased"):
    '''
    R sginature:
    function (x, weights = NULL, normwt = FALSE, na.rm = TRUE, method = c("unbiased", "ML"))

    https://github.com/harrelfe/Hmisc/blob/8bbb192103e0091398cb58b257ed590e481cfa35/R/wtd.stats.s#L15

    This implementation is consistent with the definition of weighted variance with frequency weights
    # https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Frequency_weights
    but different from described in Heijl's A package for the statistical analysis of visual fields
    The weights passed in should be the inverse of variances, not standard deviations.
    The return is variance (i.e. square of PSD), not standard deviation
    '''
    if weights is None:
        return np.var(x)

    x = np.asarray(x)
    weights = np.asarray(weights)

    if normwt:
        weights = weights * len(x) * 1.0 / np.sum(weights)  # Normalizes sum(weights) to len(x) = n

    if normwt or method.upper() == "ML":
        # R stats::cov.wt
        # function (x, wt = rep(1/nrow(x), nrow(x)), cor = FALSE, center = TRUE, method = c("unbiased", "ML"))
        wt = weights * 1.0 / np.sum(weights)
        center = np.dot(wt, x)
        x = np.sqrt(wt) * (x - center)  # R: sqrt(wt) * sweep(x, 2, center, check.margin = FALSE)
        # R:   cov <- switch(match.arg(method), unbiased = crossprod(x)/(1 -
        #     sum(wt^2)), ML = crossprod(x))
        # TODO: R crossprod is "matrix cross-product", which is dot product?
        if method.lower() == "unbiased":
            cov = np.dot(x, x) / (1.0 - np.sum(wt ** 2))
        elif method.upper() == "ML":
            cov = np.dot(x, x)
        else:
            raise ValueError(f"Invalid method = {method}")
        return cov

    sw = np.sum(weights)
    if sw <= 1:
        warnings.warn('only one effective observation; variance estimate undefined')

    xbar = np.sum(weights * x) * 1.0 / sw # This is essentially MD
    ret = np.sum(weights * ((x - xbar) ** 2)) * 1.0 / (sw - 1)
    return ret

class Model:
    """
    Base class for models for calculating normal healthy population threshold values
    """
    def __init__(self, eval_pattern, age=None, *args, **kwargs):
        """

        Parameters
        ----------
        eval_pattern : see Strategy
        age : age of the patient in years
        args
        kwargs
        """
        self.args = args
        self.param = DEFAULT_PARAMETERS
        self.param.update(kwargs)
        self.param['eval_pattern'] = eval_pattern
        self.param['age'] = age

    def get_mean(self):
        raise NotImplementedError()

    def get_std(self):
        raise NotImplementedError()

    def _get_non_blindspot_mask(self):
        try:
            non_blindspot = ~(self.param['eval_pattern']["blindspot"])
        except ValueError:
            _logger.warning("No blindspot field available in eval_pattern, falling back to using valid std values")
            std = self.get_std()
            non_blindspot = std > 0
        return non_blindspot

    def get_td(self, vf):
        vf = np.asarray(vf)
        mean = self.get_mean()
        assert mean.shape == vf.shape
        td = vf - mean
        non_blindspot = self._get_non_blindspot_mask()
        td[~non_blindspot] = np.nan
        return td

    def get_gh(self, vf):
        td = self.get_td(vf)
        non_blindspot = self._get_non_blindspot_mask()
        td_nbs = td[non_blindspot]
        # TODO: Make sure general height is calculated at 85% percentile without blindspots, though the difference is likely insignificant anyways
        general_height = np.percentile(td_nbs, self.param['gh_percentile'] * 100)
        return general_height

    def get_pd(self, vf):
        td = self.get_td(vf)
        general_height = self.get_gh(vf)
        pd = td - general_height
        return pd

    def get_md(self, vf):
        non_blindspot = self._get_non_blindspot_mask()

        std = self.get_std()
        std_valid = np.isfinite(std) & (std > 0)

        std = std[non_blindspot & std_valid]
        weights = 1.0 / (std ** 2)
        weights_normalized = weights * 1.0 / np.sum(weights)
        assert np.allclose(weights_normalized.sum(), 1.0)

        td_nbs = self.get_td(vf)[non_blindspot & std_valid]

        return np.dot(td_nbs, weights_normalized)

    def get_psd(self, vf=None, td_nbs=None, md=None, method="r"):
        non_blindspot = self._get_non_blindspot_mask()

        std = self.get_std()
        std_valid = np.isfinite(std) & (std > 0)

        std = std[non_blindspot & std_valid]
        weights = 1.0 / (std ** 2)

        if td_nbs is None:
            td_nbs = self.get_td(vf)[non_blindspot & std_valid]

        if method.lower() == "r" or method.lower() == "turpin":
            # Weighted standard deviation of TD (i.e. Marin-Franch2013 R package)
            # Weight is normalized by wtd_var
            return np.sqrt(wtd_var(td_nbs, weights, normwt=True))
        elif method.lower() == "non-weighted":
            # Non-weighted standard deviation of TD, which is similar to above
            return td_nbs.std()
        elif method.lower() == "heijl":
            # "Heijl package" original formula - seems to overestimate
            N = len(std)
            if md is None:
                md = self.get_md(vf)
            td = td_nbs
            psd_squared = 1.0 / N * np.sum(std ** 2) / (N-1) * np.sum(
                ((td - md) ** 2) * weights
            )
            return np.sqrt(psd_squared)
        else:
            raise ValueError("Invalid method: " + method)

    def get_vf_stats(self, vf):
        """
        Vectorized version of visual field indices calculation
        Currently assumes 24-2 visual fields

        Parameters
        ----------
        vf : ndarray
            Array of N x 54 visual field data (54 for 24-2 pattern)

        Returns
        -------
            Dataframe containing the calculated statistics
        """
        # Format VF and age input
        vf = np.atleast_2d(vf)
        N, M = vf.shape
        age = np.array(self.param['age']).reshape(N, 1)

        # calculate baseline
        mean = self._get_vf_stats_mean(age)

        ## Total Deviation
        total_deviation = vf - mean
        total_deviation_p = self._get_vf_stats_probability_map(total_deviation, self.param["td_thresholds"])

        ## General Height
        general_height = np.nanpercentile(total_deviation,
                                          self.param['gh_percentile'] * 100,
                                          axis=1, keepdims=True)

        ## Pattern Deviation
        pattern_deviation = total_deviation - general_height
        pattern_deviation_p = self._get_vf_stats_probability_map(pattern_deviation, self.param["pd_thresholds"])

        ## MD, PSD
        mask = self.param['md_weights'] != 0  # Need this mask to remove nans
        mean_deviation = total_deviation[:, mask] @ self.param['md_weights'][mask].reshape(-1, 1)
        pattern_standard_deviation = np.apply_along_axis(wtd_var, axis=1, arr=pattern_deviation[:, mask],
                                                         weights=self.param['psd_weights'][mask], normwt=True)
        pattern_standard_deviation = np.sqrt(pattern_standard_deviation)  # Take the sqrt of the var

        # Save data into a dataframe
        # Generate the column headers
        columns_headers = ["gh", "md", "psd"]
        columns_headers.extend([f"TD{i}" for i in range(total_deviation.shape[1])])
        columns_headers.extend([f"PD{i}" for i in range(pattern_deviation.shape[1])])
        columns_headers.extend([f"TDP{i}" for i in range(total_deviation_p.shape[1])])
        columns_headers.extend([f"PDP{i}" for i in range(pattern_deviation_p.shape[1])])
        # Concatenate data columns
        data = np.column_stack([general_height, mean_deviation, pattern_standard_deviation,
                                total_deviation, pattern_deviation, total_deviation_p, pattern_deviation_p])
        return pandas.DataFrame(data=data, columns=columns_headers)

    def _get_vf_stats_mean(self, age):
        """
        Helper function for get_vf_stats
        to obtain the baseline (normative mean values)
        """
        intercept = np.array(self.param["intercept"]).reshape(1, -1)
        slope = np.array(self.param["slope"]).reshape(1, -1)
        return slope * age + intercept

    @staticmethod
    def _get_vf_stats_probability_map(total_deviation, thresholds_dict):
        """

        Parameters
        ----------
        total_deviation : ndarray
            N x M array containing the visual field information for N visual fields.
            This could either be the total deviation or pattern deviation

        thresholds_dict : dict
            Dictionary mapping p value thresholds (0.05, 0.01, 0.005, ...)
            to an array of threshold values with length M

        Returns
        -------
        ndarray
            Probability map with same size as input total_deviation
        """
        total_deviation_p = np.ones_like(total_deviation, dtype=np.float64)
        # Iterate from high to low p = 0.05, 0.01, 0.005, ...
        for p in sorted(thresholds_dict, reverse=True):
            total_deviation_p = np.where(total_deviation < thresholds_dict[p], p, total_deviation_p)
        total_deviation_p = np.where(np.isnan(total_deviation), np.nan, total_deviation_p)
        return total_deviation_p


class AgeLinearModel(Model):
    def __init__(self, eval_pattern, age, model_pattern, intercept, slope, std=None, *args, **kwargs):
        """

        Parameters
        ----------
        eval_pattern
        age
        model_pattern
        intercept
        slope
        std : 
        
            standard deviation at each of the model_pattern location. Optional. 
            If not provided, this is set to None and any operation that involves std will cause undefined behavior
            Si^2 should be "the variance of normal field measurements at point i" (? intra-individual)
        
        args
        kwargs
        """
        super().__init__(eval_pattern, age, *args, **kwargs)
        self.param["intercept"] = intercept
        self.param["slope"] = slope
        self.param["std"] = std
        self.param["model_pattern"] = model_pattern

        # If the input model_pattern is not the same as output eval_pattern, then interpolation must be performed.
        # However this is not currently implemented
        if (len(model_pattern) != len(eval_pattern) or
                not np.all(model_pattern[XOD] == eval_pattern[XOD]) or
                not np.all(model_pattern[YOD] == eval_pattern[YOD])):
            raise ValueError("currently model_pattern must match eval_pattern exactly")

    def get_mean(self):
        return self.param["intercept"] + self.param["slope"] * self.param['age']

    def get_std(self):
        if self.param["std"] is None:
            raise ValueError("std model values were not defined")
        else:
            return self.param["std"]


class Heijl1987p24d2Model(AgeLinearModel):
    """
    References
    --------------------

    .. [1] Heijl A, Lindgren G, Olsson J. Normal variability of static perimetric threshold values across the central
    visual field. Arch Ophthalmol. 1987;105(11):1544‐1549. doi:10.1001/archopht.1987.01060110090039
    """
    def __init__(self, eval_pattern, age, *args, **kwargs):
        model_pattern = PATTERN_P24D2
        intercept = [29.9,28.65,29.25,28.9,31,31.45,31.65,30.9,31.6,31.7,30.75,33.05,33.8,33.95,33.3,32.25,31.95,31.9,29.7,31.95,34,34,34.8,34.95,34.1,0.0,32.7,30.05,32.75,33.9,34.9,35.15,35.35,35.25,0.0,33.15,31.95,33.05,34.4,33.4,34.05,34.35,32.55,32.1,32.55,32.45,33.25,33.1,33.5,33.25,30.9,31.9,32.7,32.65]
        intercept = np.array(intercept)
        slope = [-0.082,-0.049,-0.075,-0.072,-0.066,-0.063,-0.057,-0.056,-0.072,-0.082,-0.067,-0.073,-0.054,-0.061,-0.064,-0.051,-0.063,-0.08,-0.07,-0.067,-0.066,-0.046,-0.05,-0.055,-0.068,0.0,-0.066,-0.081,-0.075,-0.054,-0.058,-0.049,-0.053,-0.077,0.0,-0.065,-0.073,-0.057,-0.048,-0.036,-0.055,-0.063,-0.049,-0.058,-0.073,-0.055,-0.061,-0.06,-0.06,-0.065,-0.056,-0.068,-0.066,-0.067]
        slope = np.array(slope)
        std = [4.7, 4.2, 3.7, 4.3, 3.4, 3.1, 2.8, 3.0, 2.9, 3.0, 3.4, 3.3, 2.4, 2.3, 2.1, 2.2, 3.5, 3.4, 4.2, 2.7, 2.3, 2.3, 2.0, 2.2, 2.4, -1, 3.7, 5.8, 2.2, 2.3, 1.9, 1.6, 1.8, 2.1, -1, 3.3, 3.0, 2.4, 1.8, 2.4, 2.2, 2.4, 3.9, 3.5, 2.7, 3.4, 2.4, 2.4, 2.3, 2.5, 3.7, 2.4, 2.4, 3.0]
        std = np.array(std)

        super().__init__(eval_pattern, age, model_pattern, intercept, slope, std, *args, **kwargs)

class TPP2020p24d2Model(AgeLinearModel):
    """
    TPP values fitted from home monitoring study in Jan 2021
    """
    def __init__(self, eval_pattern, age, *args, **kwargs):
        model_pattern = PATTERN_P24D2
        # intercept = [32.08824275, 30.92295853, 32.17743015, 31.51870534, 32.87456178,
        # 33.51907916, 33.09017278, 32.52224245, 33.24031098, 33.57482224,
        # 32.9039818 , 34.8688908 , 34.45368195, 34.69178137, 33.40556135,
        # 32.09998805, 33.53526918, 33.7216212 , 30.69405089, 33.81792501,
        # 35.03093398, 34.27810253, 35.32294172, 33.85958032, 32.76768885,
        #  0.        , 33.98641634, 31.22363348, 34.10637434, 34.0330131 ,
        # 34.15773038, 34.90653608, 34.08869632, 34.19556307,  0.        ,
        # 34.00392901, 33.03076778, 33.3319171 , 32.9674811 , 32.97891129,
        # 34.19093765, 34.06157483, 33.24127153, 32.9432472 , 32.79471081,
        # 32.27178169, 32.76706453, 32.93238227, 32.93598955, 32.9344832 ,
        # 30.71397004, 31.62929386, 31.66282881, 31.68927266]
        intercept = [32.52437374, 31.10166366, 32.18969273, 31.4311408 , 33.2118529 ,
        33.34974079, 32.80304099, 32.54477052, 33.24969843, 33.2458298 ,
        33.30460588, 34.64964118, 33.62092344, 33.74851936, 33.69859872,
        33.01885622, 33.58922696, 33.33427971, 30.72212914, 33.97907344,
        34.62508117, 33.73645884, 34.61501654, 34.716843  , 34.63091328,
         0.        , 33.67936936, 31.05566242, 34.16560828, 33.81573319,
        34.13586996, 34.41654415, 34.76220214, 35.27454074,  0.        ,
        33.82726832, 33.00654489, 33.29590225, 32.86616658, 32.29096905,
        33.43526476, 34.04559391, 33.40445721, 32.78204474, 32.78835858,
        32.38316565, 32.7712431 , 32.92044364, 33.15209245, 33.09149308,
        30.59278211, 31.79149986, 31.89089307, 31.69537519]
        # intercept = [32.14070011, 30.90745665, 32.08773465, 31.51690899, 32.97554153,
        # 33.52024506, 33.14489741, 32.65014495, 33.13485751, 33.39425567,
        # 33.03968531, 34.79294491, 34.22073664, 34.56309261, 33.68122244,
        # 32.39750573, 33.27276662, 33.88753583, 30.71863631, 33.87590101,
        # 34.85229625, 34.15662783, 35.57210519, 33.39182461, 33.15082136,
        #  0.        , 33.90060832, 31.18849005, 34.11482724, 33.97633588,
        # 34.28259162, 34.77789118, 34.44908889, 34.1165892 ,  0.        ,
        # 33.99378831, 32.95890952, 33.39431444, 33.15833742, 32.88078804,
        # 33.78872192, 33.72915595, 33.09053905, 33.13680472, 32.73749681,
        # 32.49466191, 33.01947245, 33.05190841, 33.02567726, 33.04283152,
        # 30.43891972, 31.64270156, 31.72723333, 31.59517329]
        # intercept = [32.54924189, 30.17563325, 32.17257295, 31.1116725 , 32.43431604,
        # 33.63267485, 33.39976285, 32.51526517, 33.22278361, 33.40109375,
        # 33.0965915 , 35.14431833, 34.48701055, 33.5673832 , 33.73939476,
        # 33.01870269, 33.3734156 , 34.81728357, 31.23156603, 34.02251743,
        # 35.36024197, 34.48510318, 35.46753834, 34.00113786, 32.16627437,
        #     0.        , 34.10098271, 30.62770718, 33.83923961, 34.11664018,
        # 34.3583346 , 34.76822585, 34.12103138, 33.8304572 ,  0.        ,
        # 34.06833688, 33.18135536, 32.80717025, 32.77261706, 32.93642267,
        # 34.51965637, 34.59563782, 32.69415177, 31.98796423, 32.88505907,
        # 32.11224897, 32.41923505, 32.94214478, 33.12732996, 32.66092947,
        # 30.51526517, 31.57368178, 32.61950123, 31.98548039]
        intercept = np.array(intercept)
        slope = [-0.082,-0.049,-0.075,-0.072,-0.066,-0.063,-0.057,-0.056,-0.072,-0.082,-0.067,-0.073,-0.054,-0.061,-0.064,-0.051,-0.063,-0.08,-0.07,-0.067,-0.066,-0.046,-0.05,-0.055,-0.068,0.0,-0.066,-0.081,-0.075,-0.054,-0.058,-0.049,-0.053,-0.077,0.0,-0.065,-0.073,-0.057,-0.048,-0.036,-0.055,-0.063,-0.049,-0.058,-0.073,-0.055,-0.061,-0.06,-0.06,-0.065,-0.056,-0.068,-0.066,-0.067]
        slope = np.array(slope)
    #     std = [2.51770021, 2.13459298, 2.4863459 , 3.68106809, 1.65263932,
    #    1.28016473, 1.5212036 , 1.94431405, 2.08038352, 2.50467718,
    #    1.9896718 , 1.74976848, 1.35442357, 1.52576719, 2.50131701,
    #    2.82271877, 2.29607996, 1.78326638, 1.01276521, 2.97402615,
    #    1.99866316, 1.9495726 , 1.98110493, 3.08474073, 3.56418446,
    #    -1.        , 1.62804426, 3.52970961, 4.06292303, 2.60259655,
    #    2.733647  , 2.22228512, 2.64674727, 3.03622126, -1.        ,
    #    1.44752368, 5.96511175, 3.62592159, 2.84219778, 2.56595279,
    #    2.22946501, 2.081675  , 1.81007356, 1.3773106 , 6.19039693,
    #    3.61970007, 2.76524966, 2.30054966, 2.05775936, 2.50880104,
    #    8.87279627, 5.94861873, 5.0309542 , 5.65476843]
        # std = [1.63461607, 1.25088713, 1.59290228, 1.71437139, 1.47416423,
        # 1.05462289, 1.15357528, 1.45888827, 1.68902749, 1.77308521,
        # 1.66739162, 1.56618073, 0.77214773, 1.49812292, 1.50468245,
        # 1.57565351, 1.1538098 , 1.30537517, 1.08412392, 1.5008295 ,
        # 1.31265073, 1.40168421, 1.40334326, 1.72795327, 1.99428295,
        # 1.85852429, 1.30748759, 1.85864075, 1.90720893, 1.88881992,
        # 1.26135535, 1.64924881, 1.66949976, 1.66362276, 5.06093818,
        # 1.22005251, 2.30445174, 2.21237175, 2.00394632, 1.17789893,
        # 1.36772475, 1.56462518, 1.44917502, 1.55967168, 1.4943617 ,
        # 1.78060742, 1.80586466, 1.75322831, 1.37325253, 1.57335085,
        # 3.57325927, 2.87265809, 1.43053437, 2.01003974]
        std = [4.10548856, 3.87803847, 3.87803847, 4.10548856, 3.76024984,
       3.25552549, 2.96518859, 2.96518859, 3.25552549, 3.76024984,
       3.86225527, 3.13951138, 2.53117223, 2.15112972, 2.15112972,
       2.53117223, 3.13951138, 3.86225527, 4.33121599, 3.51284386,
       2.73765823, 2.05548035, 1.5938977 , 1.5938977 , 2.05548035,
       0.        , 3.51284386, 4.1960993 , 3.38236138, 2.61648374,
       1.95515724, 1.53641439, 1.53641439, 1.95515724, 0.        ,
       3.38236138, 3.46554251, 2.76238814, 2.18800517, 1.84939996,
       1.84939996, 2.18800517, 2.76238814, 3.46554251, 3.10498978,
       2.62466641, 2.35550767, 2.35550767, 2.62466641, 3.10498978,
       3.17363582, 2.95656187, 2.95656187, 3.17363582]
        std = np.array(std)

        super().__init__(eval_pattern, age, model_pattern, intercept, slope, std, *args, **kwargs)

class RVisualFieldsp24d2Model(AgeLinearModel):
    """
    """
    def __init__(self, eval_pattern=PATTERN_P24D2, age=None, *args, **kwargs):
        intercept = np.array((29.92888, 30.29715, 30.38557, 30.19415, 31.51319, 32.16129, 32.52956, 32.61798, 32.42656, 31.95530, 32.22227, 33.15021, 33.79832, 34.16658, 34.25500, 34.06359, 33.59233, 32.84123, 32.05612, 33.26391, 34.19186, 34.83996, 35.20822, 35.29665, 35.10523,      np.nan, 33.88287, 32.50238, 33.71017, 34.63812, 35.28622, 35.65448, 35.74291, 35.55149,      np.nan, 34.32913, 33.56105, 34.48900, 35.13710, 35.50536, 35.59379, 35.40237, 34.93111, 34.18001, 33.74449, 34.39260, 34.76086, 34.84928, 34.65786, 34.18661, 33.05271, 33.42097, 33.50940, 33.31798, ))
        slope = np.array((-0.06315807, -0.06012610, -0.06084715, -0.06532122, -0.06343690, -0.05665189, -0.05361992, -0.05434097, -0.05881504, -0.06704215, -0.06995794, -0.05941990, -0.05263489, -0.04960292, -0.05032397, -0.05479804, -0.06302515, -0.07500529, -0.08272118, -0.06843011, -0.05789208, -0.05110707, -0.04807509, -0.04879614, -0.05327022,         np.nan, -0.07347747, -0.08368254, -0.06939147, -0.05885344, -0.05206843, -0.04903645, -0.04975750, -0.05423158,         np.nan, -0.07443882, -0.07284201, -0.06230397, -0.05551897, -0.05248699, -0.05320804, -0.05768212, -0.06590922, -0.07788936, -0.06824369, -0.06145868, -0.05842670, -0.05914775, -0.06362183, -0.07184894, -0.06988758, -0.06685560, -0.06757665, -0.07205073, ))
        sds = np.array((
            (3.118858, 3.064624, 2.401378, 2.5050791),(3.022890, 2.973179, 2.322537, 2.4201881),(3.056706, 3.006811, 2.360698, 2.4558714),(3.220304, 3.165519, 2.515862, 2.6121290),(2.612925, 2.543299, 1.990947, 2.0488437),(2.387174, 2.326777, 1.795105, 1.8433784),(2.291207, 2.235332, 1.716264, 1.7584874),(2.325022, 2.268964, 1.754425, 1.7941707),(2.488621, 2.427672, 1.909588, 1.9504283),(2.782003, 2.711458, 2.181753, 2.2272602),(2.457477, 2.368681, 1.886565, 1.9158417),(2.101943, 2.027082, 1.573721, 1.5898021),(1.876192, 1.810561, 1.377878, 1.3843368),(1.780225, 1.719116, 1.299037, 1.2994458),(1.814040, 1.752748, 1.337198, 1.3351291),(1.977639, 1.911456, 1.492361, 1.4913867),(2.271021, 2.195242, 1.764527, 1.7682186),(2.694187, 2.604104, 2.153693, 2.1656248),(2.652514, 2.540771, 2.088232, 2.1060731),(2.167196, 2.074096, 1.658385, 1.6594592),(1.811662, 1.732497, 1.345541, 1.3334195),(1.585911, 1.515975, 1.149698, 1.1279542),(1.489943, 1.424530, 1.070857, 1.0430632),(1.523759, 1.458162, 1.109018, 1.0787465),(1.687358, 1.616871, 1.264182, 1.2350041),(  np.nan,   np.nan,   np.nan,    np.nan),(2.403905, 2.309519, 1.925514, 1.9092423),(2.582933, 2.467817, 2.049099, 2.0523496),(2.097616, 2.001141, 1.619252, 1.6057357),(1.742082, 1.659543, 1.306408, 1.2796961),(1.516331, 1.443021, 1.110565, 1.0742308),(1.420363, 1.351576, 1.031724, 0.9893397),(1.454179, 1.385208, 1.069885, 1.0250230),(1.617777, 1.543916, 1.225048, 1.1812807),(  np.nan,   np.nan,   np.nan,    np.nan),(2.334325, 2.236564, 1.886380, 1.8555188),(2.248736, 2.149817, 1.769166, 1.7546713),(1.893202, 1.808219, 1.456321, 1.4286317),(1.667451, 1.591697, 1.260479, 1.2231663),(1.571484, 1.500252, 1.181638, 1.1382753),(1.605299, 1.533884, 1.219799, 1.1739586),(1.768898, 1.692593, 1.374962, 1.3302163),(2.062280, 1.976378, 1.647127, 1.6070482),(2.485446, 2.385240, 2.036294, 2.0044544),(2.265024, 2.178526, 1.795282, 1.7802263),(2.039273, 1.962004, 1.599439, 1.5747610),(1.943305, 1.870559, 1.520598, 1.4898700),(1.977121, 1.904191, 1.558759, 1.5255533),(2.140720, 2.062900, 1.713923, 1.6818109),(2.434102, 2.346685, 1.986088, 1.9586428),(2.631795, 2.553942, 2.127446, 2.1290147),(2.535828, 2.462497, 2.048605, 2.0441237),(2.569643, 2.496129, 2.086767, 2.0798070),(2.733242, 2.654837, 2.241930, 2.2360646),
        )) #        sens       td       pd     pdghr
        super().__init__(eval_pattern, age, PATTERN_P24D2, intercept, slope, sds[:, 0], *args, **kwargs)

class HFASitaStandardp24d2Model(AgeLinearModel):
    """
    Values from SG and NP study's HFA SFA reports
    """
    # Row 0: Intercept 0 years old
    # Row 1: Slope db per year
    # Row 2: Normalized weights for MD. Note the +15, -9 OD point is also nan
    # Row 3: Example of normal values for a 50 year old
    # Row 4: Rough estimates of the equivalent standard deviation values at each location based on Row 2
    _DATA = np.array([[31.349933293, 31.312414416, 31.169739532, 30.882612385,
        32.588484259, 33.076235356, 33.133126378, 33.243637973,
        32.593688003, 32.25872542 , 32.309453205, 33.403797913,
        34.361128539, 34.84212894 , 34.622988482, 34.134933308,
        33.28108327 , 32.776603045, 31.094173661, 32.699711387,
        34.157334348, 35.302164332, 35.768661454, 35.596776643,
        34.872284668,          nan, 33.135021322, 31.203513499,
        32.744357571, 34.298435927, 35.319608295, 35.892821992,
        35.734421575, 35.106355909,          nan, 33.107561327,
        32.274466065, 33.745948497, 34.747230416, 35.211243942,
        35.009893642, 34.41222315 , 33.63516923 , 32.906396341,
        32.488400928, 33.413697678, 33.819462803, 33.748204153,
        33.36543537 , 33.127312154, 31.685092576, 32.1794866  ,
        32.317946534, 32.235124663],
       [-0.075817247, -0.072909852, -0.075175658, -0.079044156,
        -0.070353108, -0.065272186, -0.062907624, -0.069014646,
        -0.067426022, -0.075163357, -0.066401198, -0.057082506,
        -0.057386475, -0.061507134, -0.062171979, -0.065100995,
        -0.06471435 , -0.072229495, -0.075731615, -0.058483369,
        -0.052979535, -0.056006571, -0.059697527, -0.061627668,
        -0.06043234 ,          nan, -0.06317035 , -0.075984626,
        -0.056088778, -0.051385679, -0.052167634, -0.056535479,
        -0.057938387, -0.059073015,          nan, -0.056320839,
        -0.057522544, -0.051461355, -0.050996777, -0.053066416,
        -0.051570916, -0.051679014, -0.051249615, -0.053477272,
        -0.050593126, -0.047800924, -0.046561642, -0.046997395,
        -0.04659614 , -0.053001291, -0.047593823, -0.046412432,
        -0.046091855, -0.04601346 ],
       [ 0.010419819,  0.010166466,  0.010190305,  0.007243378,
         0.015583108,  0.016835846,  0.017428225,  0.017660117,
         0.012799102,  0.010663685,  0.013504209,  0.02189439 ,
         0.024259266,  0.023817303,  0.021313381,  0.01996011 ,
         0.017153099,  0.012795519,  0.00730894 ,  0.019037978,
         0.025717604,  0.029349571,  0.027620525,  0.024532012,
         0.022220079,          nan,  0.015931601,  0.008023233,
         0.018464697,  0.027220862,  0.029081296,  0.027552527,
         0.029042445,  0.022851615,          nan,  0.017088342,
         0.014950162,  0.024765219,  0.028339385,  0.03140365 ,
         0.025624526,  0.022922407,          nan,  0.016324785,
         0.018829505,  0.023440004,  0.024611209,  0.023968592,
         0.02232414 ,  0.018399735,  0.015762739,  0.017948766,
         0.018671526,  0.016775077],
       [27.559070939, 27.666921822, 27.410956609, 26.930404573,
        29.070828871, 29.812626047, 29.987745161, 29.792905691,
        29.222386888, 28.500557567, 28.989393292, 30.549672635,
        31.491804812, 31.766772241, 31.514389514, 30.879883561,
        30.045365757, 29.165128302, 27.307592914, 29.77554293 ,
        31.508357612, 32.501835791, 32.783785118, 32.515393222,
        31.85066766 ,          nan, 29.976503818, 27.404282186,
        29.939918694, 31.729151983, 32.711226599, 33.066048043,
        32.837502218, 32.152705159,          nan, 30.29151937 ,
        29.39833887 , 31.172880746, 32.197391557, 32.55792313 ,
        32.431347858, 31.82827243 , 31.072688485, 30.232532731,
        29.958744634, 31.0236515  , 31.491380698, 31.398334392,
        31.035628354, 30.477247616, 29.305401404, 29.858864982,
        30.0133538  , 29.934451669],
       [ 3.133825972,  3.172633799,  3.168920622,  3.758671619,
         2.562582912,  2.465400562,  2.423139267,  2.4071778  ,
         2.827582033,  3.097785179,  2.752772942,  2.161914609,
         2.05383774 ,  2.072806053,  2.191183708,  2.264245376,
         2.442494856,  2.827977838,  3.741775727,  2.318432982,
         1.994755668,  1.867256944,  1.924814994,  2.042388602,
         2.146012057,          nan,  2.534400556,  3.57133235 ,
         2.354148567,  1.93889381 ,  1.875849872,  1.927188699,
         1.877104147,  2.116150319,          nan,  2.447118477,
         2.616266648,  2.032749556,  1.900245669,  1.805156648,
         1.998375225,  2.1128801  ,          nan,  2.503693869,
         2.331232075,  2.089421803,  2.039099814,  2.06625397 ,
         2.141004559,  2.358300703,  2.547939499,  2.387743448,
         2.341073522,  2.46986209 ]])

    def __init__(self, eval_pattern, age, *args, **kwargs):
        super().__init__(eval_pattern, age, PATTERN_P24D2,
                         HFASitaStandardp24d2Model._DATA[0],
                         HFASitaStandardp24d2Model._DATA[1],
                         HFASitaStandardp24d2Model._DATA[4], *args, **kwargs)

class TPP2021p24d2Model(AgeLinearModel):
    """
    Values from SG-NP-ON study, in reference to HFASitaStandardp24d2Model (HFA SFA)
    """
    # _PARABOLA_OFFSET = np.array([
    #     [-1.7492476, -1.84760367, -1.84760367, -1.7492476, -1.84760367,
    #      -2.04431581, -2.14267188, -2.14267188, -2.04431581, -1.84760367,
    #      -1.7492476, -2.04431581, -2.24102795, -2.33938402, -2.33938402,
    #      -2.24102795, -2.04431581, -1.7492476, -1.45417938, -1.84760367,
    #      -2.14267188, -2.33938402, -2.4377401, -2.4377401, -2.33938402,
    #      0.0, -1.84760367, -1.45417938, -1.84760367, -2.14267188,
    #      -2.33938402, -2.4377401, -2.4377401, -2.33938402, 0.0,
    #      -1.84760367, -1.7492476, -2.04431581, -2.24102795, -2.33938402,
    #      -2.33938402, -2.24102795, -2.04431581, -1.7492476, -1.84760367,
    #      -2.04431581, -2.14267188, -2.14267188, -2.04431581, -1.84760367,
    #      -1.7492476, -1.84760367, -1.84760367, -1.7492476]
    #         ])
    _PARABOLA_OFFSET = np.array([
        [-1.20876073, -1.328788  , -1.328788  , -1.20876073, -1.328788  ,
       -1.56884255, -1.68886983, -1.68886983, -1.56884255, -1.328788  ,
       -1.20876073, -1.56884255, -1.80889711, -1.92892438, -1.92892438,
       -1.80889711, -1.56884255, -1.20876073, -0.8486789 , -1.328788  ,
       -1.68886983, -1.92892438, -2.04895166, -2.04895166, -1.92892438,
               0.0, -1.328788  , -0.8486789 , -1.328788  , -1.68886983,
       -1.92892438, -2.04895166, -2.04895166, -1.92892438,         0.0,
       -1.328788  , -1.20876073, -1.56884255, -1.80889711, -1.92892438,
       -1.92892438, -1.80889711, -1.56884255, -1.20876073, -1.328788  ,
       -1.56884255, -1.68886983, -1.68886983, -1.56884255, -1.328788  ,
       -1.20876073, -1.328788  , -1.328788  , -1.20876073]
    ])

    def __init__(self, eval_pattern, age, *args, **kwargs):
        super().__init__(eval_pattern, age, PATTERN_P24D2,
                         HFASitaStandardp24d2Model._DATA[0] + TPP2021p24d2Model._PARABOLA_OFFSET[0],
                         HFASitaStandardp24d2Model._DATA[1],
                         HFASitaStandardp24d2Model._DATA[4], *args, **kwargs)

class ConstantModel(Model):
    """
    Returns a pre-specified constant for testing purposes
    """
    def __init__(self, eval_pattern, mean, std, *args, **kwargs):
        super().__init__(eval_pattern, *args, **kwargs)
        self.mean = mean * 1.0
        self.param["std"] = std * 1.0

    def get_mean(self):
        return np.full(shape=len(self.param['eval_pattern']), fill_value=self.mean)

    def get_std(self):
        return np.full(shape=len(self.param['eval_pattern']), fill_value=self.param["std"])
