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
import warnings
from pyvf.strategy import XOD, YOD, PATTERN_P24D2


def wtd_var(x, weights=None, normwt=False):
    '''
    https://github.com/harrelfe/Hmisc/blob/8bbb192103e0091398cb58b257ed590e481cfa35/R/wtd.stats.s#L15
    TODO: Previous note: This function does not produce the exact same result as the R package, but accurate within 0.1 dB so good for our purpose for now
    This implementation is consistent with the definition of weighted variance with frequency weights
    # https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Frequency_weights
    but slightly different from described in Heijl's A package for the statistical analysis of visual fields
    The weights passed in should be the inverse of variances, not standard deviations.
    The return is variance (i.e. square of PSD), not standard deviation
    '''
    if weights is None:
        return np.var(x)

    if normwt:
        weights = weights * len(x) * 1.0 / np.sum(weights)  # Normalizes sum(weights) to len(x) = n

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
    def __init__(self, eval_pattern, age=None, gh_percentile=0.85, *args, **kwargs):
        """

        Parameters
        ----------
        eval_pattern : see Strategy
        age : age of the patient in years
        args
        kwargs
        """
        self.args = args
        self.param = kwargs
        self.param['eval_pattern'] = eval_pattern
        self.param['age'] = age
        self.param['gh_percentile'] = gh_percentile

    def get_mean(self):
        raise NotImplementedError()

    def get_std(self):
        raise NotImplementedError()

    def get_td(self, vf):
        td = vf - self.get_mean()
        std = self.get_std()
        # If std is non-positive or NaN then it is excluded -
        # usually this is used to indicate two locations near blindspots
        non_blindspot = std > 0
        td[~non_blindspot] = np.nan
        return td

    def get_pd(self, vf):
        td = self.get_td(vf)
        std = self.get_std()
        non_blindspot = std > 0
        td_nbs = td[non_blindspot]
        # TODO: Make sure general height is calculated at 85% percentile without blindspots, though the difference is likely insignificant anyways
        general_height = np.percentile(td_nbs, self.param['gh_percentile'] * 100)
        pd = td - general_height
        return pd

    def get_md(self, vf):
        std = self.get_std()
        non_blindspot = std > 0

        std = std[non_blindspot]
        weights = 1.0 / (std ** 2)
        weights = weights * 1.0 / np.sum(weights)

        return np.dot(weights, self.get_td(vf)[non_blindspot])

    def get_psd(self, vf=None, td_nbs=None, md=None, method="turpin"):
        std = self.get_std()
        non_blindspot = std > 0

        std = std[non_blindspot]
        weights = 1.0 / (std ** 2)

        if td_nbs is None:
            td_nbs = self.get_td(vf)[non_blindspot]

        if method.lower() == "turpin":
            # Weighted standard deviation of TD (i.e. Turpin)
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
