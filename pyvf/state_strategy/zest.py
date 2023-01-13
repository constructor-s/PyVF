from __future__ import annotations
from abc import ABC, abstractmethod
from ..ext import cached_property
from typing import Callable, Union

import attr
import numpy as np
from attrs import evolve, Factory, field

from . import Trial, State, StateNode, PointState
from ..stats.pos import pos_ramp


def shift(arr, num, fill_value):
    """
    https://stackoverflow.com/a/42642326/6610243

    Parameters
    ----------
    arr : np.ndarray
    num : int
    fill_value : float

    Returns
    -------
    Copy of shifted array by num (left if negative, right if positive),
    with the head/trail values accumulated to the first/last bin
    """
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
        result[-1] += np.sum(arr[-num:])
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
        result[0] += np.sum(arr[:-num])
    else:
        result[:] = arr
    return result


@attr.s(auto_attribs=True, slots=False, kw_only=True, frozen=True)
class BayesianMixedPrior:
    x: np.ndarray
    q0_normal: np.ndarray = field(converter=lambda v: np.array(v, dtype=np.float64) / v.sum())
    q0_abnormal: np.ndarray = field(converter=lambda v: np.array(v, dtype=np.float64) / v.sum())
    weight_normal: float
    weight_abnormal: float
    eps: float = 1e-3
    offset: float = 0.0

    @property
    def q0(self):
        return (
            1.0 / (self.weight_normal + self.weight_abnormal + self.eps * len(self.x)) *
            (self.weight_normal * self._q0_normal_shifted + self.weight_abnormal * self.q0_abnormal + self.eps)
        )

    def with_offset(self, db) -> BayesianMixedPrior:
        return evolve(self, offset=self.offset+db)

    @property
    def _q0_normal_shifted(self):
        dx = self.x[1] - self.x[0]
        assert np.all(np.diff(self.x) == dx), "Only equal spacing is supported"
        return shift(
            self.q0_normal,
            round(self.offset * 1.0 / dx),
            self.eps
        )

    @property
    def pretest(self):
        return self.x[np.argmax(self._q0_normal_shifted)]

    @staticmethod
    def unicode_format(x):
        x /= x.max()
        x *= 8
        x = np.around(x).astype(int)
        mapping = {
            0: "_",
            1: "\u2581",
            2: "\u2582",
            3: "\u2583",
            4: "\u2584",
            5: "\u2585",
            6: "\u2586",
            7: "\u2587",
            8: "\u2588"
        }
        return "".join([mapping[i] for i in x])

@attr.s(auto_attribs=True, slots=False, kw_only=True, frozen=True, init=False)
class BayesianPointState(PointState, ABC):
    """
    This class is hash by id (eq=False) because np.ndarray is not hashable and needs to use cached_property
    https://www.attrs.org/en/stable/hashing.html
    """
    # x: np.ndarray # deprecated, maintained for compatibility
    # q0: np.ndarray # deprecated, maintained for compatibility
    prior: BayesianMixedPrior
    pos_fun: Callable[[np.ndarray], float] = lambda x: pos_ramp(x, center=0, yl=0.95, yr=0.05, width=4.0)
    q_update: np.ndarray = Factory(lambda self: np.ones_like(self.prior.x, dtype=np.float64), takes_self=True)
    terminate_std: Union[float, Callable[[int], float]] = 1.5
    trials: int = 0
    # low_indices_db: float # deprecated, maintained for compatibility

    def __init__(self, **kwargs):
        if "prior" not in kwargs:
            import warnings
            warnings.warn(f"Deprecated initialization for {self.__class__.__qualname__}, use 'prior=' directly instead of 'x' and 'q0'")
            x = kwargs.pop("x")
            q0 = kwargs.pop("q0")
            low_indices_db = kwargs.pop("low_indices_db", 1.0)
            q0_abnormal = np.where(x < low_indices_db, q0, 0)
            q0_normal = np.where(x < low_indices_db, 0, q0)
            kwargs["prior"] = BayesianMixedPrior(
                x=x, q0_normal=q0_normal, q0_abnormal=q0_abnormal,
                weight_normal=q0_normal.sum(), weight_abnormal=q0_abnormal.sum(), eps=0
            )
        self.__attrs_init__(**kwargs)

    @cached_property
    def q(self) -> np.ndarray:
        q = self.prior.q0 * self.q_update
        return 1.0 / q.sum() * q

    @cached_property
    def mean(self) -> float:
        return self.prior.x @ self.q

    @cached_property
    def median(self) -> float:
        return np.interp(0.5, self.q.cumsum(), self.prior.x)

    @cached_property
    def mode(self):
        return self.prior.x[self.q.argmax()]

    @cached_property
    def var(self) -> float:
        mean2 = self.mean ** 2
        ex2 = (self.prior.x ** 2) @ self.q
        return ex2 - mean2

    @cached_property
    def terminated(self) -> bool:
        if callable(self.terminate_std):
            term_std = self.terminate_std(self.trials)
        else:
            term_std = self.terminate_std
        return self.trials > 0 and self.var < term_std ** 2

    def with_trial(self, trial: Trial) -> BayesianPointState:
        if trial.seen:
            update = self.pos_fun(trial.threshold - self.prior.x)
        else:
            update = 1 - self.pos_fun(trial.threshold - self.prior.x)
        return evolve(self, q_update=self.q_update * update, trials=self.trials + 1)

    def with_offset(self, db) -> BayesianPointState:
        return evolve(self, prior=self.prior.with_offset(db))

    @cached_property
    def pretest(self):
        return self.prior.pretest


class ZestPointState(BayesianPointState):
    @cached_property
    def estimate(self) -> float:
        if np.all(self.q_update == self.q_update[0]):  # No update has been done yet
            return self.pretest
        else:
            return self.mean

    @cached_property
    def next_trial(self) -> Trial:
        return Trial(point=self.point, threshold=self.mean)
