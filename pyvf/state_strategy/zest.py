from __future__ import annotations
from abc import ABC
from ..ext import cached_property
from typing import Callable, Union

import attr
import numpy as np
from attrs import evolve, Factory

from . import Trial, State, StateNode, PointState
from ..stats.pos import pos_ramp

EPS = 1e-3


def shift(q, offset, low_indices=1, eps=EPS):
    if offset == 0:
        return np.array(q)

    result = np.zeros_like(q)
    result[:low_indices] = q[:low_indices]
    if offset < 0:  # Move left
        result[low_indices:offset] = q[low_indices - offset:]
        result[low_indices] += np.sum(q[low_indices:low_indices - offset])
        return result.clip(eps, None)
    else:  # Move right
        result[low_indices + offset:] = q[low_indices:-offset]
        result[-1] += np.sum(q[-offset:])
        return result.clip(eps, None)


@attr.s(auto_attribs=True, slots=False, kw_only=True, frozen=True)
class BayesianPointState(PointState, ABC):
    """
    This class is hash by id (eq=False) because np.ndarray is not hashable and needs to use cached_property
    https://www.attrs.org/en/stable/hashing.html
    """
    x: np.ndarray
    q0: np.ndarray
    pos_fun: Callable[[np.ndarray], float] = lambda x: pos_ramp(x, center=0, yl=0.95, yr=0.05, width=4.0)
    q_update: np.ndarray = Factory(lambda self: np.ones_like(self.x, dtype=np.float64), takes_self=True)
    terminate_std: Union[float, Callable[[int], float]] = 1.5
    trials: int = 0

    @cached_property
    def q(self) -> np.ndarray:
        q = self.q0 * self.q_update
        return 1.0 / q.sum() * q

    @cached_property
    def mean(self) -> float:
        return self.x @ self.q

    @cached_property
    def median(self) -> float:
        return np.interp(0.5, self.q.cumsum(), self.x)

    @cached_property
    def mode(self):
        return self.x[self.q.argmax()]

    @cached_property
    def var(self) -> float:
        mean2 = self.mean ** 2
        ex2 = (self.x ** 2) @ self.q
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
            update = self.pos_fun(trial.threshold - self.x)
        else:
            update = 1 - self.pos_fun(trial.threshold - self.x)
        return evolve(self, q_update=self.q_update * update, trials=self.trials + 1)

    def with_offset(self, db) -> BayesianPointState:
        dx = self.x[1] - self.x[0]
        assert np.all(np.diff(self.x) == dx), "Only equal spacing is supported"
        q0_new = shift(
            q=self.q0,
            offset=round(db / dx),
            low_indices=(self.x < 1).sum()
        )
        return evolve(self, q0=q0_new)

    @cached_property
    def pretest(self):
        return self.x[self.x > 1][np.argmax(self.q[self.x > 1])]


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
