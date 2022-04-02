from abc import ABC
from functools import cached_property
from typing import Callable

import attr
import numpy as np
from attrs import evolve, Factory

from . import Trial, State, StateNode, PointState
from ..stats.pos import pos_ramp


@attr.s(auto_attribs=True, slots=False, kw_only=True, frozen=True)
class BayesianPointState(PointState, ABC):
    x: np.ndarray
    q0: np.ndarray
    pos_fun: Callable[[np.ndarray], float] = lambda x: pos_ramp(x, center=0, yl=0.95, yr=0.05, width=4.0)
    pretest: float
    starting: float = Factory(lambda self: self.pretest, takes_self=True)
    q_update: np.ndarray = Factory(lambda self: np.ones_like(self.x, dtype=np.float64), takes_self=True)
    terminate_std: float = 1.5

    @cached_property
    def q(self):
        q = self.q0 * self.q_update
        return 1.0 / q.sum() * q

    @cached_property
    def mean(self) -> float:
        return self.x @ self.q

    @cached_property
    def var(self) -> float:
        mean2 = self.mean ** 2
        ex2 = (self.x**2) @ self.q
        return ex2 - mean2

    @cached_property
    def terminated(self):
        return self.var < self.terminate_std ** 2

    def with_trial(self, trial: Trial) -> State:
        if trial.seen:
            update = self.pos_fun(trial.threshold - self.x)
        else:
            update = 1 - self.pos_fun(trial.threshold - self.x)
        return evolve(self, q_update=self.q_update * update)


class ZestPointState(BayesianPointState):
    @cached_property
    def estimate(self) -> float:
        return self.mean

    @cached_property
    def next_trial(self) -> Trial:
        return Trial(point=self.point, threshold=self.mean)
