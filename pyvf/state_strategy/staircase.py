from __future__ import annotations
from ..ext import cached_property

from . import *
from attrs import Factory, evolve
import numpy as np


@attr.s(auto_attribs=True, slots=False, kw_only=True, frozen=True)
class StaircasePointState(PointState):
    pretest: float
    starting: float = Factory(lambda self: self.pretest, takes_self=True)
    last_response: Optional[bool] = None
    last_threshold: Optional[float] = None
    reversals: int = 0
    steps: tuple[float] = (4, 2)
    # Ceiling and floor of the staircase procedure
    ceiling: float = 33
    floor: float = 0
    last_seen_threshold: float = Factory(lambda self: self.floor - 1, takes_self=True)

    @property
    def estimate(self) -> float:
        if self.last_threshold is None:
            assert self.last_response is None
            return self.starting
        elif not self.terminated:
            return self.last_threshold
        else:
            return self.last_seen_threshold

    @cached_property
    def next_trial(self) -> Optional[Trial]:
        if self.terminated:
            return None
        elif self.last_response is None:
            assert self.last_threshold is None
            # noinspection PyTypeChecker
            return Trial(point=self.point, threshold=np.clip(self.starting, self.floor, self.ceiling))
        elif self.last_response == True:
            # noinspection PyTypeChecker
            return Trial(point=self.point, threshold=np.clip(self.last_threshold + self.steps[self.reversals], self.floor, self.ceiling))
        else:
            assert self.last_response == False
            # noinspection PyTypeChecker
            return Trial(point=self.point, threshold=np.clip(self.last_threshold - self.steps[self.reversals], self.floor, self.ceiling))

    def with_trial(self, trial: Trial) -> StaircasePointState:
        if self.last_response is None:
            reversals = 0
        else:
            if self.last_response == False and trial.seen == True and trial.threshold <= self.last_threshold: # brighter seen
                reversals = self.reversals + 1
            elif self.last_response == True and trial.seen == False and trial.threshold >= self.last_threshold: # dimmer not seen
                reversals = self.reversals + 1
            else:
                reversals = self.reversals
        return evolve(
            self,
            last_response=trial.seen,
            last_threshold=trial.threshold,
            last_seen_threshold=trial.threshold if trial.seen else self.last_seen_threshold,
            reversals=reversals,
        )

    def with_offset(self, db) -> StaircasePointState:
        return evolve(
            self,
            starting=self.pretest + db
        )

    @cached_property
    def terminated(self) -> bool:
        if self.last_threshold is None:
            assert self.last_response is None
            return False
        return (self.reversals >= len(self.steps) or
                (self.last_threshold <= self.floor and self.last_response == False) or
                (self.last_threshold >= self.ceiling and self.last_response == True))
