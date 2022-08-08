from __future__ import annotations

from typing import Tuple

from ..ext import cached_property

from . import *
from attrs import Factory, evolve
import numpy as np


@attr.s(auto_attribs=True, slots=False, kw_only=True, frozen=True)
class StaircasePointState(PointState):
    pretest: float # This is not affected by seeding offset
    starting: float = Factory(lambda self: self.pretest, takes_self=True) # This is by default the same as pretest, and is affected by the seeding offset
    last_response: Optional[bool] = None
    last_threshold: Optional[float] = None
    reversals: int = 0
    steps: Tuple[float] = (4, 2)
    # Ceiling and floor of the staircase procedure
    ceiling: float = 33
    floor: float = 0
    last_seen_threshold: float = Factory(lambda self: self.floor - 1, takes_self=True)
    last_reversal_threshold: float = Factory(lambda self: self.floor - 1, takes_self=True) # mean of the reversal trials, last seen if all seen, -1 if all not seen
    retest: bool = False  # whether to initiate another set of staircase after the current set...
    retest_range: Tuple[float] = Factory(lambda self: (self.starting - 4, self.starting + 4), takes_self=True)  #... if the determined threshold is outside thr range here
    prev_last_seen_threshold: float = None  # store the previous set of staircase's result here
    prev_last_reversal_threshold: float = None

    @property
    def estimate(self) -> float:
        if self.last_threshold is None:
            assert self.last_response is None
            return self.starting
        elif not self.terminated:
            return self.last_threshold
        else:
            if self.prev_last_seen_threshold is None:
                return self.last_seen_threshold
            else:
                # According to Turpin 2003, return the mean of the first and second staircase
                return 0.5 * (self.last_seen_threshold + self.prev_last_seen_threshold)

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
        last_reversal_threshold = self.last_reversal_threshold

        if self.last_response is None:
            reversals = 0
        else:
            if self.last_response == False and trial.seen == True and trial.threshold <= self.last_threshold: # brighter seen
                reversals = self.reversals + 1
                last_reversal_threshold = 0.5 * (trial.threshold + self.last_threshold)
            elif self.last_response == True and trial.seen == False and trial.threshold >= self.last_threshold: # dimmer not seen
                reversals = self.reversals + 1
                last_reversal_threshold = 0.5 * (trial.threshold + self.last_threshold)
            else:
                reversals = self.reversals

        if reversals == 0 and trial.seen == True:
            last_reversal_threshold = trial.threshold

        state = evolve(
            self,
            last_response=trial.seen,
            last_threshold=trial.threshold,
            last_seen_threshold=trial.threshold if trial.seen else self.last_seen_threshold,
            last_reversal_threshold=last_reversal_threshold,
            reversals=reversals,
        )

        if state.retest and state.terminated and (
                state.estimate < state.retest_range[0] or state.estimate > state.retest_range[1]):
            # Retest another staircase
            return evolve(
                state,
                # same pretest
                starting=state.estimate, # start at the current staircase's final estimate
                last_response=None,
                last_threshold=None,
                reversals=0,
                # same steps
                # same ceiling
                # same floor
                last_seen_threshold=state.floor-1,
                last_reversal_threshold=state.floor - 1,
                retest=False,  # Do not do more retesting
                retest_range=(state.starting - 4, state.starting + 4),
                prev_last_seen_threshold=state.last_seen_threshold,
                prev_last_reversal_threshold=state.last_reversal_threshold
            )
        else:
            return state

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
