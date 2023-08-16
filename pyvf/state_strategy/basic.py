from __future__ import annotations
from . import *
from ..ext import cached_property
import attr
import numpy as np
from attrs import evolve, Factory, field

@attr.s(auto_attribs=True, slots=False, kw_only=True, frozen=True)
class RandomFieldState(State):
    """
    Randomly select the next location to be tested with no seeding
    """
    nodes: Tuple[StateNode] | List[StateNode] # PointStates
    rng: np.random.Generator = Factory(lambda: np.random.default_rng(0))

    @property
    def next_trial(self) -> Optional[Trial]:
        candidates = [n for n in self.nodes if not n.instance.terminated]
        if not candidates:
            return None
        else:
            return self.rng.choice(candidates).instance.next_trial

    def with_trial(self, trial: Trial) -> State:
        # Create a copy of the existing point nodes
        nodes = list(self.nodes)
        # Add this trial to the linked list of point states
        point_index = trial.point.index
        nodes[point_index] = nodes[point_index].add_trial(trial)

        return evolve(
            self,
            nodes=tuple(nodes),
            # rng=self.rng
        )

    @cached_property
    def estimate(self):
        return tuple(n.instance.estimate for n in self.nodes)

    @cached_property
    def terminated(self) -> bool:
        return all(n.instance.terminated for n in self.nodes)
