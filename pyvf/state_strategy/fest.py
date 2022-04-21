from functools import cached_property

import numpy as np
import scipy.stats
from attrs import evolve
from . import *
from ..stats.pos import pos_ramp
from .staircase import *


@attr.s(auto_attribs=False, slots=False, kw_only=True, frozen=True)
class FestFieldState(State):
    database = attr.ib(type=np.ndarray)
    database_p = attr.ib(
        type=np.ndarray,
        converter=lambda p: p * 1.0 / np.sum(p)
    )
    nodes = attr.ib(type=tuple[StateNode])
    seeding_terminated = attr.ib(
        type=bool,
        default=False
    )
    seeding_terminate_entropy = attr.ib(type=float)
    rng: np.random.Generator = attr.ib(factory=lambda: np.random.default_rng(0))

    @cached_property
    def next_trial(self) -> Optional[Trial]:
        if not self.seeding_terminated:
            index = self.var(p=self.database_p, x=self.database).argmax()
            # print(self.database_p.max(), self.database_p.mean(), scipy.stats.entropy(self.database_p))
            threshold = self.database_p @ self.database[:, index]
            return Trial(point=self.nodes[index].instance.point, threshold=threshold)
        else:
            candidates = list(range(len(self.nodes)))
            self.rng.shuffle(candidates)  # Much faster than random.shuffle
            for i in candidates:
                n = self.nodes[i]
                if not n.instance.terminated:
                    point_state = n.instance
                    return point_state.next_trial
            return None

    def with_trial(self, trial: Trial) -> State:
        assert trial.seen is not None, "trial.seen is None"
        # if trial.seen:
        p_update = pos_ramp(
            x=self.database[:, trial.point.index],
            center=trial.threshold,
            yl=0.05,  # fp
            yr=1 - 0.05,  # 1 - fn
            width=4
        )
        if not trial.seen:
            p_update = 1 - p_update
        assert len(p_update) == len(self.database_p)

        # Create a copy of the existing point nodes
        nodes = list(self.nodes)
        # Add this trial to the linked list of point states
        point_index = trial.point.index
        nodes[point_index] = nodes[point_index].add_trial(trial)

        if self.seeding_terminated:
            seeding_terminated = True
            nodes = nodes
        else:
            if scipy.stats.entropy(self.database_p) < self.seeding_terminate_entropy:
                seeding_terminated = True
                # Seed all the points
                # nodes = [
                #     n.add_state(
                #         state=n.instance.with_offset(e - n.instance.pretest), # TODO: Reset staircase here too?
                #         trial=None
                #     )
                #     for n, e in zip(nodes, self.estimate)
                # ]
                seeded_nodes = []
                for n, e in zip(nodes, self.estimate):
                    seeded_node = n.add_state(
                        state=n.instance.with_offset(e - n.instance.pretest),
                        trial=None
                    )
                    if isinstance(seeded_node.instance, StaircasePointState):
                        seeded_node.instance = evolve(
                            seeded_node.instance, 
                            last_response=None,
                            last_threshold=None,
                            reversals=0
                        )
                    seeded_nodes.append(seeded_node)
                nodes = seeded_nodes
            else:
                seeding_terminated = False
                nodes = nodes

        return evolve(
            self,
            database_p=self.database_p * p_update,
            seeding_terminated=seeding_terminated,
            nodes=tuple(nodes)
        )

    @property
    def estimate(self) -> np.ndarray:
        if not self.seeding_terminated:
            return self.database_p @ self.database
        else:
            return tuple(n.instance.estimate for n in self.nodes)

    @staticmethod
    def var(p, x) -> np.ndarray:
        ex_2 = (p @ x) ** 2
        e_x2 = p @ (x ** 2)
        return e_x2 - ex_2

    @property
    def terminated(self) -> bool:
        return self.seeding_terminated and all(n.instance.terminated for n in self.nodes)
