from typing import Tuple, Dict

from . import *
from .zest import *
from ..strategy import PATTERN_P24D2, XOD, YOD, LOC


def get_quadrant_map(pattern_name):
    return {
        "P24D2": {
            12: (0, 1, 4, 5, 6, 10, 11, 13, 18, 19, 20, 21, 22),
            15: (2, 3, 7, 8, 9, 14, 16, 17, 23, 24, 26),
            38: (27, 28, 29, 30, 31, 36, 37, 39, 44, 45, 46, 50, 51),
            41: (32, 33, 35, 40, 42, 43, 47, 48, 49, 52, 53),
            25: (),
            34: ()
        }
    }[pattern_name]


def get_flat_map(pattern_name):
    return {
        "P24D2": {i: tuple() for i in range(54)}
    }[pattern_name]


@attr.s(auto_attribs=True, slots=False, kw_only=True, frozen=True)
class QuadrantFieldState(State):
    nodes: Tuple[StateNode]  # PointStates
    quadrant_map: Dict[int, List[int]]
    sequential_indices: Tuple[int] = Factory(lambda self: tuple(k for k, v in self.quadrant_map.items() if v), takes_self=True)
    eager_indices: Tuple[int] = Factory(lambda self: tuple(k for k, v in self.quadrant_map.items() if not v), takes_self=True)
    rng: np.random.Generator = Factory(lambda: np.random.default_rng(0))

    @property
    def next_trial(self) -> Optional[Trial]:
        # candidates = set(self.sequential_indices)
        # candidates.update(self.eager_indices)  # Concatenate
        #
        # nodes_not_terminated = [n for i, n in enumerate(self.nodes)
        #                         if i in candidates and not n.instance.terminated]
        # if nodes_not_terminated:
        #     # n = self.rng.choice(nodes_not_terminated)  # 590.1 per hit
        #     n = nodes_not_terminated[self.rng.integers(0, len(nodes_not_terminated))]  # 44.8 per hit
        #     point_state = n.instance
        #     return point_state.next_trial
        # else:
        #     return None

        candidates = list(self.sequential_indices + self.eager_indices)
        self.rng.shuffle(candidates)  # Much faster than random.shuffle
        for i in candidates:
            n = self.nodes[i]
            if not n.instance.terminated:
                point_state = n.instance
                return point_state.next_trial
        return None

    def with_trial(self, trial: Trial) -> State:
        # Create a copy of the existing point nodes
        nodes = list(self.nodes)
        # Add this trial to the linked list of point states
        point_index = trial.point.index
        nodes[point_index] = nodes[point_index].add_trial(trial)
        # Update growth map
        sequential_indices, nodes = self._update_sequential(self.sequential_indices, nodes, self.quadrant_map)
        eager_indices, nodes = self._update_eager(self.eager_indices, nodes, self.quadrant_map)

        return evolve(
            self,
            nodes=tuple(nodes),
            sequential_indices=tuple(sequential_indices),
            eager_indices=tuple(eager_indices),
            # rng=self.rng
        )

    @cached_property
    def estimate(self):
        return tuple(n.instance.estimate for n in self.nodes)

    @cached_property
    def terminated(self) -> bool:
        return all(n.instance.terminated for n in self.nodes)

    @staticmethod
    def _update_sequential(sequential_indices: List[int], nodes: List[StateNode], quadrant_map: Dict[int, List[int]]) -> (List[int], List[StateNode]):
        if all(nodes[point_index].instance.terminated for point_index in sequential_indices):
            # Update them
            ret = []
            for point_index in sequential_indices:
                offset = nodes[point_index].instance.estimate - nodes[point_index].instance.pretest
                children = quadrant_map.get(point_index, [])
                ret.extend(children)
                for c in children:
                    nodes[c] = nodes[c].add_state(
                        state=nodes[c].instance.with_offset(offset),
                        trial=None
                    )
            return ret, nodes
        else:
            return sequential_indices, nodes

    @staticmethod
    def _update_eager(eager_indices, nodes, quadrant_map):
        eager_indices = list(eager_indices)
        i = 0
        while i < len(eager_indices):
            if nodes[eager_indices[i]].instance.terminated:
                point_index = eager_indices.pop(i)
                offset = nodes[point_index].instance.estimate - nodes[point_index].instance.pretest
                children = quadrant_map.get(point_index, [])
                eager_indices.extend(children)
                for c in children:
                    nodes[c] = nodes[c].add_state(
                        state=nodes[c].instance.with_offset(offset),
                        trial=None
                    )
            else:
                i += 1

        return tuple(eager_indices), nodes
