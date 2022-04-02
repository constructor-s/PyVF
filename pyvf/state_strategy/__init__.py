from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import attr


@attr.s(auto_attribs=True, slots=True, kw_only=True, frozen=True)
class Point:
    index: int
    x: float
    y: float


@attr.s(auto_attribs=True, slots=True, kw_only=True, frozen=True)
class Trial:
    point: Point
    threshold: float
    seen: bool = None


@attr.s(auto_attribs=True, slots=False, kw_only=True, frozen=True)
class State(ABC):
    @property
    @abstractmethod
    def next_trial(self) -> Optional[Trial]:
        pass

    @abstractmethod
    def with_trial(self, trial: Trial) -> State:
        pass

    @property
    @abstractmethod
    def estimate(self):
        pass

    @property
    @abstractmethod
    def terminated(self) -> bool:
        pass


@attr.s(auto_attribs=True, slots=False, kw_only=True, frozen=True)
class PointState(State, ABC):
    point: Point

    @property
    @abstractmethod
    def estimate(self) -> float:
        pass


@attr.s(auto_attribs=True, slots=True, kw_only=True, frozen=False)
class StateNode:
    prev: TrialEdge = None
    next: TrialEdge = None
    instance: State

    def add_trial(self, trial: Trial) -> StateNode:
        # Get the next node
        next_state = self.instance.with_trial(trial)
        return self.add_state(next_state, trial)

    def add_state(self, state: State, trial: Optional[Trial]) -> StateNode:
        next_node = StateNode(instance=state)
        # Generate an edge
        edge = TrialEdge(prev=self, next=next_node, instance=trial)
        # Connect the graph
        self.next = edge
        next_node.prev = edge
        return next_node

    # @property
    # def last_node(self):
    #     # Traverse through linked list
    #     state = self
    #     edge = state.next
    #     while edge is not None:
    #         state = edge.next
    #         edge = state.next
    #     return state

    @property
    def previous_edges(self):
        return tuple(self._previous_edges)

    @property
    def _previous_edges(self):
        if self.prev is None:
            return []
        else:
            ret = self.prev.prev._previous_edges
            ret.append(self.prev)
            return ret


@attr.s(auto_attribs=True, slots=True, kw_only=True, frozen=False)
class TrialEdge:
    prev: StateNode = None
    next: StateNode = None
    instance: Optional[Trial]


# @define(slots=False, kw_only=True, frozen=True)
# class GrowthMapFieldState(State):
#     point_states: tuple[PointState]
#     growth_map_roots: tuple[GrowthMapNode]
#     active_nodes: tuple[GrowthMapNode]
#     rng: np.random.Generator = Factory(lambda: np.random.default_rng(0))
#
#     @cached_property
#     def _calculated(self):
#         return
#
#     @property
#     def _next_trials(self) -> list[Trial]:
#         return self._calculated[0]
#
#     @property
#     def next_trial(self) -> Trial:
#         return self.rng.choice(self._next_trials)
#
#     @property
#     def estimate(self) -> np.ndarray:
#         return np.fromiter(chain(
#             s.estimate for s in self.point_states
#         ), dtype=float)
#
#     @property
#     def terminated(self) -> bool:
#         return all(s.terminated for s in self.point_states)
#
#     def with_trial(self, trial: Trial) -> GrowthMapFieldState:
#         point_states = list(self.point_states)
#         try:
#             i, state = next(filter(lambda x: x[1].point == trial.point, enumerate(self.point_states)))
#             point_states[i] = state.with_trial(trial)
#             point_states = self._grow(point_states)
#         except StopIteration:
#             pass
#
#         return attrs.evolve(
#             self,
#             trials=self.past_trials + (trial,),
#             point_states=tuple(point_states)
#         )
#
#     def _grow(self, point_states):
#         self.growth_map_roots
#         return point_states

    # @cached_property
    # def _info(self):
    #     curr = self.growth_map_roots
    #     estimates = []
    #     terminated = []
    #     trials = []
    #     while curr:
    #         for node in curr:
    #             point = self.points[node.index]
    #             state = self.point_states[node.index]
    #             if not state.terminated:
    #                 self.point_states[node.index] = node.adjust(state)
    #                 trials.append(Trial(point=point, threshold=state.next_trial_threshold, seen=None))
    #             terminated.append(state.terminated)
    #         if all(terminated):
    #             # Go one level deeper
    #             curr = chain.from_iterable([node.test_children for node in curr])


