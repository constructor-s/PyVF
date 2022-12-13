from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Any, Union, Tuple, List

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
    seen: bool = None  # None indicates a trial that has not been presented yet


@attr.s(auto_attribs=True, slots=False, kw_only=True, frozen=False)  # if frozen=True then subclasses are forced to be frozen
class State(ABC):
    @property
    @abstractmethod
    def next_trial(self) -> Optional[Trial]:
        """

        Returns
        -------
        Next trial to be presented or None if there is no more trials possible

        Notes
        -------
        This does not necessarily return None if the state is terminated
        """
        pass

    @abstractmethod
    def with_trial(self, trial: Trial) -> State:
        """

        Parameters
        ----------
        trial : new trial that has been presented and responded or timed out

        Returns
        -------
        A new state given the new information.

        Notes
        -------
        This class is immutable and this method does not modify the current state in place
        """
        pass

    @property
    @abstractmethod
    def estimate(self) -> Union[float, Tuple[float]]:
        """

        Returns
        -------
        Estimate of the threshold at the current point or multiple points
        """
        pass

    @property
    @abstractmethod
    def terminated(self) -> bool:
        pass


@attr.s(auto_attribs=True, slots=False, kw_only=True, frozen=False)  # if frozen=True then subclasses are forced to be frozen
class PointState(State, ABC):
    point: Point

    @abstractmethod
    def with_offset(self, db) -> PointState:
        """
        Return a new copy of the current state with the starting threshold shifted,
        typically with respect to pretest threshold,
        used for seeding procedures

        Parameters
        ----------
        db : Amount of offset in decibels to shift up (positive input) or down (negative input)

        Returns
        -------
        A new updated copy of the current state
        """
        pass


@attr.s(auto_attribs=True, slots=True, kw_only=True, frozen=False, str=False, repr=False)
class StateNode:
    """
    Generic wrapper around a state instance representing a node in the testing graph
    with references to the previous and next edge.
    The state may be a field state or a point state.
    """
    prev: TrialEdge = None
    next: TrialEdge = None
    instance: State

    def add_trial(self, trial: Trial) -> StateNode:
        """
        Add a trial to the graph

        Parameters
        ----------
        trial

        Returns
        -------
        The state node after adding this trial
        """
        next_state = self.instance.with_trial(trial)
        return self.add_state(next_state, trial)

    def add_state(self, state: State, trial: Optional[Trial]) -> StateNode:
        """
        Add a state to the graph

        Parameters
        ----------
        state : Next state
        trial : A Trial instance if a trial is used to arrived at this new state, or None

        Returns
        -------
        The state node after adding this trial
        """
        # Get the next node
        next_node = StateNode(instance=state)
        # Generate an edge
        edge = TrialEdge(prev=self, next=next_node, instance=trial)
        # Connect the graph
        self.next = edge
        next_node.prev = edge
        return next_node

    @property
    def previous_edges(self) -> Tuple[TrialEdge]:
        """

        Returns
        -------
        All edges from root to the current state
        """
        return tuple(self._previous_edges)

    @property
    def _previous_edges(self) -> List[TrialEdge]:
        if self.prev is None:
            return []
        else:
            ret = self.prev.prev._previous_edges
            ret.append(self.prev)
            return ret

    @property
    def previous_states(self) -> Tuple[StateNode]:
        """

        Returns
        -------
        All states from root to the current state, including the current state (self)
        """
        return tuple(self._previous_states)

    @property
    def _previous_states(self) -> List[StateNode]:
        if self.prev is None or self.prev.prev is None:
            return [self]
        else:
            ret = self.prev.prev._previous_states
            ret.append(self)
            return ret

    def __str__(self):
        return str(self)

    def __repr__(self):
        """
        Override repr to remove long recursions

        Returns
        -------

        """
        return f"StateNode(prev={self.prev.__class__}, next={self.next.__class__}, instance={self.instance.__class__})"


@attr.s(auto_attribs=True, slots=True, kw_only=True, frozen=False, str=False, repr=False)
class TrialEdge:
    """
    Generic wrapper around a trial instance representing an edge in the testing graph
    with references to the previous and next edge.
    The instance may also be None to represent an update in the state that is not
    triggered by a new trial, such as a seeding event.
    """
    prev: StateNode = None
    next: StateNode = None
    instance: Optional[Trial]

    def __str__(self):
        return str(self)

    def __repr__(self):
        """
        Override repr to remove long recursions

        Returns
        -------

        """
        return f"TrialEdge(prev={self.prev.__class__}, next={self.next.__class__}, instance={self.instance.__class__})"
