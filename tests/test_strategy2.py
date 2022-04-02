"""

Copyright 2022 Bill Runjie Shi
At the Vision and Eye Movements Lab, University of Toronto.
Visit us at: http://www.eizenman.ca/

This file is part of PyVF.

PyVF is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

PyVF is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with PyVF. If not, see <https://www.gnu.org/licenses/>.
"""
from unittest import TestCase

from pyvf.state_strategy import *
from pyvf.state_strategy.quadrant import *
from pyvf.state_strategy.zest import *
import attrs.exceptions


class TestStrategy(TestCase):
    def test_strategy(self):
        t = Trial(point=Point(index=0, x=0, y=0), threshold=30)
        def fun():
            t.seen = True
        self.assertRaises(attrs.exceptions.FrozenError, fun)
        t1 = attrs.evolve(t)
        self.assertEqual(t, t1)
        self.assertIsNot(t, t1)

        # def fun():
        #     State(prev=None, next=None, trials=tuple()).next_trial
        #     State(prev=None, next=None, trials=tuple()).estimates
        # self.assertRaises(RuntimeError, fun)

        # print(Wrapper().rng is Wrapper().rng)
        # print(Wrapper().rng.random())
        # print(Wrapper().rng.random())

        # @define
        # class Parent(ABC):
        #     @property
        #     def field(self) -> int:
        #         return 0
        #
        # @define
        # class Child(Parent):
        #     field: int = 10
        #
        # print(Child().field)

    def test_zest(self):
        def pos(x):
            return (x > 0) * 0.8 + 0.1

        point = ZestPointState(
            point=Point(index=0, x=0, y=0),
            x=np.array([0, 30]),
            q0=np.array([0.1, 0.9]),
            pos_fun=pos,
            pretest=30
        )
        print(point)

    def test_growth(self):
        @attr.s
        class MockTerminatedState(State):
            pretest = attr.ib(default=30)
            starting = attr.ib(default=30)

            @property
            def terminated(self) -> bool:
                return True

            @property
            def next_trial(self) -> Optional[Trial]:
                pass

            def with_trial(self, trial: Trial) -> State:
                pass

            @property
            def estimate(self):
                return 20

        class MockNotTerminatedState(MockTerminatedState):
            @property
            def terminated(self) -> bool:
                return False

        print(
            QuadrantFieldState._update_eager(
                (0, 2),
                [StateNode(instance=MockTerminatedState()), StateNode(instance=MockNotTerminatedState()), StateNode(instance=MockNotTerminatedState())],
                {0: [1]}
            )
        )

    def test_quadrant_field_state(self):
        def run():
            field_state_node = StateNode(instance=QuadrantFieldState.new_instance())
            trial = field_state_node.instance.next_trial
            i = 0
            while trial is not None:
                i += 1
                seen = trial.threshold < 30 if trial.point.index != 34 else False
                trial = evolve(trial, seen=seen)
                field_state_node = field_state_node.add_trial(trial)
                # print(i, trial)
                trial = field_state_node.instance.next_trial
            print(np.around(field_state_node.instance.estimate))

        from line_profiler.line_profiler import LineProfiler
        profiler = LineProfiler()
        # profiler.add_function(StateNode.add_trial)
        # profiler.add_function(QuadrantFieldState._update_sequential)
        # profiler.add_function(ZestPointState.with_trial)
        # profiler.add_function(pos_ramp)
        profiler.add_function(QuadrantFieldState.next_trial.fget)
        profiler.add_function(ZestPointState.var.func)
        profiler.add_function(ZestPointState.q.func)
        profiler(run)()
        print(profiler.print_stats())
