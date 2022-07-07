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
from pyvf.state_strategy.fest import FestFieldState
from pyvf.state_strategy.quadrant import *
from pyvf.state_strategy.staircase import *
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
            return (x < 0) * 0.9 + 0.05

        point = ZestPointState(
            point=Point(index=0, x=0, y=0),
            x=np.array([0, 30]),
            q0=np.array([0.1, 0.9]),
            pos_fun=pos
        )
        self.assertEqual(point.mean, 27)
        self.assertEqual(point.estimate, 30)
        point = point.with_trial(Trial(point=point.point, threshold=20, seen=True))
        self.assertEqual(point.estimate, 30 * 0.9 * 0.95 / (0.9 * 0.95 + 0.1 * 0.05))

        point = ZestPointState(
            point=Point(index=0, x=0, y=0),
            x=np.array([0, 30]),
            q0=np.array([0.1, 0.9]),
        )
        point = point.with_trial(Trial(point=point.point, threshold=20, seen=True))
        self.assertAlmostEqual(point.estimate, 30 * 0.9 * 0.95 / (0.9 * 0.95 + 0.1 * 0.05))

        point = ZestPointState(
            point=Point(index=0, x=0, y=0),
            x=np.array([-1,  1,  3,  5,  7,  9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33]),
            q0=np.array([0.09039887, 0.00844819, 0.01132768, 0.00774113, 0.0106241 ,
       0.01187741, 0.01382238, 0.01406786, 0.02154017, 0.02463611,
       0.03745739, 0.04815837, 0.0587623 , 0.06639628, 0.04909913,
       0.01998311, 0.00480609, 0.00085342])
        )

        self.assertEqual(point.estimate, 25)
        self.assertEqual(point.with_offset(-0.9).estimate, 25)
        self.assertEqual(point.with_offset(-2).estimate, 23)
        self.assertEqual(point.with_offset(+3.1).estimate, 29)

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
            field_state_node = StateNode(instance=QuadrantFieldState(
                nodes=[
                    StateNode(instance=
                        ZestPointState(
                            point=Point(index=pt[LOC], x=pt[XOD], y=pt[YOD]),
                            x=np.arange(0, 40, 1),
                            q0=np.ones(40, dtype=float),
                            terminate_std=1.5
                        )
                        # StaircasePointState(
                        #     point=Point(index=pt[LOC], x=pt[XOD], y=pt[YOD]),
                        #     pretest=30
                        # )
                    ) for pt in PATTERN_P24D2
                ],
                quadrant_map=get_quadrant_map("P24D2")
            ))
            trial = field_state_node.instance.next_trial
            i = 0
            while trial is not None:
                i += 1
                seen = trial.threshold <= 30 if trial.point.index != 34 else False
                trial = evolve(trial, seen=seen)
                field_state_node = field_state_node.add_trial(trial)
                # print(i, trial, np.around(field_state_node.instance.estimate))
                trial = field_state_node.instance.next_trial
            # print(np.around(field_state_node.instance.estimate))
            return field_state_node

        from line_profiler.line_profiler import LineProfiler
        profiler = LineProfiler()
        profiler.add_function(run)
        # profiler.add_function(StateNode.add_trial)
        # profiler.add_function(QuadrantFieldState._update_sequential)
        # profiler.add_function(ZestPointState.with_trial)
        # profiler.add_function(StaircasePointState.with_trial)
        # profiler.add_function(pos_ramp)
        # profiler.add_function(QuadrantFieldState.next_trial.fget)
        # profiler.add_function(ZestPointState.var.func)
        # profiler.add_function(ZestPointState.q.func)
        # profiler.add_function(StaircasePointState.next_trial.func)
        # profiler.add_function(StaircasePointState.terminated.func)
        profiler(lambda : [run() for _ in range(1)])()
        print(profiler.print_stats())

    def test_var(self):
        self.assertAlmostEqual(FestFieldState.var(
            p=np.array([0.25, 0.75]),
            x=np.array([0, 1])
        ), 0.25 * 0.75)
        self.assertAlmostEqual(FestFieldState.var(
            p=np.array([0.25, 0.25, 0.25, 0.25]),
            x=np.array([0, 1, 1, 1])
        ), 0.25 * 0.75)

    def test_shift(self):
        x = [10, 1, 2, 3, 4, 5]
        self.assertListEqual(shift(x, 0).tolist(), [10, 1, 2, 3, 4, 5])
        self.assertListEqual(shift(x, 1).tolist(), [10.0, EPS, 1.0, 2.0, 3.0, 9.0])
        self.assertListEqual(shift(x, 3).tolist(), [10.0, EPS, EPS, EPS, 1.0, 14.0])
        self.assertListEqual(shift(x, -1).tolist(), [10.0, 3.0, 3.0, 4.0, 5.0, EPS])
        self.assertListEqual(shift(x, -3).tolist(), [10.0, 10.0, 5.0, EPS, EPS, EPS])
        self.assertListEqual(shift(x, -3, low_indices=2).tolist(), [10.0, 1.0, 14.0, EPS, EPS, EPS])

    def test_staircase(self):
        s = StaircasePointState(point=None, pretest=30, steps=(3, ))
        print(s)
        while not s.terminated:
            trial = s.next_trial
            trial = evolve(trial, seen=trial.threshold < -1)
            print(trial)
            s = s.with_trial(trial)
            print(s)
