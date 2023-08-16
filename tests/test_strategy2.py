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
from pyvf.state_strategy.basic import *
from pyvf.state_strategy.quadrant import *
from pyvf.state_strategy.sors import *
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
        self.assertEqual(point.mode, 30)
        self.assertEqual(point.pretest, 30)
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

    def test_new_zest_with_prior(self):
        prior = BayesianMixedPrior(
            x=np.array([0, 15, 30]),
            q0_normal=np.array([1, 0, 0]),
            q0_abnormal=np.array([0, 0, 1]),
            weight_normal=0.75,
            weight_abnormal=0.25,
            eps=0.01
        )
        self.assertAlmostEqual(prior.q0.sum(), 1.0)
        self.assertGreater(prior.q0[1], 0.0)

        x = np.array([-1, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33])
        normal = np.array([0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 2, 1, 1])
        abnormal = np.array([8, 4, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        prior0 = BayesianMixedPrior(
            x=x,
            q0_normal=normal,
            q0_abnormal=abnormal,
            weight_normal=0.75,
            weight_abnormal=0.25,
            eps=0
        )
        self.assertAlmostEqual(prior0.q0.sum(), 1.0)
        self.assertEqual(prior0.pretest, 27)

        prior = prior0.with_offset(-4)
        self.assertAlmostEqual(prior.q0.sum(), 1.0)
        self.assertEqual(prior.pretest, 23)
        # point = ZestPointState(
        #     point=Point(index=0, x=0, y=0),
        #     prior=BayesianPrior(
        #         x=x,
        #         q0=normal,
        #         low_indices_db=1
        #     )
        # )

        prior = prior0.with_offset(+12).with_offset(-12)
        self.assertListEqual(prior0.q0.tolist(), prior.q0.tolist())

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

            def with_offset(self, db):
                return evolve(self, starting=self.pretest + db)

        class MockNotTerminatedState(MockTerminatedState):
            @property
            def terminated(self) -> bool:
                return False

        nodes = [
            StateNode(instance=MockTerminatedState()),
            StateNode(instance=MockNotTerminatedState()),
            StateNode(instance=MockNotTerminatedState())
        ]
        self.assertListEqual([n.instance.starting for n in nodes], [30, 30, 30])
        eager_indices, nodes = (
            QuadrantFieldState._update_eager(
                eager_indices=(0, 2),
                nodes=nodes,
                quadrant_map={0: [1]}
            )
        )
        self.assertSetEqual(set(eager_indices), set((1, 2)))  # 0 has terminated, so 1 can be evaluated now
        self.assertListEqual([n.instance.starting for n in nodes], [30, 20, 30])

    def test_random_field_state(self):
        def run():
            field_state_node = StateNode(instance=RandomFieldState(
                nodes=[
                    StateNode(instance=
                        ZestPointState(
                            point=Point(index=pt[LOC], x=pt[XOD], y=pt[YOD]),
                            prior=BayesianMixedPrior(
                                x=np.arange(0, 40, 1), q0_normal=np.ones(40, dtype=float), q0_abnormal=np.ones(40, dtype=float), weight_normal=1.0, weight_abnormal=0.0
                            ),
                            terminate_std=1.5
                        )
                        # StaircasePointState(
                        #     point=Point(index=pt[LOC], x=pt[XOD], y=pt[YOD]),
                        #     pretest=30
                        # )
                    ) for pt in PATTERN_P24D2
                ]
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

        expected = np.full(54, fill_value=30)
        expected[34] = 0
        result = run()
        self.assertTrue(np.allclose(result.instance.estimate, expected, atol=2))

    def test_quadrant_field_state(self):
        def run():
            field_state_node = StateNode(instance=QuadrantFieldState(
                nodes=[
                    StateNode(instance=
                        ZestPointState(
                            point=Point(index=pt[LOC], x=pt[XOD], y=pt[YOD]),
                            prior=BayesianMixedPrior(
                                x=np.arange(0, 40, 1), q0_normal=np.ones(40, dtype=float),
                                q0_abnormal=np.ones(40, dtype=float), weight_normal=1.0, weight_abnormal=0.0
                            ),
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

        expected = np.full(54, fill_value=30)
        expected[34] = 0
        result = run()
        self.assertTrue(np.allclose(result.instance.estimate, expected, atol=2))
        """
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
        """

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
        x = np.array([10.0, 1, 2, 3, 4, 5])
        EPS = 1e-5
        self.assertListEqual(shift(x, 0, fill_value=EPS).tolist(), [10, 1, 2, 3, 4, 5])
        self.assertListEqual(shift(x, 1, fill_value=EPS).tolist(), [EPS, 10.0, 1.0, 2.0, 3.0, 9.0])
        self.assertListEqual(shift(x, 3, fill_value=EPS).tolist(), [EPS, EPS, EPS, 10.0, 1.0, 14.0])
        self.assertListEqual(shift(x, -1, fill_value=EPS).tolist(), [11.0, 2.0, 3.0, 4.0, 5.0, EPS])
        self.assertListEqual(shift(x, -3, fill_value=EPS).tolist(), [16.0, 4.0, 5.0, EPS, EPS, EPS])

    def test_staircase(self):
        for seed in range(2):
            rng = np.random.default_rng(seed)
            s = StaircasePointState(point=None, pretest=30, steps=(4, 2, 1), retest=True)
            trials_history = []
            # print(s)
            while not s.terminated:
                trial = s.next_trial
                trial = evolve(trial, seen=trial.threshold < 25.1 if rng.random() < 0.8 else rng.random() < 0.5)
                trials_history.append(trial)
                # print(trial)
                s = s.with_trial(trial)
                # print(s)

            # print(trials_history)
            # print(f"s.estimate = {s.estimate}")
            if seed == 0:
                self.assertListEqual([30, 26, 22, 24, 26, 25, 25, 29, 27, 25, 26, 27], [t.threshold for t in trials_history])
                self.assertListEqual([ 0,  0,  1,  1,  0,  1,  1,  0,  0,  1,  1,  0], [t.seen for t in trials_history])
                self.assertEqual(s.estimate, 0.5 * (25 + 26))  # Threshold is the last seen trial, return average of first and second repetation
            elif seed == 1:
                self.assertListEqual([30, 26, 28, 30, 29], [t.threshold for t in trials_history])
                self.assertListEqual([ 0,  1,  1,  0,  1], [t.seen for t in trials_history])
                self.assertEqual(s.estimate, 29)  # Threshold is the last seen trial

    def test_sors(self):
        from collections import defaultdict
        class QuadrantModel:
            map = {
                (0, 12): (0, 1, 4, 5, 6, 10, 11, 13, 18, 19, 20, 21, 22),  # 12
                (1, 15): (2, 3, 7, 8, 9, 14, 16, 17, 23, 24, 26),  # 15
                (2, 38): (27, 28, 29, 30, 31, 36, 37, 39, 44, 45, 46, 50, 51),  # 38
                (3, 41): (32, 33, 35, 40, 42, 43, 47, 48, 49, 52, 53)  # 41
            }

            def __init__(self, hill):
                # hill = Model(eval_pattern=PATTERN_P24D2, age=50)._get_vf_stats_mean(age=50)
                self.hill = hill

            def fit(self, X, y):
                return self

            def predict(self, X):
                X = np.atleast_2d(np.asarray(X))
                y = np.tile(self.hill, (X.shape[0], 1))
                for (i, k), v in QuadrantModel.map.items():
                    if i >= X.shape[1]:
                        break
                    offset = X[:, i] - self.hill[:, k]
                    y[:, v] += offset.reshape(-1, 1)
                    y[:, k] += offset
                y[:, 25] = 0
                y[:, 34] = 0
                return y.ravel()

        field_state = GenericSorsFieldState(
            nodes=[
                StateNode(instance=StaircasePointState(
                        pretest=np.nan_to_num(pretest),
                        point=Point(index=i, x=pt[0], y=pt[1])
                    )
                ) for i, (pretest, pt) in enumerate(zip(np.full(54, 30.0), PATTERN_P24D2))
            ],
            batches=((12,),(15,),(38,),(41,),(27,),(2,),(31,),(10,),(44,),(7,),(29,),(6,),(23,),(48,),(53,),(32,),(33,),(43,),(5,),(4,),(30,),(46,),(47,),(9,),(39,),(0,),(8,),(16,),(11,),(20,),(37,),(24,),(28,),(45,),(52,),(21,),(17,),(35,),(34,),(25,),(13,),(49,),(1,),(19,),(42,),(51,),(14,),(40,),(22,),(3,),(50,),(18,),(36,),(26,)),
            models=defaultdict(lambda : QuadrantModel(hill=np.full((1, 54), 30.0)))
        )
        # for i in field_state.nodes:
        #     print(i.instance)

        final_node = self.run_simulation(field_state, true_threshold=np.full(54, 20.0))
        # Get to first node
        node = final_node
        while node.prev and node.prev.prev:
            node = node.prev.prev
        # Step through linked list of nodes
        for i in range(50):
            starting = np.array([n.instance.starting for n in node.instance.nodes])
            estimate = node.instance.estimate
            if i == 0:
                self.assertTrue(np.allclose(30, starting))
            else:
                for (_, center_index), children_indices in QuadrantModel.map.items():
                    if node.instance.nodes[center_index].instance.terminated:
                        center_estimate = estimate[center_index]
                        # print(center_index, center_estimate)
                        self.assertTrue(np.allclose(starting[list(children_indices)], center_estimate),
                                        msg=f"{i = }, {center_index = }, {estimate[center_index] = }, {starting[list(children_indices)] = }")
            node = node.next.next

        # SORS-ZEST
        field_state = GenericSorsFieldState(
            nodes=[
                StateNode(instance=ZestPointState(
                        point=Point(index=i, x=pt[0], y=pt[1]),
                        prior=BayesianMixedPrior(
                            x=np.arange(0, 37, 2),
                            q0_normal=np.array([0.  , 0.  , 0.  , 0.01, 0.02, 0.04, 0.08, 0.14, 0.24, 0.37, 0.53,  0.7 , 0.85, 0.96, 1.  , 0.5, 0.1, 0. , 0.]), # mode at 28
                            q0_abnormal=np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                            weight_normal=0.8,
                            weight_abnormal=0.2
                        )
                    )
                ) for i, (pretest, pt) in enumerate(zip(np.full(54, 30.0), PATTERN_P24D2))
            ],
            batches=(
            (12,), (15,), (38,), (41,), (27,), (2,), (31,), (10,), (44,), (7,), (29,), (6,), (23,), (48,), (53,), (32,),
            (33,), (43,), (5,), (4,), (30,), (46,), (47,), (9,), (39,), (0,), (8,), (16,), (11,), (20,), (37,), (24,),
            (28,), (45,), (52,), (21,), (17,), (35,), (34,), (25,), (13,), (49,), (1,), (19,), (42,), (51,), (14,),
            (40,), (22,), (3,), (50,), (18,), (36,), (26,)),
            models=defaultdict(lambda: QuadrantModel(hill=np.full((1, 54), 30.0)))
        )

        final_node = self.run_simulation(field_state, true_threshold=np.full(54, 20.0))
        """
        # Get to first node
        node = final_node
        while node.prev and node.prev.prev:
            node = node.prev.prev
        # Step through linked list of nodes
        for [loc] in node.instance.batches:
            # starting = np.array([n.instance.starting for n in node.instance.nodes])
            print(f"{loc = }, {node.instance.nodes[loc].instance.estimate = }")
            while not node.instance.nodes[loc].instance.terminated:
                print(f"{loc = }, {node.instance.nodes[loc].instance.estimate = }, {BayesianMixedPrior.unicode_format(node.instance.nodes[loc].instance.q)}")
                print(f"{node.next.instance}")
                node = node.next.next
            print(f"{loc = }, {node.instance.nodes[loc].instance.estimate = }")
        """

    @staticmethod
    def run_simulation(state, true_threshold, max_trials=600, seed=0):
        field_state_node = StateNode(instance=state)

        trial = field_state_node.instance.next_trial
        i = 0
        while trial is not None:
            seen = np.random.RandomState(seed).uniform() < pos_ramp(trial.threshold, true_threshold[trial.point.index],
                                                                yl=0.999, yr=0.001, width=0.01)
            # print(trial)
            trial = evolve(trial, seen=seen)
            # print(trial)
            field_state_node = field_state_node.add_trial(trial)
            i += 1
            if i > max_trials:
                print(f"Max trials of {max_trials} reached")
                break
            # print(i, trial, np.around(field_state_node.instance.estimate))
            # print(field_state_node.instance.database_p)
            # print(i, field_state_node.instance.seeding_terminated, np.around(field_state_node.instance.estimate, 1).tolist())
            trial = field_state_node.instance.next_trial

        return field_state_node




