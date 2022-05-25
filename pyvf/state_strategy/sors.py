import numpy as np
from attrs import Factory, evolve
from functools import cached_property
from itertools import chain
from . import *


# noinspection PyPep8Naming
def train_sors(database, print_progress=False):
    M = database.shape[1]
    N = len(database)
    X = database.T  # Training data
    assert X.shape == (M, N)
    Omega = set(range(0, M))  # location set
    S = M

    Omega_star_S = []
    D_star = []

    for k in range(1, S + 1):
        error = {}
        D_k = {}

        for l in Omega - set(Omega_star_S):
            Omega_km1_l = Omega_star_S[:k - 1] + [l]
            Y_Omega_km1_l = X[Omega_km1_l, :]
            D_l_k = X.dot(Y_Omega_km1_l.T.dot(np.linalg.inv(Y_Omega_km1_l.dot(Y_Omega_km1_l.T))))
            X_hat = D_l_k.dot(Y_Omega_km1_l)
            error[l] = np.linalg.norm(X - X_hat)
            D_k[l] = D_l_k

        l_star_k = min(error, key=error.get)  # Equivalent to: min(d.keys(), key=lambda x: d[x])
        Omega_star_S.append(l_star_k)
        D_star.append(D_k[l_star_k])
        if print_progress:
            print(f"{len(Omega_star_S) = :2d}, {error[l_star_k] = :6.1f}: {Omega_star_S[-1] = }")

    return Omega_star_S, D_star


@attr.s(auto_attribs=True, slots=False, kw_only=True, frozen=True)
class GenericSorsFieldState(State):
    nodes: tuple[StateNode]  # PointStates
    batches: tuple[tuple[int]]
    curr_batch_index: int = 0
    models: tuple  # Model should define a predict(X) method that maps partial measurements to full field
    rng: np.random.Generator = Factory(lambda: np.random.default_rng(0))

    @property
    def next_trial(self) -> Optional[Trial]:
        if self.curr_batch_index >= len(self.batches):
            return None

        batch = self.batches[self.curr_batch_index]
        if batch:
            batch = list(batch)
            self.rng.shuffle(batch)

            for i in batch:
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

        if self.curr_batch_index >= len(self.batches):
            return evolve(
                self,
                nodes=tuple(nodes)
            )

        # Move to the next batch if current batch is finished
        batch = self.batches[self.curr_batch_index]
        if batch and all(nodes[i].instance.terminated for i in batch):
            # Seed the points
            completed_indices = chain.from_iterable(self.batches[:self.curr_batch_index+1])  # Flatten up to current finished batch
            completed_estimates = [self.nodes[i].instance.estimate for i in completed_indices]
            n_completed = len(completed_estimates)
            model = self.models[n_completed-1]
            reconstruction = model.predict(completed_estimates)
            for i, (n, r) in enumerate(zip(nodes, reconstruction)):
                if i < n_completed:
                    pass
                else:
                    nodes[i] = n.add_state(
                        state=n.instance.with_offset(r - n.instance.pretest),
                        trial=None
                    )
            return evolve(
                self,
                nodes=tuple(nodes),
                curr_batch_index=self.curr_batch_index + 1
            )
        else:
            return evolve(
                self,
                nodes=tuple(nodes)
            )

    @cached_property
    def estimate(self):
        return tuple(n.instance.estimate for n in self.nodes)

    @cached_property
    def terminated(self) -> bool:
        return all(n.instance.terminated for n in self.nodes)
