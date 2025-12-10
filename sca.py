# sca.py
import numpy as np


class SineCosineAlgorithm:
    """
    Simple implementation of the Sine Cosine Algorithm (SCA)
    for continuous optimization.
    """

    def __init__(
        self,
        obj_func,
        dim,
        n_agents=30,
        max_iter=100,
        lower=-10.0,
        upper=10.0,
        a=2.0,
        seed=None,
    ):
        self.obj_func = obj_func          # objective function f(x)
        self.dim = dim                    # dimension of the problem
        self.n_agents = n_agents          # population size
        self.max_iter = max_iter          # number of iterations
        self.lower = lower                # lower bound
        self.upper = upper                # upper bound
        self.a = a                        # initial value for r1
        self.rng = np.random.default_rng(seed)

        # Initialize population
        self.positions = self.rng.uniform(
            low=self.lower,
            high=self.upper,
            size=(self.n_agents, self.dim)
        )

        # Evaluate initial fitness
        self.fitness = np.apply_along_axis(self.obj_func, 1, self.positions)

        # Identify best agent
        best_idx = np.argmin(self.fitness)
        self.best_position = self.positions[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]

    def _apply_bounds(self, positions):
        """Clamp positions to [lower, upper]."""
        return np.clip(positions, self.lower, self.upper)

    def optimize(self):
        """
        Run the SCA optimization process.

        Returns
        -------
        best_position : np.ndarray
            Best solution found.
        best_fitness : float
            Fitness of the best solution.
        history : list[float]
            Best fitness value at each iteration.
        """
        history = []

        for t in range(self.max_iter):
            # r1 decreases linearly from a to 0
            r1 = self.a - t * (self.a / self.max_iter)

            # For each agent, update its position
            for i in range(self.n_agents):
                r2 = self.rng.uniform(0, 2 * np.pi, size=self.dim)  # angle
                r3 = self.rng.uniform(0, 2, size=self.dim)          # distance weight
                r4 = self.rng.uniform(0, 1, size=self.dim)          # sine/cosine switch

                # Vector from agent to best
                diff = np.abs(r3 * self.best_position - self.positions[i])

                # Sine or cosine update
                sine_mask = r4 < 0.5
                cosine_mask = ~sine_mask

                new_pos = self.positions[i].copy()

                # sine update
                new_pos[sine_mask] = (
                    self.positions[i, sine_mask]
                    + r1 * np.sin(r2[sine_mask]) * diff[sine_mask]
                )

                # cosine update
                new_pos[cosine_mask] = (
                    self.positions[i, cosine_mask]
                    + r1 * np.cos(r2[cosine_mask]) * diff[cosine_mask]
                )

                # Apply bounds
                new_pos = self._apply_bounds(new_pos)

                # Evaluate new fitness
                new_fit = self.obj_func(new_pos)

                # Greedy replacement
                if new_fit < self.fitness[i]:
                    self.positions[i] = new_pos
                    self.fitness[i] = new_fit

                    # Update global best
                    if new_fit < self.best_fitness:
                        self.best_fitness = new_fit
                        self.best_position = new_pos.copy()

            history.append(self.best_fitness)

        return self.best_position, self.best_fitness, history
