# demo.py
import numpy as np
import matplotlib.pyplot as plt

from sca import SineCosineAlgorithm


def sphere(x: np.ndarray) -> float:
    """Sphere benchmark function: f(x) = sum(x_i^2)."""
    return float(np.sum(x ** 2))


def main():
    dim = 30
    n_agents = 30
    max_iter = 100
    lower, upper = -10.0, 10.0

    sca = SineCosineAlgorithm(
        obj_func=sphere,
        dim=dim,
        n_agents=n_agents,
        max_iter=max_iter,
        lower=lower,
        upper=upper,
        a=2.0,
        seed=42,
    )

    best_pos, best_fit, history = sca.optimize()

    print("=== Sine Cosine Algorithm (SCA) Demo ===")
    print(f"Dimension: {dim}")
    print(f"Agents   : {n_agents}")
    print(f"Iterations: {max_iter}")
    print(f"Best fitness found: {best_fit:.6e}")
    print("Best position (first 5 dims):", best_pos[:5])

    # Plot convergence curve
    plt.figure(figsize=(6, 4))
    plt.plot(history)
    plt.xlabel("Iteration")
    plt.ylabel("Best fitness")
    plt.title("SCA Convergence on Sphere Function")
    plt.yscale("log")  # optionnel: échelle log
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
