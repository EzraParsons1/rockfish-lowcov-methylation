import numpy as np
import matplotlib.pyplot as plt
from simulate import simulate
from estimators import estimate

coverages = [3, 5, 8, 12, 15, 30]
n_seeds = 50
methods = ["naive", "v1", "v2", "v3", "v3b"]
results = {meth: np.zeros((len(coverages), n_seeds)) for meth in methods}

for ci, c in enumerate(coverages):
        for s in range(n_seeds):
                data = simulate(seed=s, coverage=c)
                u = data["n"] - data["m"]
                for meth in methods:
                        with np.errstate(divide="ignore", invalid="ignore"):
                                est = estimate(data["m"], u, meth)
                        results[meth][ci, s] = np.mean(np.abs(est - data["true_p"]))

for meth in methods:
        mean = results[meth].mean(axis=1)
        std  = results[meth].std(axis=1)
        plt.plot(coverages, mean, marker="o", label=meth)
        plt.fill_between(coverages, mean - std, mean + std, alpha=0.2)
plt.xlabel("coverage")
plt.ylabel("MAE vs true methylation")
plt.title("methods benchmark")
plt.legend()
plt.savefig("figures/methods_benchmark.png", dpi=120)
