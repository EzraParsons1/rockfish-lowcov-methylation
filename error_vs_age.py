import numpy as np
import matplotlib.pyplot as plt
from simulate import simulate
from estimators import estimate
from clock import pred_rel

depths = [3, 5 ,8, 12, 15, 30]
n_seeds = 5
n_specimens = 90
errors = np.zeros((len(depths), n_seeds, n_specimens))
ages = np.zeros((len(depths), n_seeds, n_specimens))
edges = np.linspace(0, 1, 11)
centers = (edges[:-1] + edges[1:]) / 2
bin_means = np.full(10, np.nan)

for di, depth in enumerate(depths):
    for seed in range(n_seeds):
        data = simulate(seed=seed, coverage=depth, n_specimens=90)
        X = estimate(data["m"], data["n"] - data["m"], "v3b")
        errors[di, seed] = pred_rel(X, data) - data["relative_age"]
        ages[di, seed] = data["relative_age"]

    e = errors[di].ravel()
    a = ages[di].ravel()

    bin_idx = np.digitize(a, edges)
    for i in range(1, 11):
        bin_means[i - 1] = e[bin_idx == i].mean()
    plt.plot(centers, bin_means, marker="o", label=f"{depth}X")

plt.axhline(0, color="gray", lw=0.8)
plt.xlabel("true relative age")
plt.ylabel("mean signed error")
plt.legend()
plt.title("mean error vs relative age")
plt.savefig("figures/error_vs_age.png", dpi=120)