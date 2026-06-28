import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simulate import simulate
from estimators import estimate
from clock import naive_features
from clock import pred_rel
from clock import loso_mae_years

coverages = [3, 5, 8, 12, 15, 30]
nseeds = 3

naive_maes, v3b_maes = [], []
for c in coverages:
    n_runs, v_runs = [], []
    for s in range(nseeds):
        data = simulate(seed=s, coverage=c)
        n_runs.append(loso_mae_years(naive_features(data), data))
        v_runs.append(loso_mae_years(estimate(data["m"], data["n"] - data["m"], "v3b"), data))
    naive_maes.append(np.mean(n_runs))
    v3b_maes.append(np.mean(v_runs))
    print(f"coverage {c:>2}:  naive {naive_maes[-1]:5.2f}   v3b {v3b_maes[-1]:5.2f}")

plt.figure()
plt.plot(coverages, naive_maes, marker="o", label="naive + impute")
plt.plot(coverages, v3b_maes, marker="o", label="v3b shrinkage")
plt.xlabel("coverage")
plt.ylabel("LOSO MAE (years)")
plt.title("When does shrinking clock features help? (crossover)")
plt.legend()
plt.savefig("figures/crossover.png", dpi=120)
