import numpy as np
import matplotlib.pyplot as plt
from simulate import simulate
from crossover import loso_mae_years
from estimators import estimate

def subsampler(X, fraction, rng):
     subsample = rng.choice(X.shape[1], size=round(X.shape[1] * fraction), replace=False)
     return X[:, subsample]

depths = [3, 5, 8, 12, 15, 30]
n_seeds = 5
fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
cube = np.zeros((len(depths), len(fractions), n_seeds))

for di, depth in enumerate(depths):
    for seed in range(n_seeds):
        data = simulate(coverage=depth, n_sites=10000, seed=seed)
        X_full = estimate(data["m"], data["n"] - data["m"], "v3b")
        rng = np.random.default_rng(seed)
        for fi, fraction in enumerate(fractions):
            X = subsampler(X_full, fraction, rng)
            cube[di, fi, seed] = loso_mae_years(X, data)

np.save("cube.npy", cube)
MAEs = np.median(cube, axis=2)
print(np.round(MAEs, 2))

# data = simulate(n_sites=10000, coverage=5, seed=0)
# X_full = estimate(data["m"], data["n"] - data["m"], "v3b")
# X = subsampler(X_full, 0.5, np.random.default_rng(0))
# print(loso_mae_years(X, data))
