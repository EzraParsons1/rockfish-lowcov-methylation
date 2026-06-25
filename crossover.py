import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import LeaveOneGroupOut
from simulate import simulate
from estimators import estimate
from estimators import impute_nans

def naive_features(data):
    with np.errstate(invalid="ignore", divide="ignore"):
        prop = data["m"] / data["n"]
    site_mean = np.nanmean(prop, axis=0)
    X = np.where(np.isnan(prop), site_mean, prop)
    return np.where(np.isnan(X), 0.5, X)


def loso_mae_years(X, data):
    clock = make_pipeline(
        StandardScaler(),
        ElasticNetCV(l1_ratio=0.5, cv=3, max_iter=5000, n_jobs=-1),
    )
    pred_rel = cross_val_predict(
        clock, X, data["relative_age"],
        cv=LeaveOneGroupOut(), groups=data["species"],
    )
    pred_years = pred_rel * data["lifespans"][data["species"]]
    return np.mean(np.abs(pred_years - data["age"]))


coverages = [3, 5, 8, 12, 20, 30]
nseeds = 3

naive_maes, v3b_maes = [], []
for c in coverages:
    n_runs, v_runs = [], []
    for s in range(nseeds):
        data = simulate(seed=s, coverage=c)
        n_runs.append(loso_mae_years(naive_features(data), data))
        v_runs.append(loso_mae_years(estimate(data["m"], data["n"] - data["m"]), data))
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
