import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simulate import simulate
from estimators import estimate
from clock import mae_years
from clock import pred_rel
from clock import calibrate

depths = [3, 5, 8, 12, 15, 30]
n_seeds = 5

raw_maes, calibrated_maes, c1s = [], [], []
for d in depths:
    r_runs, c_runs, c1_runs = [], [], []
    for s in range(n_seeds):
        data = simulate(seed=s, coverage=d)
        X = estimate(data["m"], data["n"] - data["m"], "v3b")
        preds = pred_rel(X, data)
        cal_pred, c1 = calibrate(preds, data)
        r_runs.append(mae_years(preds, data))
        c_runs.append(mae_years(cal_pred, data))
        c1_runs.append(c1)
    raw_maes.append(np.mean(r_runs))
    calibrated_maes.append(np.mean(c_runs))
    c1s.append(np.mean(c1_runs))
    print(f"depth {d:>2}:  raw {raw_maes[-1]:5.2f}   cal {calibrated_maes[-1]:5.2f}   slope {c1s[-1]:.2f}")

plt.figure()
plt.plot(depths, raw_maes, marker="o", label="raw")
plt.plot(depths, calibrated_maes, marker="o", label="calibrated")
plt.xlabel("depth")
plt.ylabel("LOSO MAE (years)")
plt.title("Calibrated vs raw methylation predictions")
plt.legend()
plt.savefig("figures/calibration.png", dpi=120)