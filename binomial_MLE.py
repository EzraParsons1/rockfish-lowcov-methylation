import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simulate import simulate
from estimators import estimate
from clock import naive_features
from clock import pred_rel
from clock import loso_mae_years
from clock import mae_years
from clock import group_reads

def mle_pred_rel(data, top_k):
    ref_meth = estimate(data["m"], data["n"] - data["m"], "v3b")
    n_sites = data["m"].shape[1]
    n_specimens = data["m"].shape[0]
    pred_rel = np.zeros(n_specimens)
    ages = np.linspace(0, 1, 101)
    species = data["species"]

    for held in np.unique(species):
        train = species != held
        test = species == held
        train_ages = data["relative_age"][train]
        slopes = np.zeros(n_sites)
        intercepts = np.zeros(n_sites)
        corrs = np.zeros(n_sites)
        for i in range(n_sites):
            slopes[i], intercepts[i] = np.polyfit(train_ages, ref_meth[train][:, i], 1)
            corrs[i] = np.corrcoef(ref_meth[train][:, i], train_ages)[0,1]
        corrs = np.nan_to_num(corrs)
        top = np.argsort(np.abs(corrs))[-top_k:]
        for s in np.where(test)[0]:
            likelyhoods = []
            for age in ages:
                sum_a = 0
                for i in top:
                    m_i = data["m"][s][i]
                    n_i = data["n"][s][i]
                    p_i = np.clip(slopes[i] * age + intercepts[i], 1e-6, 1 - 1e-6)
                    sum_a += m_i * np.log(p_i) + (n_i - m_i) * np.log(1 - p_i)
                likelyhoods.append(sum_a)
            pred_rel[s] = ages[np.argmax(likelyhoods)]
    return pred_rel

depths = [3, 5, 8, 12]
n_seeds = 3
top_k = 100
en_maes, mle_maes, engrp_maes, mlegrp_maes = [], [], [], []

#for d in depths:
#    en_runs, mle_runs = [], []
#    for s in range(n_seeds):
#       data = simulate(seed=s, coverage=d)
#        mle = mae_years(mle_pred_rel(data, top_k), data)
#        X = estimate(data["m"], data["n"] - data["m"], "v3b")
#        en = mae_years(pred_rel(X, data), data)
#        en_runs.append(en)
#        mle_runs.append(mle)
#   en_maes.append(np.mean(en_runs))
#   mle_maes.append(np.mean(mle_runs))
#   print(f"depth {d:>2}:  EN {en_maes[-1]:5.2f}   MLE {mle_maes[-1]:5.2f}")

#for cov in [3, 30]:
#    data = simulate(seed=0, coverage=cov)
#    mle = mle_pred_rel(data, 100)
#    X = estimate(data["m"], data["n"] - data["m"], "v3b")
#    en = pred_rel(X, data)
#    true = data["relative_age"]
#    print(f"--- coverage {cov} ---")
#    for name, pred in [("EN", en), ("MLE", mle)]:
#        err   = pred - true
#        slope = np.polyfit(true, pred, 1)[0]   # 1.0 = no shrinkage, <1 = shrinks to mean
#        print(f"{name:3}  meanerr {err.mean():+.3f}  std {err.std():.3f}  slope {slope:.2f}  r {r:.2f}")

#for cov in [30]:
#    data = simulate(seed=0, coverage=cov)
#    print("MLE", cov, mae_years(mle_pred_rel(data, 100), data))
#   X = estimate(data["m"], data["n"] - data["m"], "v3b")
#   print("EN ", cov, mae_years(pred_rel(X, data), data))

#data = simulate(seed=0, coverage=30)
#mle = mle_pred_rel(data, 100)
#X = estimate(data["m"], data["n"] - data["m"], "v3b")
#en = pred_rel(X, data)
#true, species = data["relative_age"], data["species"]

#for sp in np.unique(species):
#    msk = species == sp
#    L = data["lifespans"][sp]
#    for name, pred in [("MLE", mle), ("EN", en)]:
#        err = pred[msk] - true[msk]
#        print(f"sp{sp} L={L:>3} {name}: yrsMAE {np.mean(np.abs(err))*L:6.2f}  "
#             f"relMAE {np.mean(np.abs(err)):.3f}  meanerr {err.mean():+.3f}")

for d in depths:
    en_runs, mle_runs, engrp_runs, mlegrp_runs = [], [], [], []
    for s in range(n_seeds):
        data = simulate(seed=s, coverage=d, block_size=5, frac_grouped=1.0)
        gdata = group_reads(data)

        mle = mae_years(mle_pred_rel(data,  top_k), data)
        mle_grp = mae_years(mle_pred_rel(gdata, top_k), gdata)

        X = estimate(data["m"],  data["n"]  - data["m"],  "v3b")
        en = mae_years(pred_rel(X, data), data)
        Xg = estimate(gdata["m"], gdata["n"] - gdata["m"], "v3b")
        en_grp = mae_years(pred_rel(Xg, gdata), gdata)

        en_runs.append(en)
        mle_runs.append(mle)
        engrp_runs.append(en_grp)
        mlegrp_runs.append(mle_grp)
        
    en_maes.append(np.mean(en_runs))
    mle_maes.append(np.mean(mle_runs))
    engrp_maes.append(np.mean(engrp_runs))
    mlegrp_maes.append(np.mean(mlegrp_runs))
    print(f"depth {d:>2}:  EN {en_maes[-1]:5.2f} EN_GRP {engrp_maes[-1]:5.2f}  MLE {mle_maes[-1]:5.2f} MLE_GRP {mlegrp_maes[-1]:5.2f}")






