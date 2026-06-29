import numpy as np
from simulate import simulate
from estimators import estimate
from clock import pred_rel, mae_years


def mle_pred_rel(data, top_k):
    ref_meth = estimate(data["m"], data["n"] - data["m"], "v3b")
    n_sites = data["m"].shape[1]
    n_specimens = data["m"].shape[0]
    preds = np.zeros(n_specimens)
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
            corrs[i] = np.corrcoef(ref_meth[train][:, i], train_ages)[0, 1]
        corrs = np.nan_to_num(corrs)
        top = np.argsort(np.abs(corrs))[-top_k:]
        for s in np.where(test)[0]:
            likelihoods = []
            for age in ages:
                sum_a = 0.0
                for i in top:
                    m_i = data["m"][s][i]
                    n_i = data["n"][s][i]
                    p_i = np.clip(slopes[i] * age + intercepts[i], 1e-6, 1 - 1e-6)
                    sum_a += m_i * np.log(p_i) + (n_i - m_i) * np.log(1 - p_i)
                likelihoods.append(sum_a)
            preds[s] = ages[np.argmax(likelihoods)]
    return preds


if __name__ == "__main__":
    # Bias-variance: binomial MLE vs elastic net (v3b features), low vs high coverage.
    # LOSO predictions pooled over seeds; slope = polyfit(true, pred), 1 = no shrinkage.
    n_seeds, top_k = 5, 100
    for cov in [3, 30]:
        true_all, en_all, mle_all, en_mae, mle_mae = [], [], [], [], []
        for s in range(n_seeds):
            data = simulate(seed=s, coverage=cov)
            true = data["relative_age"]
            X = estimate(data["m"], data["n"] - data["m"], "v3b")
            en = pred_rel(X, data)
            mle = mle_pred_rel(data, top_k)
            true_all.append(true)
            en_all.append(en)
            mle_all.append(mle)
            en_mae.append(mae_years(en, data))
            mle_mae.append(mae_years(mle, data))
        true_all = np.concatenate(true_all)
        en_all = np.concatenate(en_all)
        mle_all = np.concatenate(mle_all)
        print(f"=== coverage {cov}X (pooled over {n_seeds} seeds) ===")
        for name, pred in [("EN ", en_all), ("MLE", mle_all)]:
            r = np.corrcoef(true_all, pred)[0, 1]
            slope = np.polyfit(true_all, pred, 1)[0]
            err_std = (pred - true_all).std()
            print(f"  {name}  r {r:.2f}  slope {slope:.2f}  err_std {err_std:.3f}")
        print(f"  years-MAE:  EN {np.mean(en_mae):.2f}  MLE {np.mean(mle_mae):.2f}")
