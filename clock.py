import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import LeaveOneGroupOut

def naive_features(data):
    with np.errstate(invalid="ignore", divide="ignore"):
        prop = data["m"] / data["n"]
    site_mean = np.nanmean(prop, axis=0)
    X = np.where(np.isnan(prop), site_mean, prop)
    return np.where(np.isnan(X), 0.5, X)

def pred_rel(X, data):
    clock = make_pipeline(
        StandardScaler(),
        ElasticNetCV(l1_ratio=0.5, cv=3, max_iter=5000, n_jobs=-1),
    )
    preds = cross_val_predict(
        clock, X, data["relative_age"],
        cv=LeaveOneGroupOut(), groups=data["species"],
    )
    preds = np.clip(preds, 0, 1)
    return preds

def group_reads(data):
    m = data["m"]
    n = data["n"]
    block_id = data["block_id"]
    n_specimens = m.shape[0]
    n_blocks = block_id.max() + 1

    grouped_m = np.zeros((n_specimens, n_blocks))
    grouped_n = np.zeros((n_specimens, n_blocks))
    for b in range(n_blocks):
        cols = block_id == b
        grouped_m[:, b] = m[:, cols].sum(axis=1)
        grouped_n[:, b] = n[:, cols].sum(axis=1)
    
    return {
        "m": grouped_m,
        "n": grouped_n,
        "species": data["species"],
        "relative_age": data["relative_age"],
        "age": data["age"],
        "lifespans": data["lifespans"]
    }

def mae_years(pred, data):
    pred_years = pred * data["lifespans"][data["species"]]
    return np.mean(np.abs(pred_years - data["age"]))

def calibrate(pred, data):
    species = data["species"]
    true_age = data["relative_age"]
    cal = np.zeros_like(pred)
    slopes = []

    for s in np.unique(species):
        train = species != s
        test = species == s

        pc = pred[train].astype(float).copy()
        tc = true_age[train].astype(float).copy()
        sp_train = species[train]

        for t in np.unique(sp_train):
            m = sp_train == t
            pc[m] -= pc[m].mean()
            tc[m] -= tc[m].mean()

        c1, c0 = np.polyfit(pc, tc, 1)
        center = pred[test].mean()
        cal[test] = (pred[test] - center) * c1 + center
        slopes.append(c1)
    return np.clip(cal, 0, 1), np.mean(slopes)

def loso_mae_years(X, data):
    return mae_years(pred_rel(X, data), data)