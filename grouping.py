import numpy as np
from scipy.sparse.csgraph import connected_components
from estimators import estimate

def discover_blocks(data, threshold):
    meth = estimate(data["m"], data["n"] - data["m"], "v3b")
    C = np.corrcoef(meth, rowvar=False)
    adj = C > threshold

    n_comp, labels = connected_components(adj, directed=False)
    return labels

def pair_count(sizes):
    return np.sum(sizes * (sizes - 1) // 2)

def within_block_corr(data):
    meth = estimate(data["m"], data["n"] - data["m"], "v3b")
    C = np.corrcoef(meth, rowvar=False)
    bid = data["block_id"]
    vals = []
    for b in np.unique(bid):
        cols = np.where(bid == b)[0]
        if len(cols) < 2:
            continue
        sub = C[np.ix_(cols, cols)]
        iu = np.triu_indices(len(cols), 1)
        vals.extend(sub[iu])
    vals = np.nan_to_num(np.array(vals), nan=0.0)
    return vals.mean()

def score_discovery(true, pred):
    true_pairs = pair_count(np.bincount(true))
    pred_pairs = pair_count(np.bincount(pred))

    tp = 0
    for b in np.unique(true):
        members = pred[true == b]
        tp += pair_count(np.bincount(members))
    
    precision = tp / pred_pairs if pred_pairs else 0.0
    recall = tp / true_pairs if true_pairs else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision+recall) else 0.0
    return precision, recall, f1

def block_corr_stats(data):
      meth = estimate(data["m"], data["n"] - data["m"], "v3b")
      C = np.corrcoef(meth, rowvar=False)
      bid = data["block_id"]
      n_sites = len(bid)
      iu = np.triu_indices(n_sites, 1)
      ci, cj = iu
      corrs = np.nan_to_num(C[iu], nan=0.0)
      same = bid[ci] == bid[cj]
      within = corrs[same].mean()
      between = corrs[~same].mean()
      return within, between



