import numpy as np

def estimate(m, u, method):
        n = m + u
        match method:
                case "naive":
                        alpha = beta = 0.0
                case "v1":
                        alpha = beta = 1
                case "v2":
                        mu = m.sum(axis=0) / (m+u).sum(axis=0)
                        s = 20
                        alpha = mu * s
                        beta = (1-mu) * s
                case "v3":
                        mu = m.sum(axis=0) / (m+u).sum(axis=0)
                        with np.errstate(invalid="ignore"):
                                prop = m/(m+u)

                        v = np.maximum(np.nanvar(prop, axis=0), 1e-6)

                        s = (mu * (1-mu))/v - 1
                        s = np.maximum(s, 1e-6)

                        alpha = mu * s
                        beta = (1-mu) * s
                case "v3b":
                        mu = m.sum(axis=0) / (m+u).sum(axis=0)
                        with np.errstate(invalid="ignore"):
                                prop = m / (m + u)

                        with np.errstate(divide='ignore'):
                                inv_n = np.where((m+u) > 0, 1/(m+u), np.nan)

                        mean_inv_n = np.nanmean(inv_n, axis=0)

                        sampling_var = mu * (1 - mu) * mean_inv_n

                        observed_var = np.maximum(np.nanvar(prop, axis=0), 1e-6)

                        bio_var = np.maximum(observed_var - sampling_var, 1e-6)

                        strength = (mu * (1 - mu))/bio_var - 1
                        strength = np.maximum(strength, 1e-6)

                        alpha = mu * strength
                        beta  = (1 - mu) * strength
                case _:
                        raise ValueError(f"unknown method: {method}")
        est = (m + alpha) / (n + alpha + beta)
        return impute_nans(est)

def impute_nans(est):
        site_mean = np.nanmean(est, axis=0)
        return np.where(np.isnan(est), site_mean, est)
