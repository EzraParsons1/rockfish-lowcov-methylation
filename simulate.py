import numpy as np

def simulate(seed=42, lifespans=None, b_offsets=None, n_specimens=90, n_sites=1000, frac_age_linked=0.1, slope_sd=0.3, noise_sd=0.02, coverage=12):
    rng = np.random.default_rng(seed)

    if lifespans is None:
            lifespans = np.array([14, 30, 100, 200])
    if b_offsets is None:
            b_offsets = np.array([0.001, 0.05, 0.2, 0.1])

    species = rng.integers(low=0, high=len(lifespans), size=n_specimens)
    relative_age = rng.random(size=n_specimens)
    absolute_age = lifespans[species] * relative_age
    mask = rng.random(size=n_sites) < frac_age_linked
    slopes = rng.normal(0, slope_sd, size=n_sites)
    slopes *= mask
    baseline = rng.random(size=n_sites)
    noise = rng.normal(0, noise_sd, size=(n_specimens,n_sites))

    p = baseline + b_offsets[species][:, None] + slopes * relative_age[:, None] + noise
    p = np.clip(p, 0, 1)

    n = rng.poisson(lam=coverage, size=(n_specimens, n_sites))
    m = rng.binomial(n, p)
    return {
            "m": m, "n": n,
            "true_p": p,
            "age": absolute_age,
            "species": species,
            "is_age_linked": mask,
        "slopes": slopes,
        "relative_age": relative_age,
        "lifespans": lifespans
        }






