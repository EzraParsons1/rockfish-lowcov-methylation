import numpy as np

def simulate(seed=42, lifespans=None, b_offsets=None, n_specimens=90, n_sites=1000, frac_age_linked=0.1, slope_sd=0.3, noise_sd=0.02, coverage=12, block_size=5, block_sd=0.0, frac_grouped=0.0):
        rng = np.random.default_rng(seed)

        if lifespans is None:
                lifespans = np.array([14, 30, 100, 200])
        if b_offsets is None:
                b_offsets = np.array([0.001, 0.05, 0.2, 0.1])

        species = rng.integers(low=0, high=len(lifespans), size=n_specimens)
        relative_age = rng.random(size=n_specimens)
        absolute_age = lifespans[species] * relative_age

        target = int(n_sites * frac_grouped)

        block_sizes = []
        total = 0

        while total < target:
                s = max(2, rng.poisson(block_size))
                if total + s > n_sites:
                        break
                block_sizes.append(s)
                total += s

        n_single = n_sites - total
        sizes = np.array(block_sizes + [1] * n_single)
        n_blocks = len(sizes)

        block_mask = rng.random(size=n_blocks) < frac_age_linked
        block_slopes = rng.normal(0, slope_sd, size=n_blocks) * block_mask
        block_baseline = rng.random(size=n_blocks)

        mask = np.repeat(block_mask, sizes)
        slopes = np.repeat(block_slopes, sizes)
        baseline = np.repeat(block_baseline, sizes)
        block_id = np.repeat(np.arange(n_blocks), sizes)

        noise = rng.normal(0, noise_sd, size=(n_specimens,n_sites))

        block_effect = rng.normal(0, block_sd, size=(n_specimens, n_blocks))

        p = baseline + b_offsets[species][:, None] + slopes * relative_age[:, None] + block_effect[:, block_id] + noise
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
                "lifespans": lifespans,
               "block_id": block_id
                }






