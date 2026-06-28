import numpy as np
import matplotlib.pyplot as plt
from simulate import simulate
from grouping import block_corr_stats, discover_blocks, score_discovery

configs = [
      {"coverage": 15, "block_size": 5,  "n_specimens": 90,  "label": "baseline"},
      {"coverage": 5,  "block_size": 5,  "n_specimens": 90,  "label": "5X"},
      {"coverage": 30, "block_size": 5,  "n_specimens": 90,  "label": "30X"},
      {"coverage": 15, "block_size": 3,  "n_specimens": 90,  "label": "block_size 3"},
      {"coverage": 15, "block_size": 10, "n_specimens": 90,  "label": "block_size 10"},
      {"coverage": 15, "block_size": 5,  "n_specimens": 45,  "label": "n=45"},
      {"coverage": 15, "block_size": 5,  "n_specimens": 180, "label": "n=180"},
  ]
block_sds  = [0.05, 0.08, 0.11, 0.14, 0.17, 0.20, 0.25, 0.30]
thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
n_seeds = 3

results = {}
for cfg in configs:
        contrasts, f1s = [], []
        for bsd in block_sds:
            seed_c, seed_f1 = [], []
            for s in range(n_seeds):
                data = simulate(seed=s, block_sd=bsd, frac_grouped=0.5,
                              coverage=cfg["coverage"], block_size=cfg["block_size"],
                              n_specimens=cfg["n_specimens"])
                w, b = block_corr_stats(data)
                true = data["block_id"]
                best_f1, best_t = 0.0, thresholds[0]
                for t in thresholds:
                        f1 = score_discovery(true, discover_blocks(data, t))[2]
                        if f1 > best_f1:
                            best_f1, best_t = f1, t                        # your threshold scan, max F1 (unchanged)
                seed_c.append(w - b); seed_f1.append(best_f1)
            contrasts.append(np.mean(seed_c)); f1s.append(np.mean(seed_f1))
        results[cfg["label"]] = (contrasts, f1s)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
for lab in ["baseline", "5X", "30X", "block_size 3", "block_size 10"]:
    c, f = results[lab]; axL.plot(c, f, 'o-', label=lab)
for lab in ["n=45", "baseline", "n=180"]:
      c, f = results[lab]; axR.plot(c, f, 'o-', label=lab)

for ax in (axL, axR):
    ax.axvline(0.5, ls='--', color='gray'); ax.axhline(0.9, ls=':', color='gray')
    ax.set_xlabel('within − between correlation'); ax.legend()
axL.set_ylabel('best F1')
axL.set_title('invariant to depth & block size')
axR.set_title('shifts with cohort size (n)')
plt.tight_layout(); plt.savefig('figures/collapse_robust.png', dpi=150)

