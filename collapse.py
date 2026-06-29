import numpy as np
import matplotlib.pyplot as plt
from simulate import simulate
from grouping import block_corr_stats, discover_blocks, score_discovery

block_sds  = [0.05, 0.08, 0.11, 0.14, 0.17, 0.20, 0.25, 0.30]
depths  = [5, 15, 30]
thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
n_seeds, block_size, frac_grouped = 3, 5, 0.5

f1_map = np.zeros((len(block_sds), len(depths)))
t_map  = np.zeros((len(block_sds), len(depths)))
within_map  = np.zeros((len(block_sds), len(depths)))
between_map = np.zeros((len(block_sds), len(depths)))

for bi, bsd in enumerate(block_sds):
    for di, depth in enumerate(depths):
        seed_f1, seed_t, seed_w, seed_b = [], [], [], []
        for s in range(n_seeds):
            data = simulate(seed=s, coverage=depth, block_sd=bsd, block_size=block_size, frac_grouped=frac_grouped)
            w, b = block_corr_stats(data)
            seed_w.append(w)
            seed_b.append(b)
            true = data["block_id"]
            best_f1, best_t = 0.0, thresholds[0]
            for t in thresholds:
                f1 = score_discovery(true, discover_blocks(data, t))[2]
                if f1 > best_f1:
                    best_f1, best_t = f1, t
            seed_f1.append(best_f1)
            seed_t.append(best_t)
        f1_map[bi, di] = np.mean(seed_f1)
        t_map[bi, di]  = np.mean(seed_t)
        within_map[bi, di]  = np.mean(seed_w)
        between_map[bi, di]  = np.mean(seed_b)
        print(f"block_sd {bsd:.2f}  depth {depth:>2}:  F1 {f1_map[bi, di]:.3f}  best_t {t_map[bi, di]:.2f} within {within_map[bi, di]:.2f} between {between_map[bi, di]:.2f}")

# --- collapse figure: F1 vs raw within (drifts) vs contrast (collapses) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
colors = ['tab:blue', 'tab:orange', 'tab:green']
for di, depth in enumerate(depths):
    ax1.plot(within_map[:, di], f1_map[:, di], 'o-', color=colors[di], label=f'{depth}X')
    ax2.plot(within_map[:, di] - between_map[:, di], f1_map[:, di], 'o-', color=colors[di], label=f'{depth}X')
ax1.set_title('raw within-block corr')
ax1.set_xlabel('within-block correlation'); ax1.set_ylabel('best F1')
ax2.set_title('within − between (contrast)')
ax2.set_xlabel('within − between correlation')
ax2.axvline(0.5, ls='--', color='gray'); ax2.axhline(0.9, ls=':', color='gray')
ax1.legend(); ax2.legend()
plt.tight_layout(); plt.savefig('figures/collapse.png', dpi=150)

# --- discovery F1 heatmap over block_sd x depth ---
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(f1_map, origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=1)
ax.set_xticks(range(len(depths)));    ax.set_xticklabels(depths)
ax.set_yticks(range(len(block_sds))); ax.set_yticklabels(block_sds)
ax.set_xlabel('depth (X)'); ax.set_ylabel('block_sd')
ax.set_title('block discovery: best F1')
fig.colorbar(im, label='best F1')
plt.tight_layout(); plt.savefig('figures/discovery_f1_map.png', dpi=150)