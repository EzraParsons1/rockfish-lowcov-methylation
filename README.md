# Rockfish low-coverage methylation clock — simulated-data methods work

Methods investigation toward a cross-species DNA-methylation aging clock in rockfish (*Sebastes*), focused on the **low-coverage** regime. **All results are on simulated data** — no real WGBS or any identifying data is in this repo.

## Reports
1. **Low-coverage error, estimators, and model comparison** — [`reports/01-low-coverage-report.pdf`](reports/01-low-coverage-report.pdf). The elastic-net clock regresses toward the mean at low coverage; Beta-Binomial shrinkage (v3b) cuts low-coverage estimation error ~3× at no high-coverage cost; depth beats breadth (better to read fewer sites deeper).
2. **Block discovery and the correlation it requires** — [`reports/02-block-discovery-correlation.md`](reports/02-block-discovery-correlation.md). *Main finding:* co-methylation block discovery reaches F1 > 0.9 once the **within-region minus between-region CpG correlation gap exceeds ≈ 0.5** — invariant to sequencing depth and block size, and scaling with cohort size (≈0.68 at 45 fish, ≈0.5 at 90, ≈0.42 at 180). Also: post-hoc calibration of the low-coverage bias is a no-op, and a binomial-MLE clock captures more signal but loses to the elastic net on absolute error (bias-variance).

## Why it matters
The correlation-gap threshold is **measurable in real WGBS**, so it indicates *before* building a clock whether read-pooling via discovered co-methylation blocks can work on our data.

## Code
- `simulate.py` — multi-species methylation read-count simulator with co-methylation blocks and known ground truth
- `estimators.py` — Beta-Binomial methylation shrinkage estimators
- `clock.py` — elastic-net clock, calibration, read-pooling
- `binomial_MLE.py` — likelihood-based clock
- `grouping.py` — block discovery, pair-based scoring, within/between correlation stats

Figure scripts: `benchmark.py`, `crossover.py`, `breadth_depth.py` + `heatmap.py`, `error_vs_age.py`, `calibration.py`, `collapse.py` (→ `collapse.png` + `discovery_f1_map.png`), `collapse_robust.py`. Figures in `figures/`.

## Run
```bash
conda env create -f environment.yml
conda activate rockfish-lowcov
python collapse.py          # main discovery / collapse figures
python collapse_robust.py   # robustness across depth, block size, cohort size
```
