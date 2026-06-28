import matplotlib.pyplot as plt
import numpy as np

fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
depths = [3, 5, 8, 12, 15, 30]
cube = np.load("cube.npy")
MAEs = np.median(cube, axis=2)

plt.figure()
plt.imshow(MAEs, aspect="auto", cmap="viridis_r")
plt.xticks(range(len(fractions)), fractions)
plt.yticks(range(len(depths)), depths)
plt.xlabel("fraction of sites observed")
plt.ylabel("average depth")
plt.colorbar(label="LOSO MAE (years)")
for di in range(MAEs.shape[0]):
    for fi in range(MAEs.shape[1]):
        plt.text(fi, di,f"{MAEs[di,fi]:.1f}", ha="center", va="center")
plt.title("Clock error vs. depth and breadth")
plt.savefig("figures/breadth_depth.png", dpi=120)