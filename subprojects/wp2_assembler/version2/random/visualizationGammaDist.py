from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma

x = np.linspace(0, 20, 1000)
# Varying shape (fixed scale=1)
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
for k in [0.5, 1, 2, 5, 9]:
    plt.plot(x, gamma.pdf(x, a=k, scale=1), label=f"k={k}, θ=1")
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.title("Effect of Shape Parameter (k)")
plt.legend()
plt.grid(True, alpha=0.3)

# Varying scale (fixed shape=2)
plt.subplot(1, 2, 2)
for theta in [0.5, 1, 2, 3]:
    plt.plot(x, gamma.pdf(x, a=2, scale=theta), label=f"k=2, θ={theta}")
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.title("Effect of Scale Parameter (θ)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

# Save plot instead of showing (cluster-friendly)
output_dir = Path(__file__).parent.parent / "data" / "imagePlots"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "gamma_distribution_comparison.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Plot saved as: {output_path}")
plt.close()  # Clean up memory
