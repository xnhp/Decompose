import math

import matplotlib.pyplot as plt
import matplotx
import numpy as np

from decompose.classifiers import lerp_b, transform_weights, y_drf_xu_chen
from decompose.dvc_utils import cwd_path
from illustrations.plot_utils import put_caption

x = np.linspace(0, 1, 100)

plt.style.use(matplotx.styles.dufte)

scale = 2
plt.figure(figsize=(scale*4, scale*3))


def y_drf(x):
    return x



dynamic_threshold = 1- 1 / 8  # assuming this is computed based on margins

ms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25]
for i, m in enumerate(ms):
    b = lerp_b(m)
    # x = 1/2 * x
    # shift s.t. inflection point is at thresh
    ys = transform_weights(1*(x - dynamic_threshold), b)
    ys = np.clip(ys, 0, 1/2)
    plt.plot(x, ys, color="orange", alpha=min(i / len(ms) + 0.1, 1) )

# plt.plot(x, y_drf(x), label="DRF weights")
# plt.plot(x, [y_drf_xu_chen(x) for x in x], label="Xu/Chen weights")

plt.axvline(dynamic_threshold, color="red", label="Majority vote threshold", alpha=0.6)

plt.ylabel("Weight")
plt.xlabel("$1/M \\sum_{i=1}^{M}L_{01}(q_i, y)$")

plt.xlim(0,1)

alpha = 0.2
ax = plt.gca()
plt.axvspan(0, dynamic_threshold, color="green", alpha=alpha, label="$X_+$ (ensemble correct)", ymin=-0.025, ymax=0.025)
plt.axvspan(dynamic_threshold, 1, color="orange", alpha=alpha, label="$X_-$ (ensemble incorrect)", ymin=-0.025, ymax=0.025)

plt.axhline(1/2, color="gray", alpha = 0.2)

plt.xticks([0, 1/2, 1])
plt.yticks([0,1/2, 1])

# plt.legend()
plt.tight_layout()

plot_id = "sigmoid-fns-k"
plt.savefig(cwd_path(f"{plot_id}.png"))

put_caption(
    """
    Weighting functions for the original DRF \cite{DRF} weighting scheme and the variation proposed by \cite{XuChen}.
    """
    , cwd_path(f"{plot_id}.tex")
)


plt.show()