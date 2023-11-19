import math

import matplotlib.pyplot as plt
import matplotx
import numpy as np

from decompose.classifiers import lerp_b, transform_weights, y_drf_xu_chen
from decompose.dvc_utils import cwd_path
from illustrations.plot_utils import put_caption

x = np.linspace(0, 1, 1000)

plt.style.use(matplotx.styles.dufte)

scale = 2
plt.figure(figsize=(scale*4, scale*3))


def y_drf(x):
    return x



# ms = [-5, 0, 1, 5, 10, 15, 20, 25]
# for i, m in enumerate(ms):
#     b = lerp_b(m)
#     ys = transform_weights(x, b)
#     plt.plot(x, ys, color="orange", alpha=i / len(ms) + 0.1 )

plt.plot(x, y_drf(x), label="DRF weights")
plt.plot(x, [y_drf_xu_chen(x) for x in x], label="Xu/Chen weights")

plt.axvline(1/2, color="red", alpha=0.6)

plt.ylabel("Weight")
plt.xlabel("$1/M \\sum_{i=1}^{M}L_{01}(q_i, y)$")

plt.xlim(0,1)

alpha = 0.2
ax = plt.gca()
plt.axvspan(0, 1/2, color="green", alpha=alpha,  ymin=-0.025, ymax=0.025)
plt.axvspan(1/2, 1, color="orange", alpha=alpha, ymin=-0.025, ymax=0.025)

plt.axhline(1/2, color="gray", alpha = 0.2)

plt.xticks([0, 1/2, 1])
plt.yticks([0,1/2, 1])

plt.legend()
plt.tight_layout()

plot_id = "drf-fns"
plt.savefig(cwd_path(f"{plot_id}.png"))

put_caption(
    """
    Weighting functions for the original DRF \cite{DRF} weighting scheme and the variation proposed by \cite{XuChen}.
    """
    , cwd_path(f"{plot_id}.tex")
)


plt.show()