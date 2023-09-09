import math

import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(0, 1, 1000)


# TODO factor out from classfiers.py if we do more work on this

def y_drf(x):
    return 1-x

def y_drf_xu_chen(x):
    if (x < 1/2):  # "incorrect"
        return 1 - x**2
    else:
        return 1 - math.sqrt(x)

plt.plot(x, y_drf(x), label="y_drf")
plt.plot(x, [y_drf_xu_chen(x) for x in x], label="y_drf_xu_chen")

plt.axvline(1/2, color="red", label="majority vote threshold")

plt.ylabel("Weight")
plt.xlabel("$1 - 1/M \\sum_{i=1}^{M}L_{01}(q_i, y)$ \n (Ratio trees correct) ")

plt.xlim(0,1)

alpha = 0.2
plt.axvspan(1/2, 1, color="green", alpha=alpha, label="$X_+$ (ensemble correct)")
plt.axvspan(0, 1/2, color="orange", alpha=alpha, label="$X_-$ (ensemble incorrect)")

plt.axhline(1/2, color="gray")

plt.xticks([0, 1/2, 1])
plt.yticks([0,1/2, 1])

plt.legend()
plt.tight_layout()

plt.savefig("weighting-fns.png")
plt.show()