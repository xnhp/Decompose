import numpy as np
import matplotlib.pyplot as plt

from decompose.dvc_utils import cwd_path
from illustrations.plot_utils import put_caption


# Define the generator function (squared error)
def squared_error(x):
    return x**2

def squared_error_grad(x):
    return 2*x



def linear_approx_point(x):
    fx = squared_error(x)
    slope = squared_error_grad(0.5)
    step = 1
    return fx + slope * step


# Define the reference point
ref_point = 0.5

# Define the range of x values to plot
xs = np.linspace(-1, 1, 100)

plt.figure(figsize=(3, 4))

# Plot the results
plt.plot(xs, squared_error(xs), label='Squared Error')

linear_approximation = [linear_approx_point(x) for x in xs]

plt.scatter(0.5 + 1, linear_approx_point(0.5), label='Linear Approximation Term')
# plt.plot(xs, linear_approximation, label='Linear Approximation')
# plt.plot(xs, bregman_divergences, label='Bregman Divergence')
plt.legend()
plt.tight_layout()

plot_id = "bregman-intuition-1d"

plt.savefig(cwd_path(f"{plot_id}.png"))

caption = """
    $1$-dimensional illustration of the Bregman divergence with with $\phi(x) = x^2$, yielding the squared error.
    """
put_caption(caption,
         cwd_path(f"{plot_id}.tex"))
