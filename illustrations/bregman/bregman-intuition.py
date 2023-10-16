import matplotx
import numpy as np
import matplotlib.pyplot as plt

from decompose.dvc_utils import cwd_path
from illustrations.plot_utils import put_caption


# Define the generator function (squared error)
def squared_error_gen(x):
    return x**2

def squared_error_gen_grad(x):
    return 2*x

def ikura_saito_gen(x):
    return - np.log(x)

def ikura_saito_gen_grad(x):
    return - 1/x

def linear_approx_point(x):
    fx = squared_error_gen(x)
    slope = squared_error_gen_grad(0.5)
    step = 1
    return fx + slope * step


def linear_approx(y, x, fn, grad):
    return fn(x) + (y - x) * grad(x)

def linear_approx_sqerr(y, ref):
    return linear_approx(y, ref, squared_error_gen, squared_error_gen_grad)

def plot_linear_approx_line(maxy, ref, gen, gen_grad):
    p2 = (maxy, linear_approx(maxy, ref, gen, gen_grad))
    p1 = (ref, squared_error_gen(ref))
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]])

    plt.plot([p2[0], maxy], [p2[1], squared_error_gen(maxy)], color="red")
    # plt.text(p2[0] + 0.05, p2[1] + 0.1, "Divergence", color="red")


plt.style.use(matplotx.styles.dufte)

plt.figure(figsize=(3, 4))
xs = np.linspace(0.2, 1, 100)
plt.plot(xs, squared_error_gen(xs), label='Squared Error')
plot_linear_approx_line(1, 0.5, squared_error_gen, squared_error_gen_grad)
plt.tight_layout()
plot_id = "bregman-intuition-1d"
plt.savefig(cwd_path(f"{plot_id}.png"))
caption = """
    $1$-dimensional illustration of the Bregman divergence with with $\phi(x) = x^2$, yielding the squared error.
    """
put_caption(caption,
         cwd_path(f"{plot_id}.tex"))
plt.show()
plt.close()

plt.figure(figsize=(3, 4))
xs = np.linspace(0.01, 2, 100)
plt.plot(xs, ikura_saito_gen(xs), label="Ikura-Saito")
plot_linear_approx_line(0.5, 0.1, ikura_saito_gen, ikura_saito_gen_grad)
plt.tight_layout()
plot_id = "bregman-intuition-1d-ikura-saito"
plt.savefig(cwd_path(f"{plot_id}.png"))
caption = """
    $1$-dimensional illustration of the Bregman divergence with with $\phi(x) = - \log x$, yielding the Ikura-Saito distance.
    """
put_caption(caption,
            cwd_path(f"{plot_id}.tex"))
plt.show()
plt.close()



# # Define the reference point
# ref = 0.5
#
# p1 = (ref, )
#
# # Define the range of x values to plot
# xs = np.linspace(-1, 1, 100)
#
# plt.figure(figsize=(3, 4))
#
# # Plot the results
# plt.plot(xs, squared_error(xs), label='Squared Error')
#
# linear_approximation = [linear_approx_point(x) for x in xs]
#
# plt.scatter(0.5 + 1, linear_approx_point(0.5), label='Linear Approximation Term')
#
# plt.plot([0.5, 1.5], [0.5, linear_approx_point(0.5)])

# plt.plot(xs, linear_approximation, label='Linear Approximation')
# plt.plot(xs, bregman_divergences, label='Bregman Divergence')
# plt.legend()
# plt.tight_layout()
#
# plot_id = "bregman-intuition-1d"
#
# plt.show()
#
# plt.savefig(cwd_path(f"{plot_id}.png"))
#
# caption = """
#     $1$-dimensional illustration of the Bregman divergence with with $\phi(x) = x^2$, yielding the squared error.
#     """
# put_caption(caption,
#          cwd_path(f"{plot_id}.tex"))
