import numpy as np
from matplotlib import pyplot as plt

from decompose.dvc_utils import cwd_path
from illustrations.plot_utils import plot_1d_fn_mini


def hinge_loss(x):
    """Hinge loss"""
    return np.maximum(0, 1-x)

def hinge_loss_grad(x):
    """Subgradient"""
    return -1 if x < 1 else 0


if __name__ == "__main__":

    plot_1d_fn_mini((-0.2, 2), hinge_loss, xlabel="margin", ylabel="loss", yrange=(-0.2, 1), xticks=[0, 1], yticks=[0])
    plt.title("Hinge loss")
    plt.tight_layout()
    plt.savefig(cwd_path("hinge-loss.png"))

    plt.figure(figsize=(6,6))
    plot_1d_fn_mini((-0.2, 2), hinge_loss_grad, xlabel="margin", ylabel="loss", yrange=(-1.2, 1), xticks=[0, 1], yticks=[0])
    plt.title("Hinge loss subgradient")
    plt.tight_layout()
    plt.savefig(cwd_path("hinge-loss-grad.png"))


