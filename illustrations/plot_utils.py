import matplotx
import numpy as np
from matplotlib import pyplot as plt

def put_caption(text, filename):
    with open(filename, "w") as f:
        s = "\\caption{" + text + "}"
        f.write(s)

def variance_plot(series, labels, colors, means, xlabel, ylabel, filename, plot_mean=False):
    plt.style.use(matplotx.styles.dufte)
    plt.figure(figsize=(3,6))
    glyph_args = {
        "alpha": 0.3,
        "s": 70
    }
    for i, e  in enumerate(zip(series, labels, colors, means)):
        points, label, color, mean = e
        plt.scatter([i+1 for e in points], points, color=color, label=label, **glyph_args)
        plt.axhline(y=mean, color=color, linestyle='-', label=label)
        if plot_mean:
            plt.axhline(y=np.mean(points), color=color, linestyle='--')

    plt.xlim(0, len(series)+1)
    plt.yticks([])
    plt.xticks([])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def plot_1d_fn_mini(xrange, fn, yrange=None, xlabel=None, ylabel=None, xticks=None, yticks=None,
                    figsize=(6,6)):
    plt.figure(figsize)
    plt.style.use(matplotx.styles.dufte)
    xs = np.linspace(*xrange, 100)

    fs = [fn(x) for x in xs]

    if yrange is not None:
        plt.ylim(*yrange)

    if xticks is not None:
        plt.xticks(xticks)

    if yticks is not None:
        plt.yticks(yticks)

    matplotx.line_labels()  # line labels to the right
    plt.plot(xs, fs, label=fn.__doc__)
