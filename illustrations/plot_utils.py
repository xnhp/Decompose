import numpy as np
from matplotlib import pyplot as plt

def put_caption(text, filename):
    with open(filename, "w") as f:
        s = "\\caption{" + text + "}"
        f.write(s)

def variance_plot(series, labels, colors, means, xlabel, ylabel, filename, plot_mean=False):
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
