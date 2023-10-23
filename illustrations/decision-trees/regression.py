import matplotx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_blobs

from decompose.dvc_utils import cwd_path
from illustrations.plot_utils import put_caption

# I rather want I rather want: a; regression; decision; tree; on; a small, synthetically generated 2 - dimensional
# dataset that is sampled from a normal distribution.Then, it should plot the normal distribution as a
# heatmap.Further, in a separate plot, it should show the decision boundaries of the constructed tree, as well as the
# data points used for constructing this tree.The data points should be coloured according to their target value.


# from https://scikit-learn.org/stable/modules/generated/sklearn.inspection.DecisionBoundaryDisplay.html


# Generate a synthetic dataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.tree import DecisionTreeClassifier


plt.figure(figsize=(3, 4))

iris = load_iris()
feature_1, feature_2 = np.meshgrid(
    np.linspace(iris.data[:, 0].min(), iris.data[:, 0].max()),
    np.linspace(iris.data[:, 1].min(), iris.data[:, 1].max())
)
grid = np.vstack([feature_1.ravel(), feature_2.ravel()]).T
tree = DecisionTreeClassifier().fit(iris.data[:, :2], iris.target)
y_pred = np.reshape(tree.predict(grid), feature_1.shape)
display = DecisionBoundaryDisplay(
    xx0=feature_1, xx1=feature_2, response=y_pred
)
display.plot()
display.ax_.scatter(
    iris.data[:, 0], iris.data[:, 1], c=iris.target, edgecolor="black"
)

plt.xticks([])
plt.yticks([])

plt.style.use(matplotx.styles.dufte)

plt.tight_layout()
plt.show()

plot_id = "decision-tree-boundaries"

display.figure_.savefig(cwd_path(f"{plot_id}.png"))

caption = """
    Decision boundaries of a tree in two (arbitrary) features constructed on the \\textit{iris} dataset.
     Visualisation based on \\cite{_SklearnInspectionDecisionBoundaryDisplay_}.
    """
put_caption(caption,
            cwd_path(f"{plot_id}.tex"))
