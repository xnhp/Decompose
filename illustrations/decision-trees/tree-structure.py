from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn import tree

from decompose.dvc_utils import cwd_path
from illustrations.plot_utils import put_caption

# from https://scikit-learn.org/stable/modules/tree.html
iris = load_iris()
X, y = iris.data, iris.target
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

plt.figure(figsize=(12, 16))
tree.plot_tree(clf)

plt.tight_layout()

plot_id = "tree-structure"

plt.savefig(cwd_path(f"{plot_id}.png"))

caption = """
        Rendering of a decision tree structure. Each inner node corresponds to a partitioning of the examples 
        of the parent edge. 
        TODO: attribute source; maybe have colours match the decision boundaries in other plot.
    """
put_caption(caption,
            cwd_path(f"{plot_id}.tex"))
