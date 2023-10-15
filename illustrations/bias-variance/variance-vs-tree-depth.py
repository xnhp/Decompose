import matplotx
from matplotlib import pyplot as plt

from decompose.dvc_utils import get_fn_color, cwd_path
from illustrations.plot_utils import plot_1d_fn_mini, put_caption
import numpy as np
from numpy import mean
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

X, y = make_regression(n_samples=1000, n_features=1, noise=10, random_state=42)

# Define a range of maximum tree depths to explore
max_depths = range(1,20)
n_trials = 3


# Train decision trees with different depths and estimate bias-variance
variances = []
for depth in max_depths:

    trial_preds = []

    for trial in range(n_trials):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

        model = DecisionTreeRegressor(max_depth=depth, random_state=42)
        model.fit(X_train, y_train)

        y_test_pred = model.predict(X_test)
        trial_preds.append(y_test_pred)

    mean_pred = mean(trial_preds)
    variances_per_example = [(trial_pred - mean_pred)**2 for trial_pred in trial_preds]
    variance_for_depth = mean(variances_per_example)

    variances.append(variance_for_depth)

plt.style.use(matplotx.styles.dufte)
plt.figure(figsize=(6,6))
plt.plot(max_depths, variances, color=get_fn_color("variance"))
plt.xlabel("Maximum tree depth")
plt.ylabel("Variance")
plt.tight_layout()

plot_id = "variance-vs-tree-depth"
plt.savefig(cwd_path(f"{plot_id}.png"))
put_caption(
    """
    Variances of decision trees of increasing depths. Evaluated for squared-error regression on a synthetic dataset.
    """,
    cwd_path(f"{plot_id}.tex")
)
