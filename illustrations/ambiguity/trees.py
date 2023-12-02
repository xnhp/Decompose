import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from decompose.dvc_utils import get_model_color, get_fn_color
from illustrations.plot_utils import variance_plot, put_caption

# Generate synthetic data
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# Number of decision trees in the Random Forest
n_estimators = 100

# Fraction of the dataset to use for each tree (adjust as needed)
subset_fraction = 0.8

# Lists to store the test errors of individual trees
individual_tree_errors = []

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest ensemble
model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set using the ensemble
y_pred_ensemble = model.predict(X_test)

# Calculate the mean squared error for the ensemble
ensemble_error = mean_squared_error(y_test, y_pred_ensemble)

# Calculate the ambiguity (spread) of the individual tree errors around the ensemble error
for tree in model.estimators_:
    # Make predictions on the test set using an individual tree
    y_pred_tree = tree.predict(X_test)

    # Calculate the mean squared error for the individual tree
    tree_error = mean_squared_error(y_test, y_pred_tree)
    individual_tree_errors.append(tree_error)

# Calculate the ambiguity (spread) of the individual tree errors around the ensemble error
ambiguity = np.var(individual_tree_errors)

plot_id = "rf-ambiguity"

variance_plot(
    [individual_tree_errors],
    [None],
    [get_fn_color("ambiguity")],
    [ensemble_error],
    "",
    "Test Error",
    f"{plot_id}.png",
    plot_mean=True,
)

caption = """The spread of individual tree predictions in a random forest ensemble. 
    Glyphs correspond to test errors of individual trees. 
    The dashed line is the average test error of individual \\tcircle{green} trees 
    $\\frac{1}{M} \\sum_1^M L(y, q_i)$.
    The solid line is the test error of the ensemblL(y, \\bar{q})$. 
    The difference between these values is the \\textit{ensemble improvement} or \\textit{ambiguity-effect}.
    """
put_caption(caption,
         f"{plot_id}.tex")