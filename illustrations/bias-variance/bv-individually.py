import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from illustrations.plot_utils import variance_plot

# Generate synthetic data
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# Number of decision trees to train
num_trees = 30

# Fraction of the dataset to use for each tree (adjust as needed)
subset_fraction = 0.8

# Lists to store test errors for each tree
dt_test_errors = []
rf_test_errors = []

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train multiple decision trees on random subsets of the data
for _ in range(num_trees):
    # Randomly sample a subset of the training data
    num_samples = int(subset_fraction * X_train.shape[0])
    random_indices = np.random.choice(X_train.shape[0], num_samples, replace=False)
    X_subset = X_train[random_indices]
    y_subset = y_train[random_indices]

    # Create and train a decision tree
    decision_tree_model = DecisionTreeRegressor(random_state=42)
    decision_tree_model.fit(X_subset, y_subset)

    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_subset, y_subset)

    # Evaluate the tree on the test set and calculate test error
    dt_test_errors.append(
        mean_squared_error(y_test, decision_tree_model.predict(X_test))
    )

    rf_test_errors.append(
        mean_squared_error(y_test, rf_model.predict(X_test))
    )



# # Create a line plot of test errors for each tree
# scale = 1
# plt.figure(figsize=(scale*3, scale*6))
# plt.axhline(y=np.mean(dt_test_errors), color='blue', linestyle='-', label='Average Test Error')
# plt.scatter([1 for e in dt_test_errors], dt_test_errors, marker='o', linestyle='-', color='blue', alpha=0.3, s=70)
# plt.axhline(y=np.mean(rf_test_errors), color='orange', linestyle='-', label='Average Test Error')
# plt.scatter([2 for e in rf_test_errors], rf_test_errors, marker='o', linestyle='-', color='orange', alpha=0.3, s=70)
# plt.xlim(0, 3)
# plt.xlabel('Models')
# plt.ylabel('Test Error')
# # plt.grid(True)
# plt.yticks([])
# plt.xticks([])
# plt.tight_layout()
# plt.savefig("test_error_of_individual_trees.png")
# plt.show()

variance_plot(
    [dt_test_errors, rf_test_errors],
    ["Decision Tree", "Random Forest"],
    ["blue", "orange"],
    [np.mean(dt_test_errors), np.mean(rf_test_errors)],
    "Models",
    "Test Error",
    "test_error_of_individual_trees.png"
)
