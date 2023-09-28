import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# TODO this doesnt do what I want yet

# Generate synthetic dataset
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# Define the number of trees in the ensemble
n_estimators = 100

# Train a Random Forest Regressor with varying number of trees
estimators_range = range(1, n_estimators + 1)
ensemble_errors = []
individual_errors = []

for n_trees in estimators_range:
    rf = RandomForestRegressor(n_estimators=n_trees, random_state=42)
    rf.fit(X, y)
    y_pred = rf.predict(X)

    # Calculate the ensemble error (MSE)
    ensemble_error = mean_squared_error(y, y_pred)
    ensemble_errors.append(ensemble_error)

    # Calculate the individual tree errors (MSE)
    tree_errors = [mean_squared_error(y, tree.predict(X)) for tree in rf.estimators_]
    individual_errors.append(np.mean(tree_errors))

# Calculate the ambiguity decomposition
ambiguity_decomposition = np.array(ensemble_errors) - np.array(individual_errors)

# Create a line plot of ensemble error and ambiguity decomposition
plt.figure(figsize=(10, 6))
plt.plot(estimators_range, ensemble_errors, label='Ensemble Error', color='black', linewidth=2)
plt.fill_between(estimators_range, 0, ambiguity_decomposition, alpha=0.5, label='Ambiguity Decomposition', color='blue')

plt.xlabel('Number of Trees in Ensemble')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend()
plt.title('Ensemble Error and Ambiguity Decomposition in Random Forest')
plt.grid(True)

plt.show()
