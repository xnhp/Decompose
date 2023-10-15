import matplotx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Generate synthetic data
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a range of maximum tree depths to explore
max_depths = [1, 2, 3, 4]

biases = []
variances = []

# Train decision trees with different depths and estimate bias-variance
for depth in max_depths:
    model = DecisionTreeRegressor(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate bias (squared) as the mean squared difference between true and predicted values
    bias = np.mean((y_train - y_train_pred) ** 2)

    # Calculate variance as the mean squared difference between predicted values and their mean
    # TODO this is BS, this is not the variance we mean
    variance = np.mean((y_train_pred - np.mean(y_train_pred)) ** 2)

    biases.append(bias)
    variances.append(variance)

# Plot the bias-variance tradeoff
sz = 6
plt.style.use(matplotx.styles.dufte)
plt.figure(figsize=(sz, sz))
plt.plot(max_depths, biases, marker='o', label='Bias$^2$')
plt.plot(max_depths, variances, marker='o', label='Variance')
plt.xlabel('Maximum Tree Depth')
plt.ylabel('Bias and Variance')
plt.title('Bias-Variance Decomposition with Decision Trees')
plt.legend()
plt.grid(True)
plt.savefig("bias_variance_decomposition_with_decision_trees.png")
plt.show()