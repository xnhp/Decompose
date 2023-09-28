import numpy as np
import matplotlib.pyplot as plt

# Define two points in 2D space
point1 = np.array([1, 2])
point2 = np.array([4, 6])

# Calculate the squared Euclidean distance
squared_distance = np.sum((point1 - point2) ** 2)

# Create a grid for plotting
x = np.linspace(0, 5, 100)
y = np.linspace(0, 8, 100)
X, Y = np.meshgrid(x, y)

# Calculate the squared error loss for each point on the grid
squared_error = (X - point1[0]) ** 2 + (Y - point1[1]) ** 2

# Create the contour plot of the squared error loss
plt.figure(figsize=(8, 6))
plt.contour(X, Y, squared_error, levels=[squared_distance], colors='r', linewidths=2)
plt.scatter([point1[0], point2[0]], [point1[1], point2[1]], c='b', marker='o', label='Points')
plt.text(point1[0] - 0.2, point1[1] + 0.2, 'Point 1', fontsize=12)
plt.text(point2[0] - 0.2, point2[1] + 0.2, 'Point 2', fontsize=12)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Squared Error Loss (Euclidean Distance)')
plt.legend()
plt.grid(True)

plt.show()
