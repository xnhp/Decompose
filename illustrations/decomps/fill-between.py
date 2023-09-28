import matplotlib.pyplot as plt
import numpy as np

def plot_stacked_functions(function1, function2):
    # Define x values
    x = np.linspace(0, 10, 100)

    # Calculate the values of the two functions and their sum
    y1 = function1(x)
    y2 = function2(x)
    y_sum = y1 + y2

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the area under the sum curve for each component function
    ax.fill_between(x, 0, y1, label='Function 1', alpha=0.5, color='blue')
    ax.fill_between(x, y1, y_sum, label='Function 2', alpha=0.5, color='red')

    # Plot the sum as a line
    ax.plot(x, y_sum, label='Sum', color='black', linewidth=2)

    # Add labels and a legend
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.legend()

    # Show the plot
    plt.show()

# Define two functions (you can replace these with your own functions)
def function1(x):
    return np.sin(x)

def function2(x):
    return np.cos(x)

# Call the function to plot the stacked functions
plot_stacked_functions(function1, function2)
