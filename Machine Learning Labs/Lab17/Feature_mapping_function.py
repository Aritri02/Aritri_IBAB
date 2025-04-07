import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LogisticRegression


def Transform(x1, x2):
    return np.array([ x1 ** 2 ,(math.sqrt(2) * x1 * x2) , x2 ** 2])  # Quadratic feature mapping


def plot_2d_data(x1, x2, labels):
    plt.figure(figsize=(8, 6))
    for i, label in enumerate(labels):
        color = "blue" if label == "Blue" else "red"
        plt.scatter(x1[i], x2[i], color=color,
                    label=label if label not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Original Data Points")
    plt.legend()
    plt.show()


def plot_3d_data(x1_t, x2_t, x3_t, labels, model):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    for i, label in enumerate(labels):
        color = "blue" if label == "Blue" else "red"
        ax.scatter(x1_t[i], x2_t[i], x3_t[i], color=color,
                   label=label if label not in ax.get_legend_handles_labels()[1] else "")
    ax.set_xlabel("x1 (original)")
    ax.set_ylabel("x2 (original)")
    ax.set_zlabel("x3 (transformed)")
    ax.set_title("Transformed Data Points in 3D")
    ax.legend()

    # Generate a meshgrid for the separating plane
    xx, yy = np.meshgrid(
        np.linspace(min(x1_t), max(x1_t), 30),
        np.linspace(min(x2_t), max(x2_t), 30)
    )

    # Retrieve coefficients
    a, b, c = model.coef_[0]
    d = model.intercept_[0]

    # Calculate zz (x3 axis) from the plane equation: a*x + b*y + c*z + d = 0 → z = -(a*x + b*y + d)/c
    zz = -(a * xx + b * yy + d) / c

    # Plot the plane
    ax.plot_surface(xx, yy, zz, alpha=0.4, color='gray', edgecolor='none')

    plt.show()



def main():
    # Given dataset
    data = np.array([
        [1, 13, "Blue"], [1, 18, "Blue"], [2, 9, "Blue"], [3, 6, "Blue"],
        [6, 3, "Blue"], [9, 2, "Blue"], [13, 1, "Blue"], [18, 1, "Blue"],
        [3, 15, "Red"], [6, 6, "Red"], [6, 11, "Red"], [9, 5, "Red"],
        [10, 10, "Red"], [11, 5, "Red"], [12, 6, "Red"], [16, 3, "Red"]
    ])

    # Extract features and labels
    x1 = data[:, 0].astype(float)
    x2 = data[:, 1].astype(float)
    labels = data[:, 2]
    y = np.array([1 if label == "Red" else 0 for label in labels])  # Convert labels to binary

    # Plot original 2D data
    plot_2d_data(x1, x2, labels)

    # Transform the data points
    transformed_data = np.array([Transform(x1[i], x2[i]) for i in range(len(x1))])
    x1_t, x2_t, x3_t = transformed_data[:, 0], transformed_data[:, 1], transformed_data[:, 2]

    # Train a logistic regression model in 3D
    model = LogisticRegression()
    model.fit(transformed_data, y)

    # Plot transformed data with proper separating plane
    plot_3d_data(x1_t, x2_t, x3_t, labels, model)


if __name__ == "__main__":
    main()
