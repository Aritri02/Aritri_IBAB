##Use of SVM & the different types of kernel functions
##Linear, Polynomial and RBF kernels are used

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay

# Sample data
X = np.array([
    [0.4, -0.7], [-1.5, -1.0], [-1.4, -0.9], [-1.3, -1.2],
    [-1.1, -0.2], [-1.2, -0.4], [-0.5, 1.2], [-1.5, 2.1],
    [1.0, 1.0], [1.3, 0.8], [1.2, 0.5], [0.2, -2.0],
    [0.5, -2.4], [0.2, -2.3], [0.0, -2.7], [1.3, 2.1],
])
y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])

def plot_training_data_with_decision_boundary(
    kernel, ax=None, long_title=True, support_vectors=True
):
    """Train an SVM with the given kernel and plot its decision boundary."""
    clf = svm.SVC(kernel=kernel, gamma=2).fit(X, y)

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))

    x_min, x_max, y_min, y_max = -3, 3, -3, 3
    ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))

    # Plot decision region
    common_params = {"estimator": clf, "X": X, "ax": ax}
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="predict",
        plot_method="pcolormesh",
        alpha=0.3,
    )
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="decision_function",
        plot_method="contour",
        levels=[-1, 0, 1],
        colors=["k", "k", "k"],
        linestyles=["--", "-", "--"],
    )

    if support_vectors:
        ax.scatter(
            clf.support_vectors_[:, 0],
            clf.support_vectors_[:, 1],
            s=150,
            facecolors="none",
            edgecolors="k",
        )

    # Plot data
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolors="k")
    ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")

    title = f"Decision boundaries of {kernel} kernel in SVC" if long_title else kernel
    ax.set_title(title)

def main():
    """Main function to plot decision boundaries using various SVM kernels."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, kernel in zip(axes, ["linear", "poly", "rbf"]):
        plot_training_data_with_decision_boundary(kernel=kernel, ax=ax, long_title=False)

    fig.suptitle("SVM Decision Boundaries with Different Kernels")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
