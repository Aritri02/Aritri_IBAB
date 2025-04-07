import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import DecisionBoundaryDisplay


def load_data():
    # Dataset: [x1, x2, label]
    data = np.array([
        [6, 5, "Blue"], [6, 9, "Blue"], [8, 6, "Red"], [8, 8, "Red"],
        [8, 10, "Red"], [9, 2, "Blue"], [9, 5, "Red"], [10, 10, "Red"],
        [10, 13, "Blue"], [11, 5, "Red"], [11, 8, "Red"], [12, 6, "Red"],
        [12, 11, "Blue"], [13, 4, "Blue"], [14, 8, "Blue"]
    ])

    X = data[:, :2].astype(float)
    y = data[:, 2]

    # Encode labels: Blue -> 0, Red -> 1
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X, y_encoded, label_encoder


def plot_svm_decision_boundary(X, y, kernel, title, ax):
    clf = svm.SVC(kernel=kernel, gamma='scale', degree=3, C=1)
    clf.fit(X, y)

    # Plot decision boundary
    DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        response_method="predict",
        plot_method="pcolormesh",
        alpha=0.3,
        ax=ax,
    )

    # Plot original data points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k', s=80)
    ax.set_title(f"{title} Kernel")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(*scatter.legend_elements(), title="Class")


def main():
    # Load data
    X, y, label_encoder = load_data()

    # Setup plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot with RBF kernel
    plot_svm_decision_boundary(X, y, kernel='rbf', title="RBF", ax=axes[0])

    # Plot with Polynomial kernel
    plot_svm_decision_boundary(X, y, kernel='poly', title="Polynomial", ax=axes[1])

    fig.suptitle("SVM Decision Boundaries Comparison")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
