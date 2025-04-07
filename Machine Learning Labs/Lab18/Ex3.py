import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def load_iris_subset():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names

    # Select only class 0 (setosa) and class 1 (versicolor)
    mask = y < 2
    X = X[mask][:, :2]  # Take only first two features
    y = y[mask]
    return X, y, feature_names[:2]


def split_data(X, y, test_ratio=0.1):
    # Separate by class and split 10% from each class for testing
    X_train, X_test, y_train, y_test = [], [], [], []

    for class_label in np.unique(y):
        X_class = X[y == class_label]
        y_class = y[y == class_label]
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
            X_class, y_class, test_size=test_ratio, random_state=42
        )
        X_train.extend(X_train_c)
        X_test.extend(X_test_c)
        y_train.extend(y_train_c)
        y_test.extend(y_test_c)

    return (
        np.array(X_train),
        np.array(X_test),
        np.array(y_train),
        np.array(y_test),
    )


def plot_decision_boundary(clf, X, y, feature_names, title="SVM Decision Boundary"):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors="k")
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title(title)
    plt.grid(True)
    plt.show()


def main():
    # Load and filter dataset
    X, y, feature_names = load_iris_subset()

    # Split the data
    X_train, X_test, y_train, y_test = split_data(X, y, test_ratio=0.1)

    # Train SVM
    clf = svm.SVC(kernel='linear', C=1.0)
    clf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc*100:.2f}%")

    # Plot decision boundary
    plot_decision_boundary(clf, np.vstack((X_train, X_test)), np.hstack((y_train, y_test)), feature_names)


if __name__ == "__main__":
    main()
