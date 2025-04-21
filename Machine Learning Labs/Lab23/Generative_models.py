##Implementation of Generative Model##
##Iris dataset is used##
##skleran implementation##

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from collections import defaultdict

def joint_probability_model(X_train, y_train, X_test, y_test):
    joint_counts = defaultdict(lambda: defaultdict(int))
    label_counts = defaultdict(int)

    for xi, yi in zip(X_train, y_train):
        key = tuple(xi)
        joint_counts[key][yi] += 1
        label_counts[yi] += 1

    joint_probs = {}
    for key, label_dict in joint_counts.items():
        total_count = sum(label_dict.values())
        joint_probs[key] = {label: count / total_count for label, count in label_dict.items()}

    def predict(X):
        preds = []
        for x in X:
            key = tuple(x)
            if key in joint_probs:
                pred = max(joint_probs[key].items(), key=lambda x: x[1])[0]
            else:
                pred = max(label_counts.items(), key=lambda x: x[1])[0]  # fallback
            preds.append(pred)
        return np.array(preds)

    y_pred = predict(X_test)
    return accuracy_score(y_test, y_pred)

def decision_tree_model(X_train, y_train, X_test, y_test):
    clf = DecisionTreeClassifier(max_depth=2, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred), clf

def visualize_tree(clf, feature_names, class_names):
    plt.figure(figsize=(10, 6))
    plot_tree(clf, feature_names=feature_names, class_names=class_names, filled=True)
    plt.title("Decision Tree (max_depth=2)")
    plt.show()

def main():
    iris = load_iris()
    X_original = iris.data[:, :2]  # Only SepalLength and SepalWidth
    y = iris.target
    feature_names = iris.feature_names[:2]

    noise_levels = [0.0, 0.1, 0.2, 0.3]
    bin_sizes = [2, 3, 4, 5, 6]
    results = []

    for noise_std in noise_levels:
        noisy_X = X_original + np.random.normal(0, noise_std, X_original.shape)

        for bins in bin_sizes:
            discretizer = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
            X_discrete = discretizer.fit_transform(noisy_X).astype(int)

            X_train, X_test, y_train, y_test = train_test_split(
                X_discrete, y, test_size=0.3, random_state=42
            )

            jp_acc = joint_probability_model(X_train, y_train, X_test, y_test)
            dt_acc, _ = decision_tree_model(X_train, y_train, X_test, y_test)

            results.append((noise_std, bins, jp_acc, dt_acc))

    # Print summary
    print(f"{'Noise':<10}{'Bins':<6}{'JP Accuracy(%)':<15}{'DT Accuracy(%)':<15}")
    for r in results:
        print(f"{r[0]:<10}{r[1]:<6}{r[2]*100:<15.2f}{r[3]*100:<15.2f}")

    # Visualize one example decision tree (e.g., noise=0.2, bins=4)
    sample_noise = 0.2
    sample_bins = 4
    noisy_X = X_original + np.random.normal(0, sample_noise, X_original.shape)
    discretizer = KBinsDiscretizer(n_bins=sample_bins, encode='ordinal', strategy='uniform')
    X_discrete = discretizer.fit_transform(noisy_X).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X_discrete, y, test_size=0.3, random_state=42
    )
    _, clf = decision_tree_model(X_train, y_train, X_test, y_test)
    visualize_tree(clf, feature_names, iris.target_names)

if __name__ == "__main__":
    main()
