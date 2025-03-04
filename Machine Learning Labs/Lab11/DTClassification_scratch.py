#Building a Decision Tree Classifier from scratch
#Iris dataset is used

import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
import math
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
# Load the Iris dataset
X, y = load_iris(return_X_y=True)


# Helper functions: Entropy and Information Gain
def entropy(y):
    # Count the occurrences of each class
    class_counts = Counter(y)
    # Total number of data points
    total_points = len(y)
    # Calculate the entropy
    entropy = 0
    for count in class_counts.values():
        probability = count / total_points  # Probability of each class
        entropy -= probability * math.log2(probability)  # Entropy formula
    return entropy


def information_gain(X, y, feature_index, threshold):
    left_index = X[:, feature_index] <= threshold
    right_index = X[:, feature_index] > threshold

    left_y, right_y = y[left_index], y[right_index]

    if len(left_y) == 0 or len(right_y) == 0:
        return 0

    entropy_before = entropy(y)
    entropy_left = entropy(left_y)
    entropy_right = entropy(right_y)

    weighted_entropy = (len(left_y) / len(y)) * entropy_left + (len(right_y) / len(y)) * entropy_right

    return entropy_before - weighted_entropy


def best_split(X, y):
    best_feature = None
    best_threshold = None
    best_gain = 0

    for feature_index in range(X.shape[1]):
        thresholds = np.unique(X[:, feature_index])

        for threshold in thresholds:
            gain = information_gain(X, y, feature_index, threshold)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_index
                best_threshold = threshold

    return best_feature, best_threshold, best_gain


# Function to build the decision tree recursively
def build_tree(X, y, depth=0, max_depth=None):
    if len(np.unique(y)) == 1:
        return {'value': y[0]}  # Leaf node with the class label

    if max_depth is not None and depth >= max_depth:
        most_common_class = Counter(y).most_common(1)[0][0]
        return {'value': most_common_class}  # Leaf node with the most common class

    feature, threshold, gain = best_split(X, y)

    if gain == 0:
        most_common_class = Counter(y).most_common(1)[0][0]
        return {'value': most_common_class}  # Leaf node if no good split is found

    # Split the data
    left_index = X[:, feature] <= threshold
    right_index = X[:, feature] > threshold

    left_node = build_tree(X[left_index], y[left_index], depth + 1, max_depth)
    right_node = build_tree(X[right_index], y[right_index], depth + 1, max_depth)

    # Return a dictionary for the current node
    return {'feature': feature, 'threshold': threshold, 'left': left_node, 'right': right_node}


# Function to predict using the decision tree
def predict_tree(node, X):
    if 'value' in node:  # If we are at a leaf node
        return node['value']

    if X[node['feature']] <= node['threshold']:
        return predict_tree(node['left'], X)
    else:
        return predict_tree(node['right'], X)


# Train the decision tree
tree = build_tree(X, y, max_depth=3)

# Predict using the trained tree
predictions = [predict_tree(tree, x) for x in X]

# Measure accuracy
accuracy = np.sum(predictions == y) / len(y)
print(f"Accuracy: {accuracy * 100:.2f}%")


