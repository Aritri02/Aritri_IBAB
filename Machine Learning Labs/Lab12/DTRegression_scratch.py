#Building Decision Tree for regression problems from scratch
#diabetes dataset is used

import numpy as np
from collections import Counter
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
# Load the Iris dataset
X, y = load_diabetes(return_X_y=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Helper function: Calculate the mean squared error of a set of values
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# Helper function: Find the best split for the data
def best_split(X, y):
    best_feature, best_threshold, best_mse, best_left_y, best_right_y = None, None, float('inf'), None, None

    for feature in range(X.shape[1]):  # Iterate through each feature
        feature_values = X[:, feature]
        thresholds = np.unique(feature_values)  # Unique values to test as thresholds

        for threshold in thresholds:
            # Split the data based on the threshold
            left_index = feature_values <= threshold
            right_index = feature_values > threshold

            left_y = y[left_index]
            right_y = y[right_index]

            # Skip split if one side is empty
            if len(left_y) == 0 or len(right_y) == 0:
                continue

            # Calculate the MSE of this split
            left_mse = np.mean((left_y - np.mean(left_y)) ** 2)
            right_mse = np.mean((right_y - np.mean(right_y)) ** 2)

            total_mse = (len(left_y) * left_mse + len(right_y) * right_mse) / (len(left_y) + len(right_y))

            # Update if this is the best split so far
            if total_mse < best_mse:
                best_feature = feature
                best_threshold = threshold
                best_mse = total_mse
                best_left_y = left_y
                best_right_y = right_y

    return best_feature, best_threshold, best_left_y, best_right_y


# Function to build the decision tree recursively
def build_tree(X, y, depth=0, max_depth=None):
    # If maximum depth is reached, return the mean value of y as the leaf value
    if depth >= max_depth or len(np.unique(y)) == 1:
        return np.mean(y)

    # Find the best split
    feature, threshold, left_y, right_y = best_split(X, y)

    # If no valid split was found, return the mean value of y as the leaf value
    if feature is None:
        return np.mean(y)

    # Split the data into left and right branches
    left_mask = X[:, feature] <= threshold
    right_mask = X[:, feature] > threshold
    left_node = build_tree(X[left_mask], y[left_mask], depth + 1, max_depth)
    right_node = build_tree(X[right_mask], y[right_mask], depth + 1, max_depth)

    # Return a dictionary for the current node
    return {'feature': feature, 'threshold': threshold, 'left': left_node, 'right': right_node}


# Function to predict using the decision tree
def predict_tree(tree, X):
    if isinstance(tree, dict):
        # If it's a node, make a decision based on the feature and threshold
        feature = tree['feature']
        threshold = tree['threshold']

        if X[feature] <= threshold:
            return predict_tree(tree['left'], X)
        else:
            return predict_tree(tree['right'], X)
    else:
        # If it's a leaf, return the predicted value (the mean)
        return tree


# Build the regression tree
max_depth = 5
tree = build_tree(X_train, y_train, max_depth=max_depth)

# Make predictions on the test set
y_pred = np.array([predict_tree(tree, x) for x in X_test])

# Evaluate the model's performance using Mean Squared Error (MSE)
error = mse(y_test, y_pred)
print(f'Mean Squared Error on the test set: {error:.2f}')


