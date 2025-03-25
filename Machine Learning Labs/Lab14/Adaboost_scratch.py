##Implementation of Adaboost classifier from scratch##
##Iris dataset was used##



import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def load_data():
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    # We will only use two classes (class 0 and class 1) for binary classification
    X = X[y != 2]
    y = y[y != 2]
    return X, y


def train_weak_classifier(X, y, sample_weights):
    # Train a weak classifier (decision stump) using sample weights
    stump = DecisionTreeClassifier(max_depth=1)
    stump.fit(X, y, sample_weight=sample_weights)
    return stump


def adaboost_fit(X, y, n_estimators=50):
    n_samples, n_features = X.shape
    # Initialize sample weights: equal weight for each sample
    sample_weights = np.ones(n_samples) / n_samples

    # Store the models and their alphas (weights)
    models = []
    alphas = []

    # Convert labels from 0, 1 to -1, 1 for binary classification
    y = 2 * (y - 1) - 1

    for _ in range(n_estimators):
        # Train a weak classifier (decision stump)
        stump = train_weak_classifier(X, y, sample_weights)

        # Make predictions on the training data
        pred = stump.predict(X)

        # Calculate the weighted error rate (error of the classifier)
        error = np.sum(sample_weights * (pred != y)) / np.sum(sample_weights)

        # Check if error is 0 or 1 and handle these edge cases
        if error == 0:
            # Perfect classifier, stop training further
            print("Perfect classifier found, stopping early.")
            break
        elif error == 1:
            # If the error is 1, we have a classifier that performs terribly, stop training
            print("Error is 1, which means the classifier is not better than random guessing. Stopping.")
            break

        # Calculate alpha (classifier weight)
        alpha = 0.5 * np.log((1 - error) / error)

        # Update sample weights: misclassified samples get higher weights
        sample_weights = sample_weights * np.exp(-alpha * y * pred)

        # Normalize the sample weights
        sample_weights = sample_weights / np.sum(sample_weights)  # Normalize to sum to 1

        # Ensure there are no NaN values in sample_weights
        if np.any(np.isnan(sample_weights)):
            print("NaN values found in sample weights. Resetting weights.")
            sample_weights = np.ones(n_samples) / n_samples  # Reset weights if NaNs are found

        # Store the weak classifier and its alpha value
        models.append(stump)
        alphas.append(alpha)

    return models, alphas


def adaboost_predict(X, models, alphas):
    # Aggregate predictions from all weak classifiers
    clf_preds = np.zeros(X.shape[0])

    for model, alpha in zip(models, alphas):
        clf_preds += alpha * model.predict(X)

    # The final prediction is the sign of the aggregated predictions
    return np.sign(clf_preds)


def main():
    # Load data
    X, y = load_data()

    # Split the data into training and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the AdaBoost model
    models, alphas = adaboost_fit(X_train, y_train, n_estimators=100)

    # Predict on the test set
    y_pred = adaboost_predict(X_test, models, alphas)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of AdaBoost Classifier (on test set): {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
