import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import resample
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score


# Function to train multiple decision trees on bootstrapped samples
def train_aggregated_trees(X, y, n_trees=10, max_depth=None):
    trees = []
    n_samples = X.shape[0]

    for _ in range(n_trees):
        X_sample, y_sample = resample(X, y, n_samples=n_samples)
        tree = DecisionTreeRegressor(max_depth=max_depth)
        tree.fit(X_sample, y_sample)
        trees.append(tree)

    return trees


# Function to aggregate predictions from all trees
def predict_aggregated_trees(trees, X):
    predictions = np.array([tree.predict(X) for tree in trees])  # Shape (n_trees, n_samples)
    return np.mean(predictions, axis=0)  # Average over all trees

def main():
    # Generate synthetic regression data
    X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train aggregated tree regressor
    trees = train_aggregated_trees(X_train, y_train, n_trees=20, max_depth=5)

    # Predict and evaluate
    y_pred = predict_aggregated_trees(trees, X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2=r2_score(y_test,y_pred)
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"The r^2 score is: {r2:.4f}")

if __name__ == "__main__":
    main()
