import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,r2_score


##loading the dataset
def load_data():
    X, y = load_diabetes(return_X_y=True)
    X = pd.DataFrame(X)
    y = pd.DataFrame(y).values.ravel()
    return X, y


# Function to create a bootstrap sample from the data (with replacement)
def bootstrap_sample(X, y, random_state=None):
    np.random.seed(random_state)
    # Generate random indices with replacement
    indices = np.random.choice(len(X), size=len(X), replace=True)
    # Use the indices to create bootstrap samples
    X_sample = X.iloc[indices]
    y_sample = y[indices]
    return X_sample, y_sample


# Bagging Regressor Implementation with DecisionTreeRegressor from scikit-learn
def bagging_regressor_fit(X_train, y_train, n_estimators=10, max_depth=5, random_state=None):
    np.random.seed(random_state)
    models = []

    for _ in range(n_estimators):
        # Create a bootstrap sample
        X_sample, y_sample = bootstrap_sample(X_train, y_train, random_state=random_state)

        # Train the base model (decision tree) on the bootstrap sample
        model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
        model.fit(X_sample, y_sample)
        models.append(model)

    return models


def bagging_regressor_predict(X_test, models):
    predictions = np.zeros((len(X_test), len(models)))

    for i, model in enumerate(models):
        predictions[:, i] = model.predict(X_test)

    # Aggregate predictions by averaging
    return predictions.mean(axis=1)


# Main Function
def main():
    # Load data
    X, y = load_data()

    # Split into training and test sets (70% train, 30% test)
    train_size = int(0.7 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Fit the Bagging Regressor with Decision Tree as base model
    models = bagging_regressor_fit(X_train, y_train, n_estimators=50, max_depth=5, random_state=42)

    # Make predictions on the test set
    y_pred = bagging_regressor_predict(X_test, models)

    # Calculate the Mean Squared Error of the predictions
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error of Bagging Regressor (Decision Tree base model): {mse:.4f}")

    #Calculate the r2_score of the predictions
    r2=r2_score(y_test,y_pred)
    print(f"R^2 score of Bagging Regressor (Decision Tree base model): {r2:.4f}")

if __name__ == "__main__":
    main()
