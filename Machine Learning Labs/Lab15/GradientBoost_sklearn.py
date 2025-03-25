# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from ISLP import load_data # Import Boston and Weekly dataset from ISLP package

# --- Regression Problem: Boston Housing Dataset ---
def boston_regression():
    # Load the Boston Housing dataset
    boston_data = load_data('Boston')

    # Convert to pandas DataFrame (if it's not already)
    boston_data = pd.DataFrame(boston_data)

    # --- EDA for Boston Housing Dataset ---
    print("Boston Housing Dataset - EDA")
    print("\nFirst 5 rows of the dataset:")
    print(boston_data.head())

    print("\nData Types and Missing Values:")
    print(boston_data.info())

    print("\nStatistical Summary of the Dataset:")
    print(boston_data.describe())

    # Check for missing values
    print("\nMissing Values in the Dataset:")
    print(boston_data.isnull().sum())

    # Plotting the distribution of the target variable ('medv' - median house value)
    plt.figure(figsize=(8, 6))
    sns.histplot(boston_data['medv'], kde=True, bins=30)
    plt.title('Distribution of Median House Value (medv)')
    plt.xlabel('Median House Value (medv)')
    plt.ylabel('Frequency')
    plt.show()

    # Correlation matrix to check the relationship between features
    plt.figure(figsize=(12, 8))
    corr_matrix = boston_data.corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix of Features')
    plt.show()

    # --- Regression Model ---
    # Split the data into features (X) and target (y)
    X_boston = boston_data.drop('medv', axis=1)  # 'medv' is the target (Median house value)
    y_boston = boston_data['medv']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_boston, y_boston, test_size=0.3, random_state=42)

    # Initialize the Gradient Boosting Regressor
    gbr_regressor = GradientBoostingRegressor(n_estimators=100, random_state=42)

    # Train the model
    gbr_regressor.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred_reg = gbr_regressor.predict(X_test)

    # Evaluate the model
    mse_reg = mean_squared_error(y_test, y_pred_reg)
    r2_reg = r2_score(y_test, y_pred_reg)

    print(f"\nRegression Model - Mean Squared Error: {mse_reg:.2f}")
    print(f"Regression Model - R^2 Score: {r2_reg:.2f}")


# --- Classification Problem: Weekly Dataset ---
def weekly_classification():
    # Load the Weekly dataset
    weekly_data = load_data('Weekly')

    # Convert to pandas DataFrame (if it's not already)
    weekly_data = pd.DataFrame(weekly_data)

    # --- EDA for Weekly Dataset ---
    print("\nWeekly Dataset - EDA")
    print("\nFirst 5 rows of the dataset:")
    print(weekly_data.head())

    print("\nData Types and Missing Values:")
    print(weekly_data.info())

    print("\nStatistical Summary of the Dataset:")
    print(weekly_data.describe())

    # Check for missing values
    print("\nMissing Values in the Dataset:")
    print(weekly_data.isnull().sum())

    # Plotting the distribution of the target variable ('Direction')
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Direction', data=weekly_data)
    plt.title('Distribution of Stock Market Direction (Up/Down)')
    plt.xlabel('Direction')
    plt.ylabel('Frequency')
    plt.show()

    # # Correlation matrix for numerical features
    # plt.figure(figsize=(12, 8))
    # corr_matrix = weekly_data.corr()
    # sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt='.2f', linewidths=0.5)
    # plt.title('Correlation Matrix of Features')
    # plt.show()

    # --- Classification Model ---
    # Convert the 'Direction' variable to a binary classification (1 for 'Up', 0 for 'Down')
    weekly_data['Direction'] = np.where(weekly_data['Direction'] == 'Up', 1, 0)

    # Split the data into features (X) and target (y)
    X_weekly = weekly_data.drop(['Direction', 'Year'], axis=1)  # Dropping Year for the model
    y_weekly = weekly_data['Direction']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_weekly, y_weekly, test_size=0.3, random_state=42)

    # Initialize the Gradient Boosting Classifier
    gbr_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)

    # Train the model
    gbr_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred_clf = gbr_classifier.predict(X_test)

    # Evaluate the model
    accuracy_clf = accuracy_score(y_test, y_pred_clf)
    print(f"\nClassification Model - Accuracy: {accuracy_clf * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_clf))


# Main function to run both regression and classification
def main():
    # Running the Regression problem (Boston housing dataset)
    print("--- Gradient Boosting Regression (Boston Housing Dataset) ---")
    boston_regression()

    # Running the Classification problem (Weekly dataset)
    print("\n--- Gradient Boosting Classification (Weekly Dataset) ---")
    weekly_classification()


if __name__ == "__main__":
    main()
