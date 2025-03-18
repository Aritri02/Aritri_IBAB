##Bagging Regressor using sklearn##
##using diabetes dataset##


# importing the modules
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns


##loading the dataset
def load_data():
    X, y = load_diabetes(return_X_y=True)
    X = pd.DataFrame(X)
    y = pd.DataFrame(y).values.ravel()
    return X, y


def main():
    X, y = load_data()
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=999)

    # Scaling the data
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # Training a single Decision Tree model
    single_tree = DecisionTreeRegressor(random_state=42)
    single_tree.fit(x_train, y_train)
    y_pred_single = single_tree.predict(x_test)
    mse_single = mean_squared_error(y_test, y_pred_single)
    r2_single = r2_score(y_test, y_pred_single)
    print(f"Single Decision Tree - MSE: {mse_single:.2f}")
    print(f"Single Decision Tree - R^2 score: {r2_single:.2f}")

    # Creating an instance of DecisionTreeRegressor for bagging
    base_estimator = DecisionTreeRegressor(random_state=42)

    # Bagging Regressor using DecisionTreeRegressor as base estimator
    bagging_model = BaggingRegressor(estimator=base_estimator, n_estimators=10, random_state=42, bootstrap=True)
    bagging_model.fit(x_train, y_train)  # Train the Bagging Regressor

    # Make predictions with the Bagging model
    y_pred_bagging = bagging_model.predict(x_test)
    mse_bagging = mean_squared_error(y_test, y_pred_bagging)
    r2_bagging = r2_score(y_test, y_pred_bagging)
    print(f"Bagging Regressor - MSE: {mse_bagging:.2f}")
    print(f"Bagging Regressor - R^2 score: {r2_bagging:.2f}")

    # Using cross_val_score with R^2 score as the scoring metric
    r2_scores = cross_val_score(bagging_model, X, y, cv=kf, scoring='r2')
    print(f"R^2 score for each fold: {r2_scores}")
    print(f"Average R^2 score across 10 folds: {np.mean(r2_scores):.2f}")


if __name__ == "__main__":
    main()



############################################################################################
##Bagging Classifier using sklearn##
##Iris dataset is used ##

# # importing the modules
# import pandas as pd
# import numpy as np
# from sklearn.datasets import load_iris
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import BaggingClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import KFold
# from sklearn.model_selection import cross_val_score
# import matplotlib.pyplot as plt
# import seaborn as sns
#
#
# # Loading the dataset
# def load_data():
#     X, y = load_iris(return_X_y=True)
#     X = pd.DataFrame(X)
#     y = pd.DataFrame(y).values.ravel()
#     return X, y
#
#
# def main():
#     X, y = load_data()
#     x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=999)
#
#     # Training a single Decision Tree model for classification
#     single_tree = DecisionTreeClassifier(random_state=42)
#     single_tree.fit(x_train, y_train)
#     y_pred_single = single_tree.predict(x_test)
#     accuracy_single = accuracy_score(y_test, y_pred_single)
#     print(f"Single Decision Tree - Accuracy: {accuracy_single:.2f}")
#
#     # Creating an instance of DecisionTreeClassifier for bagging
#     base_estimator = DecisionTreeClassifier(random_state=42)
#
#     # Bagging Classifier using DecisionTreeClassifier as base estimator
#     bagging_model = BaggingClassifier(estimator=base_estimator, n_estimators=20, random_state=42, bootstrap=True)
#     bagging_model.fit(x_train, y_train)  # Train the Bagging Classifier
#
#     # Make predictions with the Bagging model
#     y_pred_bagging = bagging_model.predict(x_test)
#     accuracy_bagging = accuracy_score(y_test, y_pred_bagging)
#     print(f"Bagging Classifier - Accuracy: {accuracy_bagging:.2f}")
#
#     # Using cross_val_score with accuracy as the scoring metric
#     accuracy_scores = cross_val_score(bagging_model, X, y, cv=KFold(n_splits=10, shuffle=True, random_state=42), scoring='accuracy')
#     print(f"Accuracy for each fold: {accuracy_scores}")
#     print(f"Average Accuracy across 10 folds: {np.mean(accuracy_scores):.2f}")
#
#
# if __name__ == "__main__":
#     main()
#
