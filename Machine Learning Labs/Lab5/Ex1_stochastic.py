#Using stochastic Gradient descent
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.matrixlib.defmatrix import asmatrix
# Load the dataset
def load_data(df):
    print("The contents of the data frame is: ")
    print(df.head)
    print("The shape of the data frame is: ")
    print(df.shape)
    print("The non-null values of the data frame is: ")
    print(df.info())
    print("The description of the data frame is: ")
    print(df.describe())
    print("The missing values of the data frame is: ")
    print(df.isnull().sum())
    print("The column names of the data frame is: ")
    print(df.columns)
    return df
def standardize_data(x_train, x_test):
    # Calculate the mean and standard deviation of the training data
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    # Standardize the training data
    x_train_standardized = (x_train - mean) / std
    # Standardize the test data using the training set mean and std
    x_test_standardized = (x_test - mean) / std
    return x_train_standardized, x_test_standardized
# Function to compute R^2 manually
def r_squared(y_true, y_pred):
    # Calculate Total Sum of Squares (TSS)
    total_sum_of_squares = np.sum((y_true - np.mean(y_true)) ** 2)
    # Calculate Residual Sum of Squares (RSS)
    residual_sum_of_squares = np.sum((y_true - y_pred) ** 2)
    # R-squared formula
    r2 = 1 - (residual_sum_of_squares / total_sum_of_squares)
    return r2
#spliting of data
def split_data(df):
    data=load_data(df)
    s_data=data.sample(frac=1)
    train_size=0.70
    train_samples=int(len(s_data)*train_size)
    train_data=s_data.iloc[:train_samples]
    test_data=s_data.iloc[train_samples:]
    x_train=train_data.drop(["BMI","BP","blood_sugar","disease_score","disease_score_fluct","Gender"],axis=1)
    y_train = train_data[["disease_score"]]
    x_test=test_data.drop(["BMI","BP","blood_sugar","disease_score","disease_score_fluct","Gender"],axis=1)
    y_test = test_data[["disease_score"]]
    # Standardize the features (X_train and X_test)
    x_train_standardized, x_test_standardized = standardize_data(x_train.values, x_test.values)
    # Add intercept term (column of ones) for both train and test sets
    x_train_standardized = np.c_[np.ones(x_train_standardized.shape[0]), x_train_standardized]
    x_test_standardized = np.c_[np.ones(x_test_standardized.shape[0]), x_test_standardized]
    # Convert the data to matrices
    x_train_matrix = np.array(x_train_standardized)
    x_test_matrix = np.array(x_test_standardized)
    y_train_matrix = np.array(y_train.values).reshape(-1, 1)
    y_test_matrix = np.array(y_test.values).reshape(-1, 1)
    return x_train_matrix, x_test_matrix, y_train_matrix, y_test_matrix
def hypothesis(x_train,theta):
    return np.dot(x_train,theta)
def cost(x_train,y_train,theta):
    y_pred=hypothesis(x_train,theta)
    e=y_train - y_pred
    sq_error = np.square(e)
    j = np.sum(sq_error) / (2 * len(y_train))
    return j
def gradient(x_train,y_train, theta,alpha,iterations):
    n=len(y_train)
    cost_list=[]
    for k in range(iterations):
        # Shuffle the data
        shuffle_indices = np.random.permutation(n)
        x_train_shuffled = x_train[shuffle_indices]
        y_train_shuffled = y_train[shuffle_indices]
    for i in range(n):
        x_sample = x_train_shuffled[i:i + 1]
        y_sample = y_train_shuffled[i:i + 1]
        y_pred=hypothesis(x_sample,theta)
        e=y_sample-y_pred
        g= -x_sample.T.dot(e)
        theta=theta-alpha*g
        cost_list.append(cost(x_train,y_train,theta))
    return theta,cost_list
def update_theta(f,alpha=0.1,iterations=40):
    x_train, x_test, y_train, y_test = split_data(f)
    theta = np.zeros((x_train.shape[1], 1))
    optimal_theta, cost_list = gradient(x_train,y_train, theta,alpha,iterations)
    y_pred=hypothesis(x_test,optimal_theta)
    r2 = r_squared(y_test, y_pred)
    print("R^2 (Coefficient of Determination):", r2)
    print("Optimal Parameters (theta):", optimal_theta)
    print("Cost History:", cost_list)
    #Plot the cost history
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(cost_list, color='blue')
    plt.title('Cost History over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.subplot(1, 2, 2)
    x_feature=x_test[:,1]
    plt.scatter(x_feature,y_test, color='red',label="Actual Data")
    plt.plot(x_feature,y_pred, color='green',linewidth=2,label="Regression Line")
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title("Stochastic Gradient Descent")
    plt.legend()
    plt.show()
    return optimal_theta, cost_list
def main():
    df = pd.read_csv('/home/ibab/Downloads/simulated_data_multiple_linear_regression_for_ML.csv')
    optimal_theta, cost_list = update_theta(df)
    # print(cost_list[-1]<cost_list[0])
    # print(cost_list[-1])
    # print(cost_list[0])
if __name__ == '__main__':
    main()