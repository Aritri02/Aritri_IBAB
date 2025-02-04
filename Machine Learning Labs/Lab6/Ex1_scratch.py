from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test,y_train, y_test = train_test_split(X, y,test_size=0.30,random_state=99)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
def k_fold_cross_validation(X, k=10):
    fold_size=len(X)//10
    indices = np.arange(len(X))
    folds = []
    for i in range(k):
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
        folds.append((train_indices, test_indices))
    return folds
k_fold=k_fold_cross_validation(X_train, k=10)
print(k_fold)
model= LogisticRegression(solver='liblinear',max_iter=1000,random_state=1)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
scores = []
# Iterate through each fold
for train_indices, test_indices in k_fold:
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    # Train the model on the training data
    model.fit(X_train, y_train)
    # Make predictions on the test data
    y_pred = model.predict(X_test)
    # Calculate the accuracy score for this fold
    fold_score = accuracy_score(y_test, y_pred)
    # Append the fold score to the list of scores
    scores.append(fold_score)
# Calculate the mean accuracy across all folds
mean_accuracy = np.mean(scores)
print("K-Fold Cross-Validation Scores:", scores)
print("Mean Accuracy:", mean_accuracy)


