##Implementation of Adaboost classifier using sklearn##
##iris dataset was used##


# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc)
import matplotlib.pyplot as plt

def load_data():
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y


def main():
    # Load data
    X, y = load_data()

    # Split the dataset into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize the AdaBoost classifier
    ada_boost_classifier = AdaBoostClassifier(n_estimators=50, random_state=42)

    # Train the classifier
    ada_boost_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = ada_boost_classifier.predict(X_test)

    # Evaluate the classifier on the test set (Normal Score)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of AdaBoost Classifier: {accuracy * 100:.2f}%")

    # --- 10-Fold Cross Validation ---
    cv_scores = cross_val_score(ada_boost_classifier, X, y, cv=10, scoring='accuracy')

    # Cross-validation scores and their mean
    print(f"Mean 10-Fold Cross Validation Accuracy: {cv_scores.mean() * 100:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Precision, Recall, F1-Score
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    # Compute ROC curve and ROC-AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    roc_auc = auc(fpr, tpr)
    print(f"ROC-AUC: {roc_auc:.2f}")

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line (random classifier)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.show()


if __name__ == "__main__":
    main()
