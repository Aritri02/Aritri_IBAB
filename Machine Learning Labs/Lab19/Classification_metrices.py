import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    classification_report
)
from sklearn.preprocessing import StandardScaler

def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)
    specificity = TN / (TN + FP)
    f1 = f1_score(y_true, y_pred)

    return {
        "Confusion Matrix": cm,
        "Accuracy": accuracy,
        "Precision": precision,
        "Sensitivity (Recall)": sensitivity,
        "Specificity": specificity,
        "F1 Score": f1
    }

def plot_roc_curve(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f"ROC Curve (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Load the dataset
    df = pd.read_csv("/home/ibab/Desktop/Datasets/datasets/heart.csv")

    # Features and target
    X = df.drop("output", axis=1)
    y = df["output"]

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Train logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predict probabilities
    y_probs = model.predict_proba(X_test)[:, 1]

    # Threshold evaluation
    thresholds = np.arange(0.1,1.0, 0.05)
    best_threshold = 0.5
    best_f1 = 0

    print("Evaluating thresholds and metrics:\n")
    for thresh in thresholds:
        y_pred_thresh = (y_probs >= thresh).astype(int)
        metrics = calculate_metrics(y_test, y_pred_thresh)
        current_f1 = metrics["F1 Score"]
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = thresh

        print(f"Threshold: {thresh:.2f}")
        for k, v in metrics.items():
            if k == "Confusion Matrix":
                print(f"{k}:\n{v}")
            else:
                print(f"{k}: {v:.4f}")
        print("-" * 40)

    # Print best threshold and classification report
    print(f"\nBest Threshold based on F1 Score: {best_threshold:.2f} (F1 Score = {best_f1:.4f})\n")
    y_best_pred = (y_probs >= best_threshold).astype(int)
    print("Classification Report at Best Threshold:")
    print(classification_report(y_test, y_best_pred))

    # Plot ROC curve
    plot_roc_curve(y_test, y_probs)

if __name__ == "__main__":
    main()
