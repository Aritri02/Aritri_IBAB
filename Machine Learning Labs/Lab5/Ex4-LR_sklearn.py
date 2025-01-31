from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
# load the breast cancer dataset
X, y = load_breast_cancer(return_X_y=True)

# split the train and test dataset
X_train, X_test,y_train, y_test = train_test_split(X, y,test_size=0.40,random_state=23)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# LogisticRegression
model= LogisticRegression(random_state=1)
model.fit(X_train_scaled, y_train)

# Prediction
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Get the probability of the positive class

acc = accuracy_score(y_test, y_pred)
print("Logistic Regression model accuracy (in %):", acc*100)

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Visualize the decision boundary with accuracy information
plt.figure(figsize=(16, 10))
plt.subplot(1, 2, 1)
sns.scatterplot(x=X_test[:, 4], y=X_test[:, 24], hue=y_test, palette={0: 'purple', 1: 'olive'}, marker='o')
plt.xlabel("texture_mean")
plt.ylabel("texture_worst")
plt.title("Logistic Regression Decision Boundary\nAccuracy: {:.2f}%".format(acc * 100))
plt.legend(title="Cancer", loc="upper right")
# Plot the ROC curve
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--',label='Random')  # Diagonal line
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()