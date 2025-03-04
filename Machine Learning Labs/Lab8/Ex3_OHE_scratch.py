import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import defaultdict
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names
df = pd.DataFrame(X, columns=feature_names)
np.random.seed(1)  # To make the random values reproducible
categories = ['low', 'medium', 'high']
df['categorical_feature'] = np.random.choice(categories, size=df.shape[0])
ordinal_mapping = {'low': 1, 'medium': 2, 'high': 3}
df['ordinal_encoded'] = df['categorical_feature'].map(ordinal_mapping)
print("Ordinal Encoding (Manual):")
print(df[['categorical_feature', 'ordinal_encoded']].head())
one_hot_encoded_df = df.copy()
categories = df['categorical_feature'].unique()
for category in categories:
    one_hot_encoded_df[category] = (df['categorical_feature'] == category).astype(int)
print("\nOne-Hot Encoding (Manual):")
print(one_hot_encoded_df[['categorical_feature', 'low', 'medium', 'high']].head())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
