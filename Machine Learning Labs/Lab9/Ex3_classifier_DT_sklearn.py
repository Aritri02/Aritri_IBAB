from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.tree import plot_tree
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('/home/ibab/Downloads/sonar.csv')
print(data.head())
print(data.dtypes)
X=data.iloc[:,:-1]
y=data.iloc[:,-1]
le=LabelEncoder()
y=le.fit_transform(y)
X_train, X_test,y_train, y_test = train_test_split(X, y,test_size=0.30,random_state=99)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model=DecisionTreeClassifier(random_state=44)
model.fit(X_train_scaled,y_train)
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
plt.figure(figsize=(10,8), dpi=150)
plot_tree(model, feature_names=X.columns)
plt.show()











