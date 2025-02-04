#Standardization using sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test,y_train, y_test = train_test_split(X, y,test_size=0.30,random_state=99)
scaler=StandardScaler()
scaler.fit(X_train)
print(scaler.transform(X_train))
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Logistic Regression model accuracy using sklearn(in %):", acc*100)



#Standardization from scratch
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test,y_train, y_test = train_test_split(X, y,test_size=0.30,random_state=99)
for col in range(X_train.shape[1]):
    X_train[:,col]=(X_train[:,col]-X_train[:,col].mean())/(X_train[:,col].std())
for col in range(X_test.shape[1]):
    X_test[:,col]=(X_test[:,col]-X_test[:,col].mean())/(X_test[:,col].std())
model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Logistic Regression model accuracy from scratch(in %):", acc*100)


