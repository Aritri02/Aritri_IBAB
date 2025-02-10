#using breast cancer dataset
#ridge regression
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
X,y=load_breast_cancer(return_X_y=True)
X_train, X_test,y_train, y_test = train_test_split(X, y,test_size=0.30,random_state=99)
ridge = Ridge(alpha=0.5)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
r2_score = r2_score(y_test, y_pred)
print("The R^2 score for Ridge regression is: ",r2_score)

#linear regression
from sklearn.metrics import r2_score
X,y=load_breast_cancer(return_X_y=True)
X_train, X_test,y_train, y_test = train_test_split(X, y,test_size=0.30,random_state=99)
model=LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
r2_score_LR = r2_score(y_test, y_pred)
print("The R^2 score for Linear regression is: ",r2_score_LR)

#lasso regression
from sklearn.metrics import r2_score
X,y=load_breast_cancer(return_X_y=True)
X_train, X_test,y_train, y_test = train_test_split(X, y,test_size=0.30,random_state=99)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)
r2_score_lasso = r2_score(y_test, y_pred)
print("The R^2 score for Lasso regression is: ",r2_score_lasso)

#using sonar dataset
#logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
df=pd.read_csv("/home/ibab/Downloads/sonar.csv")
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
le=LabelEncoder()
y=le.fit_transform(y)
X_train, X_test,y_train, y_test = train_test_split(X, y,test_size=0.30,random_state=99)
model= LogisticRegression(solver='liblinear',max_iter=1000,random_state=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc=accuracy_score(y_test, y_pred)
print("Logistic Regression model accuracy for sonar dataset(in %):", acc*100)

#using ridge regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
df=pd.read_csv("/home/ibab/Downloads/sonar.csv")
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
le=LabelEncoder()
y=le.fit_transform(y)
X_train, X_test,y_train, y_test = train_test_split(X, y,test_size=0.30,random_state=99)
ridge = Ridge(alpha=6.0)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)
acc1=accuracy_score(y_test, y_pred_binary)
print("Ridge Regression model accuracy for sonar dataset(in %):", acc1*100)

#using lasso regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
df=pd.read_csv("/home/ibab/Downloads/sonar.csv")
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
le=LabelEncoder()
y=le.fit_transform(y)
X_train, X_test,y_train, y_test = train_test_split(X, y,test_size=0.30,random_state=99)
lasso = Lasso(alpha=0.0002)
lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)
acc2=accuracy_score(y_test, y_pred_binary)
print("Lasso Regression model accuracy for sonar dataset(in %):", acc2*100)

#using california dataset
#ridge regression
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
X,y=fetch_california_housing(return_X_y=True)
X_train, X_test,y_train, y_test = train_test_split(X, y,test_size=0.30,random_state=99)
ridge = Ridge(alpha=5.0)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
r2_score = r2_score(y_test, y_pred)
print("The R^2 score for Ridge regression for california dataset is: ",r2_score)

#linear regression
from sklearn.metrics import r2_score
X,y=fetch_california_housing(return_X_y=True)
X_train, X_test,y_train, y_test = train_test_split(X, y,test_size=0.30,random_state=99)
model=LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
r2_score_LR = r2_score(y_test, y_pred)
print("The R^2 score for Linear regression for california dataset is: ",r2_score_LR)

#lasso regression
from sklearn.metrics import r2_score
X,y=fetch_california_housing(return_X_y=True)
X_train, X_test,y_train, y_test = train_test_split(X, y,test_size=0.30,random_state=99)
lasso = Lasso(alpha=0.001)
lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)
r2_score_lasso = r2_score(y_test, y_pred)
print("The R^2 score for Lasso regression for california dataset is: ",r2_score_lasso)