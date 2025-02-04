from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test,y_train, y_test = train_test_split(X, y,test_size=0.40,random_state=99)
X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=0.20,random_state=99)
best_degree = None
best_r2 = -np.inf
for d in range(1,6):
    poly_reg_model = make_pipeline(PolynomialFeatures(degree=d, include_bias=False),StandardScaler(), LinearRegression())
    poly_reg_model.fit(X_train, y_train)
    y_pred =poly_reg_model.predict(X_val)
    r2=r2_score(y_val, y_pred)
    print(f"Linear Regression model R2_score using sklearn (degree={d}):", r2)
    if r2 > best_r2:
        best_r2 = r2
        best_degree = d
X_combined = np.vstack([X_train, X_val])
y_combined = np.hstack([y_train, y_val])
best_poly_reg_model = make_pipeline(PolynomialFeatures(degree=best_degree, include_bias=False),StandardScaler(),LinearRegression())
best_poly_reg_model.fit(X_combined, y_combined)
y_test_pred = best_poly_reg_model.predict(X_test)
r2_test = r2_score(y_test, y_test_pred)
print(f"Best model (degree={best_degree}) R² score on the test set:", r2_test)



