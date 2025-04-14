import numpy as np
from sklearn.linear_model import LassoCV


def lasso_predict(x_train, y_train, x_test):
    lasso = LassoCV(cv=5, max_iter=10000)
    lasso.fit(x_train, np.log1p(y_train))
    print(f"Alpha: {lasso.alpha_}")
    print("Lasso coefficients:", lasso.coef_)
    return np.expm1(lasso.predict(x_test))