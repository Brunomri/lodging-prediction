import numpy as np
from sklearn.linear_model import LinearRegression

import statsmodels.api as sm

def lr_predict(x_train, y_train, x_test):
    lr = LinearRegression()
    lr.fit(x_train, np.log1p(y_train))
    get_model_stats(x_train, y_train)
    return np.expm1(lr.predict(x_test))

def get_model_stats(x_train, y_train):
    X2 = sm.add_constant(x_train)
    model_stats = sm.OLS(y_train.values.reshape(-1, 1), X2).fit()
    print(model_stats.summary())
