import numpy as np
from sklearn.ensemble import RandomForestRegressor

def rf_predict(x_train, y_train, x_test):
    rf = RandomForestRegressor()
    rf.fit(x_train, np.log1p(y_train))
    get_model_stats(rf)
    return np.expm1(rf.predict(x_train)), np.expm1(rf.predict(x_test))

def get_model_stats(model):
    print(model.get_params())
