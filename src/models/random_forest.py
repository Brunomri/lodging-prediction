import numpy as np
from sklearn.ensemble import RandomForestRegressor

def rf_predict(x_train, y_train, x_test):
    rf = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=5, min_samples_split=10)
    rf.fit(x_train, np.log1p(y_train))
    get_model_stats(rf)
    return np.expm1(rf.predict(x_train)), np.expm1(rf.predict(x_test))

def get_model_stats(model):
    print(model.get_params())
