import numpy as np
from xgboost import XGBRegressor

def xgboost_predict(x_train, y_train, x_test):
    xgboost = XGBRegressor(
        n_estimators=3000,      # More trees for better learning with low rate
        max_depth=8,            # Deeper trees capture more complex patterns but increase overfitting
        learning_rate=0.005,    # Gradual learning in small steps
        subsample=0.6,          # Portion of data per tree
        colsample_bytree=0.6,   # Portion of features per tree
        min_child_weight=10,    # Instances per leaf splits
        random_state=0,
        reg_alpha=1.0,          # L1 regularization (like Lasso)
        reg_lambda=10.0)        # L2 regularization (like Ridge)
    xgboost.fit(x_train, np.log1p(y_train))
    get_model_stats(xgboost)
    return np.expm1(xgboost.predict(x_train)), np.expm1(xgboost.predict(x_test))

def get_model_stats(model):
    print(model.get_params())