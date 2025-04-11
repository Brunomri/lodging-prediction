from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from src.charts.plot_data import plot_scatter
from src.models.linear_regression import lr_predict

import statsmodels.api as sm

# Split data into train and test sets
def split_train_test(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
    print(f"train input: {x_train.shape}, train target: {y_train.shape}, test input: {x_test.shape}, test target: {y_test.shape}")
    return x_train, x_test, y_train, y_test

def evaluate(test, pred):
    print(f"MAE = {mean_absolute_error(test, pred)}")

def get_model_stats(x_train, y_train):
    X2 = sm.add_constant(x_train)
    model_stats = sm.OLS(y_train.values.reshape(-1, 1), X2).fit()
    print(model_stats.summary())

# Run predictions using different models
def run_models(x_train, x_test, y_train, y_test):
    get_model_stats(x_train, y_train)
    y_pred_lr = lr_predict(x_train, y_train, x_test)
    plot_scatter(y_pred_lr, y_test)
    evaluate(y_test, y_pred_lr)