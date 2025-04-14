import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.charts.plot_data import plot_scatter
from src.models.lasso import lasso_predict
from src.models.linear_regression import lr_predict

# Split data into train and test sets
def split_train_test(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
    print(f"train input: {x_train.shape}, train target: {y_train.shape}, test input: {x_test.shape}, test target: {y_test.shape}")
    return x_train, x_test, y_train, y_test

def scale(x_train, x_test):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled

def evaluate(test, pred, model):
    print(f"{model} MAE = {mean_absolute_error(test, pred)}")

# Run predictions using different models
def run_models(x_train, x_test, y_train, y_test):
    x_train_scaled, x_test_scaled = scale(x_train, x_test)
    run_linear_regression(x_train_scaled, x_test_scaled, y_train, y_test)
    run_lasso(x_train_scaled, x_test_scaled, y_train, y_test)

def run_linear_regression(x_train, x_test, y_train, y_test):
    print("Linear Regression")
    y_pred_lr = lr_predict(x_train, y_train, x_test)
    plot_scatter(y_pred_lr, y_test, "Linear Regression")
    evaluate(y_test, y_pred_lr, "Linear Regression")
    print("--------------------")

def run_lasso(x_train, x_test, y_train, y_test):
    print("Lasso")
    y_pred_lasso = lasso_predict(x_train, y_train, x_test)
    plot_scatter(y_pred_lasso, y_test, "Lasso")
    evaluate(y_test, y_pred_lasso, "Lasso")
    print("--------------------")