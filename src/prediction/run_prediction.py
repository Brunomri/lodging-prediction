from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.charts.plot_data import plot_scatter
from src.models.lasso import lasso_predict
from src.models.linear_regression import lr_predict
from src.models.random_forest import rf_predict
from src.models.xgboost import xgboost_predict


# Split data into train and test sets
def split_train_test(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
    print(f"train input: {x_train.shape}, train target: {y_train.shape}, test input: {x_test.shape}, test target: {y_test.shape}")
    return x_train, x_test, y_train, y_test

def scale(x_train, x_test):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled

def evaluate(test, pred, model):
    print(f"{model} MAE = {round(mean_absolute_error(test, pred), 4)}")
    print(f"{model} MAPE = {round(mean_absolute_percentage_error(test, pred), 4)}")

# Run predictions using different models
def run_models(x_train, x_test, y_train, y_test):
    x_train_scaled, x_test_scaled = scale(x_train, x_test)
    run_linear_regression(x_train_scaled, x_test_scaled, y_train, y_test)
    run_lasso(x_train_scaled, x_test_scaled, y_train, y_test)
    run_random_forest(x_train_scaled, x_test_scaled, y_train, y_test)
    run_xgboost(x_train_scaled, x_test_scaled, y_train, y_test)

def run_linear_regression(x_train, x_test, y_train, y_test):
    print("Linear Regression")
    y_pred_lr_train, y_pred_lr_test = lr_predict(x_train, y_train, x_test)
    plot_scatter(y_pred_lr_train, y_train, "Linear Regression Train")
    plot_scatter(y_pred_lr_test, y_test, "Linear Regression Test")
    evaluate(y_train, y_pred_lr_train, "Linear Regression Train")
    evaluate(y_test, y_pred_lr_test, "Linear Regression Test")
    print("--------------------")

def run_lasso(x_train, x_test, y_train, y_test):
    print("Lasso")
    y_pred_lasso_train, y_pred_lasso_test = lasso_predict(x_train, y_train, x_test)
    plot_scatter(y_pred_lasso_train, y_train, "Lasso Train")
    plot_scatter(y_pred_lasso_test, y_test, "Lasso Test")
    evaluate(y_train, y_pred_lasso_train, "Lasso Train")
    evaluate(y_test, y_pred_lasso_test, "Lasso Test")
    print("--------------------")

def run_random_forest(x_train, x_test, y_train, y_test):
    print("RandomForest")
    y_pred_rf_train, y_pred_rf_test = rf_predict(x_train, y_train, x_test)
    plot_scatter(y_pred_rf_train, y_train, "RandomForest Train")
    plot_scatter(y_pred_rf_test, y_test, "RandomForest Test")
    evaluate(y_train, y_pred_rf_train, "RandomForest Train")
    evaluate(y_test, y_pred_rf_test, "RandomForest Test")
    print("--------------------")

def run_xgboost(x_train, x_test, y_train, y_test):
    print("XGBoost")
    y_pred_xgb_train, y_pred_xgb_test = xgboost_predict(x_train, y_train, x_test)
    plot_scatter(y_pred_xgb_train, y_train, "XGBoost Train")
    plot_scatter(y_pred_xgb_test, y_test, "XGBoost Test")
    evaluate(y_train, y_pred_xgb_train, "XGBoost Train")
    evaluate(y_test, y_pred_xgb_test, "XGBoost Test")
    print("--------------------")