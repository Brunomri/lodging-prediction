from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from src.charts.plot_data import plot_scatter
from src.models.linear_regression import lr_predict

# Split data into train and test sets
def split_train_test(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
    return x_train, x_test, y_train, y_test

def evaluate(test, pred):
    print(f"MAE = {mean_absolute_error(test, pred)}")

# Run predictions using different models
def run_models(x_train, x_test, y_train, y_test):
    y_pred_lr = lr_predict(x_train, y_train, x_test)
    plot_scatter(y_pred_lr, y_test)
    evaluate(y_test, y_pred_lr)