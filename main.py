from src.data_processing.load_data import process_data
from src.prediction.run_prediction import split_train_test, run_models


def main():
    x, y = process_data()
    x_train, x_test, y_train, y_test = split_train_test(x, y)
    run_models(x_train, x_test, y_train, y_test)

if __name__ == "__main__":
    main()