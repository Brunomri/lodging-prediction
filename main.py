from src.data_processing.load_data import process_data, split_input_target
from src.prediction.run_prediction import split_train_test, run_models
from src.charts.plot_data import plot_heatmap, plot_histogram

def main():
    dataframe = process_data()
    x, y = split_input_target(dataframe)
    plot_heatmap(dataframe.corr())
    plot_histogram(dataframe, 'realSum')
    x_train, x_test, y_train, y_test = split_train_test(x, y)
    run_models(x_train, x_test, y_train, y_test)

if __name__ == "__main__":
    main()