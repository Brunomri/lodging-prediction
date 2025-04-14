import pandas as pd
import glob

from pandas.core.dtypes.common import is_numeric_dtype

from src.charts.plot_data import plot_heatmap, plot_distribution


def create_dataframe():
    # Build file paths for all csv files
    file_paths = glob.glob('data/*.csv')

    dfs = []
    # Combine all csv files in a single data frame
    for file_path in file_paths:
        df = pd.read_csv(file_path)

        file_name = file_path.split('/')[-1].split('.')[0]
        city_name = file_name.split('_')[0]
        day_type = file_name.split('_')[1]

        # Add city and day_type (weekday or weekend) as new columns
        df['city'] = city_name
        df['day_type'] = day_type

        dfs.append(df)

    # Concatenate data frames from all csv files
    df = pd.concat(dfs, ignore_index=True)
    return df

def clean_dataframe(df):
    # Remove first column with row index and drop all rows with missing values
    df.drop(columns=['room_shared', 'room_private'], inplace=True)
    return df.iloc[:, 1:].dropna()

def encode_variables(df):
    # Use one-hot encoding for room_type, city and day type because they are categorical data
    df = pd.get_dummies(df, columns=["room_type"], drop_first=False, dtype=int)
    df = pd.get_dummies(df, columns=["city"], drop_first=False, dtype=int)
    df = pd.get_dummies(df, columns=["day_type"], drop_first=False, dtype=int)
    # Convert columns with true or false values to 1 or 0
    binary_cols = ["host_is_superhost"]
    df[binary_cols] = df[binary_cols].astype(int)
    return df

def split_input_target(df):
    # Split data into input (x) and target (y)
    x = df.drop('realSum', axis=1)
    y = df['realSum']
    return x, y

def handle_outliers(df, lower_q=0.03, upper_q=0.97):
    # For every column of the dataframe, drop rows with values
    # below the lower percentile or above the upper percentile
    for col in df.columns:
        if is_numeric_dtype(df[col]) and df[col].nunique() > 2:
            lower = df[col].quantile(lower_q)
            upper = df[col].quantile(upper_q)
            df = df[df[col].between(lower, upper)]
            plot_distribution(df, col, f"{col} Distribution")

    return df

def process_data():
    df = create_dataframe()
    df = clean_dataframe(df)
    df = encode_variables(df)
    df = handle_outliers(df)
    print(df.info())
    print(df.describe())
    plot_heatmap(df.corr(), "Correlation Heatmap")
    return split_input_target(df)