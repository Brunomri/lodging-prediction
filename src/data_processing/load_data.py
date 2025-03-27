import pandas as pd
import glob
from sklearn.preprocessing import LabelEncoder

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
    # Drop all rows with missing values
    return df.dropna()

def encode_variables(df):
    # Use one-hot encoding for room_type, because it has 3 possible categories
    df = pd.get_dummies(df, columns=["room_type"], drop_first=False)
    binary_cols = ["room_shared", "room_private", "host_is_superhost"]
    df[binary_cols] = df[binary_cols].astype(int)
    return df

def process_data():
    df = create_dataframe()
    df = clean_dataframe(df)
    df = encode_variables(df)
    return df