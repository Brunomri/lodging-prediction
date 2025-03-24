import pandas as pd
import glob
from sklearn.preprocessing import LabelEncoder

def create_dataframe():
    file_paths = glob.glob('data/*.csv')

    dfs = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)

        file_name = file_path.split('/')[-1].split('.')[0]
        city_name = file_name.split('_')[0]
        day_type = file_name.split('_')[1]

        df['city'] = city_name
        df['day_type'] = day_type

        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    return df

def clean_dataframe(df):
    return df.dropna()

def encode_variables(df):
    label_enc = LabelEncoder()
    df['room_shared'] = label_enc.fit_transform(df['room_shared'])
    df['room_private'] = label_enc.fit_transform(df['room_private'])
    df['host_is_superhost'] = label_enc.fit_transform(df['host_is_superhost'])

    df = pd.get_dummies(df, columns=["room_type"], drop_first=True)
    return df

def process_data():
    df = create_dataframe()
    df = clean_dataframe(df)
    df = encode_variables(df)
    return df