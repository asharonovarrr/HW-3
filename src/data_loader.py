import pandas as pd
import numpy as np
import os

def load_m4_data(n_series=100):
    url = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Train/Hourly-train.csv"
    data_path = "data/m4_hourly.csv"
    os.makedirs('data', exist_ok=True)

    if not os.path.exists(data_path):
        df = pd.read_csv(url)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)

    df = df.head(n_series)
    series_list = []

    for i in range(len(df)):
        y = df.iloc[i, 1:].dropna().values.astype(float)
        ds = pd.date_range(start='2020-01-01', periods=len(y), freq='H')
        series_df = pd.DataFrame({'ds': ds, 'y': y, 'unique_id': i})
        series_list.append(series_df)

    return series_list
