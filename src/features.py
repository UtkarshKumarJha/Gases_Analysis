import numpy as np
import pandas as pd

def add_time_features(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['month'] = df.index.month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    return df

def add_lag_features(df, targets=['O3_target', 'NO2_target'], lags=[1, 24]):
    df = df.copy()
    for target in targets:
        for lag in lags:
            col_name = f'{target}_lag_{lag}'
            df[col_name] = df[target].shift(lag)
    return df

def fill_missing_lags(df, value=-999):
    lag_cols = [c for c in df.columns if 'lag' in c]
    df[lag_cols] = df[lag_cols].fillna(value)
    return df