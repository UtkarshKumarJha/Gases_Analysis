import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def clean_column_names(df):
    df.columns = df.columns.str.strip()
    return df

def parse_dates(df):
    df['timestamp'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df = df.sort_values('timestamp')
    df = df.drop(columns=['year', 'month', 'day', 'hour'])
    return df

def enforce_hourly_index(df):
    df = df.set_index('timestamp')
    df = df.asfreq('h')
    return df