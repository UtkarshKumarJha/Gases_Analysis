import pandas as pd
import xgboost as xgb

def date_based_split(df, train_frac=0.75):
    dates = df.index.normalize().unique()
    split_idx = int(len(dates) * train_frac)
    train_dates = dates[:split_idx]
    test_dates = dates[split_idx:]
    
    train_df = df[df.index.normalize().isin(train_dates)]
    test_df = df[df.index.normalize().isin(test_dates)]
    
    return train_df, test_df

def train_xgboost(X_train, y_train):
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model