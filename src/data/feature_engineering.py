import numpy as np

def create_time_features(df, timestamp_col='Timestamp'):
    df = df.copy()
    # Hour
    df['hour'] = df.index.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    # Day of week
    df['dayofweek'] = df.index.day_of_week
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    # Month
    df['month'] = df.index.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    # Weekend indicator
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df = df.drop(columns=['hour', 'dayofweek', 'month'])
    return df

def create_wind_features(df, wd_col='WD (deg)'):
    df = df.copy()
    radians = np.deg2rad(df[wd_col])
    df['wd_sin'] = np.sin(radians)
    df['wd_cos'] = np.cos(radians)
    df = df.drop(columns=[wd_col])
    return df

def add_features(df):
    df = create_time_features(df)
    df = create_wind_features(df)
    return df

def feature_extraction(df, station_name):
    df = df.copy()
    df = add_features(df)
    print(f"\n{station_name} Feature engineered successfully.")
    return df