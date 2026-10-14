import pandas as pd
from pathlib import Path

def load_excel(path, date_col="Timestamp", usecols=None):

    if not Path(path).exists():
        raise FileNotFoundError(f"File not found at {path}")

    df = pd.read_excel(path)
    df[date_col] = pd.to_datetime(df[date_col], format="%Y-%m-%d")
    df = df.set_index(date_col)

    df.columns = df.columns.str.replace("Â", "", regex=False)

    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype("float32")

    if usecols:
        return df[usecols]
    return df

if __name__ == "__main__":
    df = load_excel("path.xlsx")

