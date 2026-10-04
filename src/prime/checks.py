import pandas as pd


def find_missing_channel(df: pd.DataFrame) -> pd.DataFrame:
    """Клиенты без канала привлечения."""
    return df[df["acquisition_channel"].isna()]
