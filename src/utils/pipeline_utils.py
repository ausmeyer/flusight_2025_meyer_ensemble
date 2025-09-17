import os
import pandas as pd
from typing import Tuple


def find_latest_imputed(base_dir: str = "data/imputed_sets") -> str:
    """Return path to latest imputed_and_stitched_hosp_*.csv in base_dir."""
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(base_dir)
    files = [f for f in os.listdir(base_dir) if f.startswith("imputed_and_stitched_hosp_") and f.endswith('.csv')]
    if not files:
        raise FileNotFoundError("No imputed_and_stitched_hosp_*.csv found")
    files.sort()
    return os.path.join(base_dir, files[-1])


def last_and_cutoff_dates(data_file: str, weeks: int = 8) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Compute last date (max date in file) and cutoff = last - weeks.
    Returns (last_date, cutoff_date) as pandas Timestamps.
    """
    df = pd.read_csv(data_file)
    if 'date' not in df.columns:
        raise ValueError("date column not found in data file")
    df['date'] = pd.to_datetime(df['date'])
    last_date = pd.to_datetime(df['date'].max())
    cutoff_date = last_date - pd.Timedelta(weeks=weeks)
    return last_date, cutoff_date

