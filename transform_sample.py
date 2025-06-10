import pandas as pd
from typing import Tuple


def load_sample(server: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load sample data from the server or local path.
    Args:
        server (bool): If True, load data from the server. If False, load from local path.
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Tuple containing prices and funding rates DataFrames.
    """
    if server:
        # Load data from the server
        prices = pd.read_csv("/share/data/jfuest_crypto/sample/prices.csv")
        fundingrate = pd.read_csv("/share/data/jfuest_crypto/sample/funding_rates.csv")
    else:
        # Load data from local path
        prices = pd.read_csv("prices.csv")
        fundingrate = pd.read_csv("funding_rates.csv")
    return prices, fundingrate

def remove_unnecessary_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes cols form the df that are not in the hw2 files.
    Args:
        df (pd.DataFrame): Input DataFrame.
    Returns:
        pd.DataFrame: DataFrame with unnecessary columns removed.
    """
    cols_to_keep = ["close_time", "close", "coin"]
    df = df[cols_to_keep].copy()
    return df

def pivot_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the DataFrame to have coins as columns and open_time as index.
    Args:
        df (pd.DataFrame): Input DataFrame.
    Returns:
        pd.DataFrame: Pivoted DataFrame.
    """
    df = df.pivot(index="close_time", columns="coin", values="close")
    df.index = pd.to_datetime(df.index, unit="ms", utc=True)
    df = df.astype("float64")
    return df

def pivot_funding_rate(prices_pivoted: pd.DataFrame, funding_rates: pd.DataFrame, tolerance: str='1min') -> pd.DataFrame:
    """
    Pivot the funding rates DataFrame to have coins as columns and calc_time as index.
    Args:
        prices_pivoted (pd.DataFrame): Pivoted prices DataFrame.
        funding_rates (pd.DataFrame): Funding rates DataFrame.
        tolerance (str): Tolerance for merging funding rates with prices index, default is '1min'.
    Returns:
        pd.DataFrame: Pivoted funding rates DataFrame with an index matched to the prices.
    """
    funding_rates = funding_rates.pivot(index="calc_time", columns="coin", values="last_funding_rate")
    funding_rates.index = pd.to_datetime(funding_rates.index, unit="ms", utc=True)
    funding_rates = funding_rates.astype("Float64")
    df_reset = funding_rates.reset_index()
    final_df = pd.DataFrame(index=prices_pivoted.index)
    final_df_reset = final_df.reset_index()
    merged = pd.merge_asof(
        final_df_reset.sort_values('close_time'),
        df_reset.sort_values('calc_time'),
        left_on='close_time',
        right_on='calc_time',
        direction='nearest',
        tolerance=pd.Timedelta(tolerance)
    )
    result = merged.set_index('close_time').drop(columns=['calc_time'])
    result = result.fillna(0)
    return result

if __name__== "__main__":
    prices, funding_rates = load_sample(server=True)
    prices = remove_unnecessary_cols(prices)
    prices = pivot_df(prices)
    funding_rates = pivot_funding_rate(prices, funding_rates)
    prices.to_csv("prices.csv")
    funding_rates.to_csv("funding_rates.csv")