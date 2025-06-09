import pandas as pd


def load_sample(server: bool = False) -> pd.DataFrame:
    """
    Load sample data from the server or local path.
    Args:
        server (bool): If True, load data from the server. If False, load from local path.
    Returns:
        pd.DataFrame: Loaded sample data.
    """
    if server:
        # Load data from the server
        prices = pd.read_csv("/share/data/jfuest_crypto/sample/prices.csv")
        # prices = pd.read_csv("/Users/pinkyvicky/Desktop/mse244/crypto_statarb/prices_raw.csv")
    else:
        # Load data from local path
        prices = pd.read_csv("prices.csv")
    return prices

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

if __name__== "__main__":
    prices = load_sample(server=True)
    prices = remove_unnecessary_cols(prices)
    prices = pivot_df(prices)
    prices.to_csv("prices.csv")