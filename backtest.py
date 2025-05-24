import hashlib
import json
import os
import pickle
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import arch.unitroot
import matplotlib.pyplot as plt
import matplotlib.ticker as mpl_ticker
import numpy as np
import pandas as pd
import petname
import quantstats_lumi as qs
import seaborn as sns
import statsmodels.api as sm
from IPython.display import display
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf



def load_data(
    config: dict, verbose: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads and processes the Nikkei 225 data.
    
    Parameters:
    -----------
    config: dict
        Configuration parameters, containing NIKKEI_CSV_PATH, the filepath to the Nikkei 225 CSV file
    verbose: bool, optional
        Whether to print verbose output
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Tuple containing the prices, returns, tickers, and metadata DataFrames
    """
    # Configure pandas display options (feel free to change these)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.max_rows', 20)
    pd.set_option('display.precision', 4)
 
    # Load the data
    prices = pd.read_csv(config['NIKKEI_CSV_PATH'], low_memory=False)
    if verbose:
        print("Price df shape at load:", prices.shape)
 
    # Slice the prices to only view data up to and including 2017
    first_2018_idx = prices[prices["Ticker"].str.contains("2018", na=False)].index[0].item()
    prices = prices.iloc[:first_2018_idx]
    if verbose:
        print("Price df shape after slicing time axis:", prices.shape)
 
    # Drop columns containing only NaNs (not considering the metadata rows)
    # These correspond to equities which only come into existence post-2017
    prices = prices.loc[:, prices.isna().sum() < prices.shape[0] - 3]
    if verbose:
        print("Price df shape after removing future asset columns:", prices.shape)
 
    # Extract the metadata: industrial classification, sector, and company names
    metadata = pd.DataFrame(prices.iloc[:3])
    metadata = metadata.T
    metadata.columns = metadata.iloc[0]
    metadata = metadata.iloc[1:]
    metadata.rename(columns={"Nikkei Industrial Classification": "Industry"}, inplace=True)
 
    # Drop the metadata rows and process date
    prices = prices.iloc[3:]
    prices.rename(columns={'Ticker':'Date'}, inplace=True)
    prices['Date'] = pd.to_datetime(prices['Date'])
    prices.set_index('Date', inplace=True, drop=True)
    prices = prices.astype(float)
    tickers = prices.columns
 
    # Calculate returns
    returns = prices.pct_change(fill_method=None)
    # Set initial return to zero
    returns.iloc[0] = 0
    
    if verbose:
        print("\nPrices head:")
        display(prices.head())
        print("\nMetadata head:")
        display(metadata.head())
 
        # Plot NaNs
        plt.imshow(prices.isna(), aspect='auto', cmap='viridis', interpolation=None)
        plt.xlabel('Stock')
        plt.ylabel('Date')
        plt.title('Missing Data in Nikkei 225 Prices')
        plt.grid(False)
        plt.show()
    
    return prices, returns, tickers, metadata

# Liquidity measures (the lower, the more liquid the coin)
# TODO: change from resampling once an hour to having hourly lookbacks each minute
# def hourly_amihud(df):
#     """Takes a DataFrame and computes the Amihud illiquidity measure for each coin.

#     Args:
#         df (pd.DataFrame): A prices df in the non-pivoted format.

#     Returns:
#         pd.DataFrane: A df with an added Amihud illiquidity measure column.
#     """
#     df['return'] = df.groupby('coin')['close'].pct_change()
#     df['amihud'] = np.abs(df['return']) / df['quote_volume'].replace(0, np.nan)
    
#     amihud_records = df.groupby(['coin', pd.Grouper(key='close_time', freq='h')])['amihud'].mean().reset_index()
#     return amihud_records

# def hourly_roll(df):
#     """
#     Computes the Roll serial cov estimator for each coin in the DataFrame.

#     Args:
#         df (pd.DataFrame): The prices DataFrame in non-pivoted format.

#     Returns:
#         pd.DataFrame: The same DataFrame with the Roll estimator for each coin added.
#     """
#     roll_records = []
#     for (coin, time), group in df.groupby(['coin', pd.Grouper(key='close_time', freq='h')]):
#         if group['close'].count() < 3:
#             roll_val = np.nan
#         else:
#             price_diff = group['close'].diff().dropna()
#             if len(price_diff) >= 2:
#                 cov = np.cov(price_diff[1:], price_diff[:-1])[0, 1]
#                 roll_val = 2 * np.sqrt(-cov) if cov < 0 else np.nan
#             else:
#                 roll_val = np.nan
#         roll_records.append({'close_time': time, 'coin': coin, 'roll_estimate': roll_val})
#     return pd.DataFrame(roll_records)

# def hourly_kyle(df):
#     """
#     Computes the Kyle illiquidity measure for each coin in the DataFrame.

#     Args:
#         df (pd.DataFrame): The prices DataFrame in non-pivoted format.

#     Returns:
#         pd.DataFrane: The same DataFrame with the Kyle estimator for each coin added.
#     """
#     df['return'] = df.groupby('coin')['close'].pct_change()
#     df['signed_volume'] = df['taker_buy_volume'] - (df['volume'] - df['taker_buy_volume'])

#     kyle_records = []
#     for (coin, time), group in df.groupby(['coin', pd.Grouper(key='close_time', freq='h')]):
#         group = group.dropna(subset=['return', 'signed_volume'])
#         if len(group) < 2:
#             kyle_val = np.nan
#         else:
#             cov = np.cov(group['return'], group['signed_volume'])[0, 1]
#             var = np.var(group['signed_volume'])
#             kyle_val = cov / var if var > 0 else np.nan
#         kyle_records.append({'close_time': time, 'coin': coin, 'kyle_lambda': kyle_val})
#     return pd.DataFrame(kyle_records)

def hourly_amihud(df: pd.DataFrame) -> pd.DataFrame:
    """
    60-minute rolling Amihud illiquidity at every timestamp.

    Returns
    -------
    DataFrame with columns ['close_time', 'coin', 'amihud']
    """
    df['amihud_raw'] = np.abs(df['return']) / df['quote_volume'].replace(0, np.nan)

    out = []
    for coin, grp in df.groupby('coin'):
        grp = grp.set_index('close_time')
        grp['amihud'] = grp['amihud_raw'].rolling('60min', min_periods=1).mean()
        out.append(grp[['amihud']].reset_index().assign(coin=coin))
    return pd.concat(out, ignore_index=True)


def hourly_roll(df: pd.DataFrame) -> pd.DataFrame:
    """
    60-minute rolling Roll (1984) spread estimate at every timestamp.

    Returns
    -------
    DataFrame with columns ['close_time', 'coin', 'roll_estimate']
    """
    df = df.copy()
    df.sort_values(['coin', 'close_time'], inplace=True)

    out = []
    for coin, grp in df.groupby('coin'):
        grp = grp.set_index('close_time')
        price_diff = grp['close'].diff()
        # custom rolling covariance between consecutive price changes
        roll_cov = price_diff.rolling('60min', min_periods=3).apply(
            lambda x: np.cov(x[1:], x[:-1])[0, 1] if len(x) >= 3 else np.nan,
            raw=False
        )
        roll_est = np.where(roll_cov < 0, 2.0 * np.sqrt(-roll_cov), np.nan)
        out.append(
            pd.DataFrame(
                {
                    'close_time': roll_cov.index,
                    'roll_estimate': roll_est,
                    'coin': coin,
                }
            )
        )
    return pd.concat(out, ignore_index=True)


def hourly_kyle(df: pd.DataFrame) -> pd.DataFrame:
    """
    60-minute rolling Kyle lambda at every timestamp.

    Returns
    -------
    DataFrame with columns ['close_time', 'coin', 'kyle_lambda']
    """
    df = df.copy()
    df.sort_values(['coin', 'close_time'], inplace=True)
    df['return'] = df.groupby('coin')['close'].pct_change()
    df['signed_volume'] = df['taker_buy_volume'] - (df['volume'] - df['taker_buy_volume'])

    out = []
    for coin, grp in df.groupby('coin'):
        grp = grp.set_index('close_time')
        cov_r_sv = grp['return'].rolling('60min', min_periods=2).cov(grp['signed_volume'])
        var_sv = grp['signed_volume'].rolling('60min', min_periods=2).var()
        kyle_lambda = cov_r_sv / var_sv
        out.append(
            pd.DataFrame(
                {
                    'close_time': cov_r_sv.index,
                    'kyle_lambda': kyle_lambda,
                    'coin': coin,
                }
            )
        )
    return pd.concat(out, ignore_index=True)


def load_data_crypto(
    config: dict, verbose: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads and preprocesses the cryptocurrency data.

    Args:
        config (dict): config, containing the path to the CSV file.
        verbose (bool, optional): Whether  or not to output details about the data. 
        Defaults to False.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: prices, returns, tickers.
    """
    # Configure pandas display options (feel free to change these)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.max_rows', 20)
    pd.set_option('display.precision', 4)
    prices = pd.read_csv(config['CRYPTO_CSV_PATH'], low_memory=False, index_col=0)
    if verbose:
        print("Price df shape at load:", prices.shape)
    # set open_time as index 
    # Calculate returns
    returns = prices.pct_change(fill_method=None)
    # Set initial return to zero
    returns.iloc[0] = 0
    prices.index = pd.to_datetime(prices.index)
    returns.index = pd.to_datetime(returns.index)
    # price df in raw format including all cols
    prices_raw = pd.read_csv(config["SAMPLE_PRICES_PATH"])
    # Convert timestamps to datetime
    prices_raw['open_time'] = pd.to_datetime(prices_raw['open_time'], unit='ms', utc=True)
    prices_raw['close_time'] = pd.to_datetime(prices_raw['close_time'], unit='ms', utc=True)
    prices_raw = prices_raw.sort_values(['coin', 'close_time'])
    prices_raw['return'] = prices_raw.groupby('coin')['close'].pct_change()
    amihud_hourly_df = hourly_amihud(prices_raw)
    #roll_hourly_df = hourly_roll(prices_raw)
    #kyle_hourly_df = hourly_kyle(prices_raw)

    # %%
    # Pivot all to common format
    amihud_pivot = amihud_hourly_df.pivot(index='close_time', columns='coin', values='amihud')
    # roll_pivot = roll_hourly_df.pivot(index='close_time', columns='coin', values='roll_estimate')
    # kyle_pivot = kyle_hourly_df.pivot(index='close_time', columns='coin', values='kyle_lambda')
    
    
    if verbose:
        print("\nPrices head:")
        display(prices.head())
 
        # Plot NaNs
        plt.imshow(prices.isna(), aspect='auto', cmap='viridis', interpolation=None)
        plt.xlabel('Coin')
        plt.ylabel('Date')
        plt.title('Missing Data in Binance Crypto Prices')
        plt.grid(False)
        plt.show()
    
    
    tickers = prices.columns
    assert amihud_pivot.columns.equals(tickers), "Amihud columns do not match prices columns"
    # assert roll_pivot.columns.equals(tickers), "Roll columns do not match prices columns"
    # assert kyle_pivot.columns.equals(tickers), "Kyle columns do not match prices columns"
    
    return prices, returns, tickers, amihud_pivot, # roll_pivot, kyle_pivot


def plot_asset_with_max_return(returns, prices, max_rank=0):
    """
    Plots the asset with the highest return in the returns dataframe.
    
    Parameters:
    -----------
    returns: pd.DataFrame
        Dataframe of returns with tickers as columns and dates as index
    prices: pd.DataFrame
        Dataframe of prices with tickers as columns and dates as index
    max_rank: int
        Rank of the asset to plot
        
    Returns:
    --------
    None. Plots the asset with the highest return in the returns dataframe
    """
    # find asset index with the specified rank of return
    max_returns = pd.DataFrame({'max_return': returns.max(axis=1), 'idx': returns.idxmax(axis=1)}, columns=['max_return', 'idx'])
    asset_idx = max_returns.sort_values(by='max_return', ascending=False).iloc[max_rank]['idx']
    max_return_date = max_returns.sort_values(by='max_return', ascending=False).index[max_rank]

    # plot this asset
    prices[asset_idx].plot()
    # plot the date of the max return
    plt.axvline(max_return_date, color='r', linestyle='--')
    # plot text of the price of the asset at the max return date
    # Add a text box with information on the side of the chart
    info_text = (
        f'Rank: {max_rank}\n'
        f'Ticker: {asset_idx}\n'
        f'Date: {max_return_date.strftime("%Y-%m-%d")}\n'
        f'Price t:    {prices[asset_idx][max_return_date]:.2f}\n'
    )
    
    try:
        # Find the index position of max_return_date in the prices dataframe
        date_idx = prices.index.get_loc(max_return_date)
        if date_idx > 0:  # Make sure there is a previous trading day
            prev_date = prices.index[date_idx - 1]
            info_text += f'Price t-1: {prices[asset_idx][prev_date]:.2f}\n'
        else:
            info_text += 'Price t-1: N/A\n'
    except:
        info_text += 'Price t-1: N/A\n'
        
    info_text += f'Return: {returns[asset_idx][max_return_date]:.1%}'
    
    # Position the text box outside the main plot area
    plt.figtext(0.85, 0.5, info_text, bbox=dict(facecolor='white', alpha=0.8), 
                verticalalignment='center', color='r')
    plt.axhline(0, color='r', linestyle='--')
    plt.show()


def select_asset_universe(
    prices: pd.DataFrame, 
    returns: pd.DataFrame, 
    date: pd.Timestamp, 
    config: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Index]:
    """
    Reduces the cross-section to only those stocks which were members of the asset universe at the given time
    with sufficient non-missing data and valid returns over the lookback period.
    
    Parameters:
    -----------
    prices: pd.DataFrame
        Dataframe of stock prices with dates as index and tickers as columns
    returns: pd.DataFrame
        Dataframe of stock returns with dates as index and tickers as columns
    date: pd.Timestamp
        The reference date
    config: dict
        Configuration parameters, including 
        - FORMATION_PERIOD: int, the number of trading days to use for forming pairs, and thus also 
            for checking data availability, e.g. 252 days
        - FILTER_MAX_ABS_RETURN: float, the maximum absolute return allowed, e.g. 0.5
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Index]
        Tuple containing the selected historical prices, returns, and valid stocks
    """
    # grab the past lookback_period days ending at but not including date
    lookback_period = config['FORMATION_PERIOD'] # num of days to use
    end_idx = prices.index.get_loc(date)
    start_idx = end_idx - lookback_period
    # get the range of dates over lookback_period
    lookback_idx = prices.index[start_idx:end_idx]

    # filter out stocks with any missing prices in lookback period
    historical_prices = prices.loc[lookback_idx]
    prices_no_nan = ~historical_prices.isna().any()

    # filter out stocks with crazy returns
    max_return = config['FILTER_MAX_ABS_RETURN']
    historical_returns = returns.loc[lookback_idx]
    valid_returns = (historical_returns.abs() <= max_return).all()

    # extract only stocks that fulfill both filter conditions
    valid_stocks = prices.columns[prices_no_nan & valid_returns]
    return historical_prices[valid_stocks], historical_returns[valid_stocks], valid_stocks


def select_asset_universe_crypto(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    amihud: pd.DataFrame,
    roll: pd.DataFrame,
    kyle: pd.DataFrame,
    date: pd.Timestamp,
    lookback_period:int,
    criterion: str,
    max_rank: int = 50,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Index]:
    """
    Reduces the cross-section to only those stocks which were members of the asset universe at the given time
    with sufficient non-missing data and valid returns over the lookback period. Also filters out coins with
    insufficient liquidity over the lookback period, as defined by the config.

    Args:
        prices (pd.DataFrame): Minute-level close prices of the coins
        returns (pd.DataFrame): Minute-level returns of the coins
        amihud (pd.DataFrame): Amihud illiquidity measure
        roll (pd.DataFrame): Roll serial cov estimator
        kyle (pd.DataFrame): Kyle illiquidity measure
        date (pd.Timestamp): The date at which to select the asset universe
        lookback_period (int): The number of minutes to look back for filtering
        criterion (str): The liquidity measure to use for filtering ('amihud', 'roll', or 'kyle')

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Index]: Prices, returns, and valid coins
    """
    # get the index of the date closest to the given date
    lookback_end_idx = prices.index.get_loc(date)
    lookback_start_idx = lookback_end_idx - lookback_period
    liquidity_window = prices.index[lookback_start_idx:lookback_end_idx]
    
    # filter out coins with insufficient liquidity
    if criterion == "amihud":
        amihud_window = amihud.loc[liquidity_window]
        ranks = amihud_window.rank(axis=1, method="min", ascending=True)
        valid_coins = (ranks <= max_rank).all(axis=0) & ranks.notna().all(axis=0)
        ranks_filtered = ranks.loc[:, valid_coins]
        valid_coins = ranks_filtered.columns
    
    elif criterion == "roll":
        roll_window = roll.loc[liquidity_window]
        ranks = roll_window.rank(axis=1, method="min", ascending=True)
        valid_coins = (ranks <= max_rank).all(axis=0) & ranks.notna().all(axis=0)
        ranks_filtered = ranks.loc[:, valid_coins]
        valid_coins = ranks_filtered.columns
    elif criterion == "kyle":
        kyle_window = kyle.loc[liquidity_window]
        ranks = kyle_window.rank(axis=1, method="min", ascending=True)
        valid_coins = (ranks <= max_rank).all(axis=0) & ranks.notna().all(axis=0)
        ranks_filtered = ranks.loc[:, valid_coins]
        valid_coins = ranks_filtered.columns
    else:
        raise ValueError(f"Unknown liquidity measure: {criterion}. Valid options are 'amihud', 'roll', or 'kyle'.")
    
    return prices[valid_coins], returns[valid_coins], valid_coins
        
        
def form_pairs(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Computes the distance between all pairs of stocks based on their normalized price series
    and returns information on the top `NUM_PAIRS` pairs with lowest distances.
    
    Parameters:
    -----------
    prices: pd.DataFrame
        Dataframe of stock prices with dates as index and tickers as columns
    config: dict
        Configuration parameters, containing the key `NUM_PAIRS`, with a value such as 20
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with top `NUM_PAIRS` pairs sorted by distance
    """
    # Normalize the prices to start at 1 for each column
    filtered_prices = (prices / prices.iloc[0]).copy()
    # Calculate the distance matrix
    distances = {}
    for i in range(len(filtered_prices.columns)):
        for j in range(i + 1, len(filtered_prices.columns)):
            stock1 = filtered_prices.iloc[:, i]
            stock2 = filtered_prices.iloc[:, j]
            if config['DISTANCE_METRIC'] == 'ssd':
                distance = np.sum((stock1 - stock2) ** 2)
            elif config['DISTANCE_METRIC'] == 'correlation':
                distance = 1 - np.corrcoef(stock1, stock2)[0, 1]
            else:
                raise ValueError(f"Unknown distance metric: {config['DISTANCE_METRIC']}")
            distances[(filtered_prices.columns[i], filtered_prices.columns[j])] = distance
    # get the top N_PAIRS pairs
    pairs = sorted(distances.items(), key=lambda x: x[1])[:config['NUM_PAIRS']]
    # create a DataFrame to hold the pairs
    pairs_df = pd.DataFrame(columns=["stock1", "stock2", f"distance"])
    for i, ((stock1, stock2), distance) in enumerate(pairs):
        pairs_df.loc[i] = [stock1, stock2, distance]
    return pairs_df

def _beta_ols(x: pd.Series, y: pd.Series) -> float:
    """
    Hedge ratio from OLS regression of y on x:
        y_t = β · x_t + ε_t
    """
    x = x.values.reshape(-1, 1)
    model = sm.OLS(y.values, sm.add_constant(x)).fit()
    return float(model.params[1])


def _beta_median(x: pd.Series, y: pd.Series) -> float:
    """
    Extra-credit hedge-ratio estimator:
    use the **median** price ratio rather than the mean / OLS—
    more robust to outliers and clearly different from OLS.
    """
    return float(np.median(y / x))  


def estimate_hedge_ratio(prices: pd.DataFrame, pairs: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Estimates the betas and cointegrating relationships between the top K pairs and returns this information.
    
    Parameters:
    -----------
    prices: pd.DataFrame
        Dataframe of stock prices with dates as index and tickers as columns
    pairs: pd.DataFrame
        Dataframe with pairs information from form_pairs()
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with betas and cointegration information for each pair
    """
    method = config.get("HEDGE_RATIO_METHOD", "ols").lower()
    thresh = config.get("COINT_THRESHOLD", 0.05)

    # Pick the appropriate estimator once
    if method == "ols":
        beta_fn = _beta_ols
    elif method == "unit":
        beta_fn = lambda x, y: 1.0
    elif method == "median":           # ← extra‑credit method
        beta_fn = _beta_median
    else:
        raise ValueError(
            f"Unknown HEDGE_RATIO_METHOD '{method}'. "
            "Valid choices are 'ols', 'unit', or 'median'."
        )

    results = []

    for _, row in pairs.iterrows():
        s1, s2 = row["stock1"], row["stock2"]
        price1, price2 = prices[[s1, s2]].dropna().T.values

        # Skip if insufficient overlapping history
        if price1.size < 30:   # arbitrary “enough data” rule of thumb
            continue

        beta = beta_fn(price1, price2)
        # Keep β positive for reporting but retain its sign in the spread
        beta_sign = np.sign(beta) if beta != 0 else 1.0
        beta_abs  = abs(beta)

        spread = price2 - beta * price1

        adf_res = arch.unitroot.ADF(spread)
        adf_stat, pval = adf_res.stat, adf_res.pvalue
        is_coint = pval <= thresh
        if is_coint:
            results.append(
                {
                    "stock1": s1,
                    "stock2": s2,
                    "hedge_ratio": beta_abs,
                    "distance": row["distance"],
                    "adf_stat": adf_stat,
                    "adf_pvalue": pval,
                    "is_cointegrated": is_coint,
                }
            )

    results_df = pd.DataFrame(results)
    return results_df


def compute_signal(prices: pd.DataFrame, hedge_ratios: pd.DataFrame, config: dict) -> dict:
    """
    Computes trading signals for each spread based on z-score.
    
    Parameters:
    -----------
    prices: pd.DataFrame
        Dataframe of stock prices with dates as index and tickers as columns
    hedge_ratios: pd.DataFrame
        Dataframe with hedge ratios and cointegration information from estimate_hedge_ratio
    config: dict
        Configuration parameters, containing: e.g. signal threshold Z_THRESHOLD, such as 2.0
        
    Returns:
    --------
    dict
        Dictionary with signals for each pair
    """
    signals = {}

    for index, row in hedge_ratios.iterrows():
        price_series_1 = prices[row["stock1"]]
        price_series_2 = prices[row["stock2"]]
        hedge_ratio = row["hedge_ratio"]
        spread = price_series_2 - hedge_ratio * price_series_1
        spread_rolling_mean = spread.rolling(window=config['ESTIMATION_PERIOD']).mean()
        spread_rolling_std = spread.rolling(window=config['ESTIMATION_PERIOD']).std()
        z_score = (spread - spread_rolling_mean) / spread_rolling_std
        signal_df = pd.DataFrame(index=prices.index)
        signal_df["signal"] = 0
        s1_prices_norm = price_series_1 / price_series_1.iloc[0]
        s2_prices_norm = price_series_2 / price_series_2.iloc[0]
        current_state = 0 # 0 for no position, 1 for long (short stock 1 long stock 2), -1 for short (long stock 1 short stock 2)
        for index_inner, row_alt in signal_df.iterrows():
            match current_state:
                case 0: # We previously had no position
                    if z_score[index_inner] > config['Z_THRESHOLD']:
                        signal_df.at[index_inner, 'signal'] = -1
                        current_state = -1
                    elif z_score[index_inner] < -config['Z_THRESHOLD']:
                        signal_df.at[index_inner, 'signal'] = 1
                        current_state = 1
                    else:
                        signal_df.at[index_inner, 'signal'] = 0
                case 1: # We previously had a long position
                    # close long position when normalized prices cross again
                    if s2_prices_norm[index_inner] >= s1_prices_norm[index_inner]:
                        signal_df.at[index_inner, 'signal'] = 0
                        current_state = 0
                    else:
                        signal_df.at[index_inner, 'signal'] = 1
                case -1:
                    # close short position when normalized prices cross again
                    if s2_prices_norm[index_inner] <= s1_prices_norm[index_inner]:
                        signal_df.at[index_inner, 'signal'] = 0
                        current_state = 0
                    else:
                        signal_df.at[index_inner, 'signal'] = -1
                case _:
                    raise ValueError(f"Unknown state: {current_state}")
        signals[f"{row['stock1']}_{row['stock2']}"] = {
            "stock1": row["stock1"],
            "stock2": row["stock2"],
            "hedge_ratio": hedge_ratio,
            "signal": signal_df["signal"],
            "z_score": z_score,
        }
    return signals


def allocate_positions(
    signals: dict,
    config: dict,
    prices: pd.DataFrame,
    date: pd.Timestamp,
    portfolio_cash: float,
) -> dict:
    """
    Simple position sizing that allocates cash equally across all active pairs.
    
    Parameters:
    -----------
    signals: dict
        Dictionary with signal information for each pair
    config: dict
        Configuration parameters, containing MAX_LEVERAGE
    prices: pd.DataFrame
        Dataframe of stock prices with dates as index and tickers as columns
    date: pd.Timestamp
        Current date for price reference
    portfolio_cash: float
        Available cash in the portfolio
    
    Returns:
    --------
    dict
        Dictionary with position information for each stock
    """
    positions = {}
    active_pairs = {}
    if date not in prices.index:
        return positions
    if portfolio_cash is None or portfolio_cash <= 0:
        return positions
    for key, value in signals.items():
        stock1 = value["stock1"]
        stock2 = value["stock2"]
        try:
            price1 = prices[stock1].loc[date]
            price2 = prices[stock2].loc[date]
        except KeyError:
            continue
        
        signal = value["signal"].shift(1).loc[date]
        
        if signal is None or np.isnan(signal) or signal == 0:
            continue
        
        active_pairs[key] = {
            "stock1": stock1,
            "stock2": stock2,
            "hedge_ratio": value["hedge_ratio"],
            "signal": signal,
            "price1": price1,
            "price2": price2,
        }
    if len(active_pairs) == 0:
        return positions
    # Calculate the position size for each pair
    capital_per_pair = (portfolio_cash * config["MAX_LEVERAGE"]) / len(active_pairs)
    for pair, info in active_pairs.items():
        stock1 = info["stock1"]
        stock2 = info["stock2"]
        if positions.get(stock1, None) is None:
            positions[stock1] = 0
        if positions.get(stock2, None) is None:
            positions[stock2] = 0
        notional_ratio = (info["hedge_ratio"] * info["price2"]) / info["price1"]
        shares1 = capital_per_pair / (info["price1"] * (1 + notional_ratio))
        shares2 = shares1 * info["hedge_ratio"]
        signal_value = info["signal"]
        positions[stock1] += -signal_value * shares1
        positions[stock2] += signal_value * shares2
    return positions


def update_portfolio(
    current_portfolio: dict,
    target_positions: dict,
    prices: pd.DataFrame,
    date: pd.Timestamp,
    config: dict,
) -> dict:
    """
    Updates the portfolio based on target positions, accounting for transaction costs.
    
    Parameters:
    -----------
    current_portfolio: dict
        Current portfolio positions and cash
    target_positions: dict
        Target positions for each stock, in number of shares
    prices: pd.DataFrame
        Dataframe of stock prices with dates as index and tickers as columns
    date: pd.Timestamp
        Current date
    config: dict
        Configuration parameters
        
    Returns:
    --------
    dict
        Dictionary with updated portfolio and performance metrics
        - portfolio: Updated portfolio
        - transaction_cost: Transaction costs
        - turnover: Turnover rate
        - portfolio_value_before_costs: Portfolio value before costs
        - portfolio_value_after_txn_costs: Portfolio value after txn costs
        - portfolio_value_after_costs: Portfolio value after all costs
        - long_exposure: Long exposure
        - short_exposure: Short exposure
        - leverage: Leverage ratio (total absolute exposure / portfolio value)
        - active_positions: Number of active positions
        - max_position_size: Maximum position size as percentage of portfolio
    """
    # Initialize new positions dictionary
    new_positions = {}
    new_cash = current_portfolio.get('cash', config['INITIAL_CASH'])
    
    # Get current prices
    current_prices = prices.loc[date]
    
    # Calculate transaction costs
    total_transaction_cost_pct = 0
    total_transaction_cost_amount = 0
    total_turnover = 0
    
    # Update positions for each stock
    for stock, target_position in target_positions.items():
        # Get target position for current date
        target_shares = target_position  # Now we expect target_position to be a simple number
        
        # Get current position (default to 0 if not already in portfolio)
        current_shares = current_portfolio.get(stock, 0)
        
        # Calculate the change in position
        shares_diff = target_shares - current_shares
        
        # Calculate the transaction cost if there's a change in position
        if shares_diff != 0:
            price = current_prices[stock]
            
            # Calculate the value of the transaction
            transaction_value = abs(shares_diff * price)
            
            # Calculate the transaction cost using specified percentage
            transaction_cost = transaction_value * config['TRANSACTION_COST']
            
            # Add to total transaction costs
            total_transaction_cost_pct += transaction_cost / transaction_value if transaction_value > 0 else 0
            total_transaction_cost_amount += transaction_cost
            
            # Add to total turnover
            total_turnover += transaction_value
            
            # Calculate the cash flow from the transaction
            cash_flow = -shares_diff * price
            
            # Update cash balance
            new_cash += cash_flow
        
        # Update the position if it's non-zero
        if target_shares != 0:
            new_positions[stock] = target_shares
    
    # Remove stocks with zero positions
    new_positions = {k: v for k, v in new_positions.items() if v != 0}
    
    # Calculate portfolio value before costs
    positions_value = (pd.Series(new_positions) * current_prices).sum()
    portfolio_value_before_costs = positions_value + new_cash
    
    # Calculate portfolio value after txn costs
    portfolio_value_after_txn_costs = portfolio_value_before_costs - total_transaction_cost_amount
    
    # Ensure that total portfolio value is positive
    # if portfolio_value_after_txn_costs <= 0:
    #     raise ValueError(f"Portfolio value after costs is negative or zero: {portfolio_value_after_txn_costs}")
    
    # Calculate exposures
    long_positions = {k: v for k, v in new_positions.items() if v > 0}
    short_positions = {k: v for k, v in new_positions.items() if v < 0}
    
    long_exposure = sum(v * current_prices[k] for k, v in long_positions.items())
    short_exposure = abs(sum(v * current_prices[k] for k, v in short_positions.items()))
    
    # Calculate leverage as total exposure relative to portfolio value
    leverage = (long_exposure + short_exposure) / portfolio_value_after_txn_costs
    
    # Calculate borrowing costs based on the current leverage
    leverage_cost = 0
    margin_cost = 0
    
    if leverage > 1.0:
        # Calculate the cost of using leverage
        leverage_cost = (leverage - 1.0) * portfolio_value_after_txn_costs * config['LEVERAGE_RATE_DAILY']
        
        # If we have excess long or short exposure, apply margin cost to the excess exposure
        excess_long_exposure = max(0, long_exposure - portfolio_value_after_txn_costs)
        excess_short_exposure = max(0, short_exposure - portfolio_value_after_txn_costs)
        margin_cost = (excess_long_exposure + excess_short_exposure) * config['MARGIN_RATE_DAILY']
    
    # Calculate borrowing cost based on short positions
    borrowing_cost = short_exposure * config['BORROW_RATE_DAILY']
    
    # Calculate total financing costs
    total_financing_cost = leverage_cost + margin_cost + borrowing_cost
    
    # Calculate total cost (transaction + financing)
    total_cost = total_transaction_cost_amount + total_financing_cost
    
    # Calculate portfolio value after all costs
    portfolio_value_after_costs = portfolio_value_after_txn_costs - total_financing_cost
    
    # Update cash balance after financing and transaction costs
    new_cash = new_cash - total_financing_cost - total_transaction_cost_amount
    
    # Create the new portfolio
    new_portfolio = new_positions.copy()
    new_portfolio['cash'] = new_cash
    
    # Calculate portfolio statistics
    active_positions = len(new_positions)
    max_position_size = 0
    if portfolio_value_after_costs > 0 and active_positions > 0:
        position_sizes = [(abs(pos * current_prices[stock]) / portfolio_value_after_costs) for stock, pos in new_positions.items()]
        max_position_size = max(position_sizes) if position_sizes else 0
    
    # Calculate turnover as a percentage of portfolio value
    turnover_pct = total_turnover / portfolio_value_after_costs if portfolio_value_after_costs > 0 else 0
    
    return {
        'portfolio': new_portfolio,
        'transaction_cost': total_transaction_cost_amount,
        'turnover': turnover_pct,
        'portfolio_value_before_costs': portfolio_value_before_costs,
        'portfolio_value_after_txn_costs': portfolio_value_after_txn_costs,
        'portfolio_value_after_costs': portfolio_value_after_costs,
        'leverage': leverage,
        'long_exposure': long_exposure,
        'short_exposure': short_exposure,
        'active_positions': active_positions,
        'max_position_size': max_position_size,
        'borrowing_cost': borrowing_cost,
        'leverage_cost': leverage_cost,
        'margin_cost': margin_cost,
        'total_financing_cost': total_financing_cost,
        'total_cost': total_cost
    }


def implement_strategy(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    config: dict,
) -> tuple:
    """
    Implements the pairs trading strategy over the entire dataset.
    
    Parameters:
    -----------
    prices: pd.DataFrame
        Dataframe of stock prices
    config: dict
        Configuration parameters
        
    Returns:
    --------
    tuple
        Portfolio history, metrics dataframe, and pairs dataframe
    """
    # Get unique dates
    trading_dates = prices.index.unique()
    
    # Initialize variables
    portfolio_history = {}
    metrics_data = []
    pairs_history = []
    current_portfolio = {'cash': config['INITIAL_CASH']}  # Initialize with just our initial cash
    prev_portfolio_value = None
    position_timestamps = {}  # Tracks when each position was entered
    
    # Split the data into formation and trading periods
    for i in range(0, len(trading_dates) - config['FORMATION_PERIOD'] - config['TRADING_PERIOD'], config['TRADING_PERIOD']):
        # Define formation and trading periods
        formation_start = trading_dates[i]
        formation_end = trading_dates[i + config['FORMATION_PERIOD'] - 1]
        trading_start = trading_dates[i + config['FORMATION_PERIOD']]
        trading_end = trading_dates[i + config['FORMATION_PERIOD'] + config['TRADING_PERIOD'] - 1]
        
        print("Formation period: %s to %s" % (formation_start, formation_end))
        print("Trading period: %s to %s" % (trading_start, trading_end))
        
        # Select stocks for the formation period
        formation_stocks, _, _ = select_asset_universe(prices, returns, formation_end, config)
        
        if formation_stocks.empty or formation_stocks.shape[1] < 2:
            print("Not enough stocks with complete data in the formation period. Skipping.")
            continue
        
        # Compute distances between pairs
        pairs = form_pairs(formation_stocks, config)
        
        # Estimate hedge ratios
        hedge_ratios = estimate_hedge_ratio(formation_stocks, pairs, config)
        
        if hedge_ratios.empty:
            print("No pairs found. Skipping.")
            continue
        else: 
            print(f"Pairs found: {len(hedge_ratios)}")
            
            for _, pair_row in hedge_ratios.iterrows():
                pairs_history.append({
                    'formation_start': formation_start,
                    'formation_end': formation_end,
                    'trading_start': trading_start,
                    'trading_end': trading_end,
                    'pair': f"{pair_row['stock1']}_{pair_row['stock2']}",
                    'stock1': pair_row['stock1'],
                    'stock2': pair_row['stock2'],
                    'distance': pair_row['distance'],
                    'hedge_ratio': pair_row['hedge_ratio'],
                    # 'half_life': pair_row['half_life'],
                    'adf_pvalue': pair_row['adf_pvalue']
                })
        
        # Get trading period data INCLUDING the last ESTIMATION_PERIOD days from formation period
        # This allows us to calculate signals immediately at the start of the trading period
        estimation_start_idx = max(0, i + config['FORMATION_PERIOD'] - config['ESTIMATION_PERIOD'])
        estimation_start = trading_dates[estimation_start_idx]
        
        # Get combined data: last ESTIMATION_PERIOD days from formation + regular trading period
        trading_data = prices.loc[estimation_start:trading_end]
        
        print(f"Using data from {estimation_start} to {trading_end} for signal calculation")
        print(f"Actual trading period: {trading_start} to {trading_end}")
        
        # Compute signals for the trading period (including estimation lookback)
        signals = compute_signal(trading_data, hedge_ratios, config)
        
        # Reset position timestamps for new trading period
        active_position_timestamps = {}
        
        # NOTE: Even though we include formation data for signal calculation,
        # we still only start trading at the proper trading_start date
        trading_dates_only = trading_data.loc[trading_start:trading_end].index
        
        # # Compute positions for the first trading day
        positions = allocate_positions(
            signals,
            config,
            trading_data,
            trading_dates_only[0],
            current_portfolio.get('cash')
        )
        
        # Update portfolio for each day in the trading period (not including estimation data)
        for date in trading_dates_only:
            # Recompute positions for each day to account for lagged signals and leverage constraints
            # Only do this after the first day, since we already did it for the first day
            if date != trading_dates_only[0]:
                positions = allocate_positions(
                    signals,
                    config,
                    trading_data,
                    date,
                    current_portfolio.get('cash')
                )
            
            # Update portfolio
            portfolio_update = update_portfolio(
                current_portfolio,
                positions,
                trading_data,
                date,
                config
            )
            
            # Extract values from the portfolio update
            current_portfolio = portfolio_update['portfolio']
            portfolio_value_before_costs = portfolio_update['portfolio_value_before_costs']
            portfolio_value_after_costs = portfolio_update['portfolio_value_after_costs']
            
            # Store portfolio state
            portfolio_history[date] = current_portfolio.copy()
            
            # Calculate return including transaction costs and borrowing costs
            if prev_portfolio_value is not None:
                # Calculate gross return (before any costs)
                gross_return = (portfolio_value_before_costs - prev_portfolio_value) / prev_portfolio_value
                
                # Calculate net return (after all costs)
                # This uses portfolio_value_after_costs which already has transaction and borrowing costs deducted
                period_return = (portfolio_value_after_costs - prev_portfolio_value) / prev_portfolio_value
            else:
                period_return = 0.0
                gross_return = 0.0
            
            # Store all metrics in a single data structure
            metrics_data.append({
                'date': date,
                'portfolio_value': portfolio_value_after_costs,
                'gross_portfolio_value': portfolio_value_before_costs,
                'cash': current_portfolio['cash'],
                'net_returns': period_return,
                'gross_returns': gross_return,
                'transaction_cost': portfolio_update['transaction_cost'],
                'borrowing_cost': portfolio_update['borrowing_cost'],
                'leverage_cost': portfolio_update['leverage_cost'],
                'margin_cost': portfolio_update['margin_cost'],
                'total_cost': portfolio_update['total_cost'],
                'total_financing_cost': portfolio_update['total_financing_cost'],
                'leverage': portfolio_update['leverage'],
                'long_exposure': portfolio_update['long_exposure'],
                'short_exposure': portfolio_update['short_exposure'],
                'num_stocks': portfolio_update['active_positions'],
                'max_position_pct': portfolio_update['max_position_size'],
                'turnover': portfolio_update['turnover'],
            })
            
            prev_portfolio_value = portfolio_value_after_costs
        
        # Merge the position timestamps from this trading period into the global set
        position_timestamps.update(active_position_timestamps)
    
    # Convert all metrics to a single DataFrame
    metrics_df = pd.DataFrame(metrics_data)
    if not metrics_df.empty:
        metrics_df.set_index('date', inplace=True)
    
    # Convert pairs history to DataFrame
    pairs_df = pd.DataFrame(pairs_history)
    
    return (
        portfolio_history, 
        metrics_df,
        pairs_df
    )


def analyze_performance(
    config: dict,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    tickers: list,
    metadata: pd.DataFrame,
    portfolio_history: dict, 
    metrics_df: pd.DataFrame, 
    pairs_df: pd.DataFrame,
) -> dict:
    """
    Analyzes the performance of the strategy.
    
    Parameters:
    -----------
    config: dict
        Configuration dictionary
    prices: pd.DataFrame
        Dataframe with stock prices
    returns: pd.DataFrame
        Dataframe with stock returns
    tickers: list
        List of stock tickers
    metadata: pd.DataFrame
        Dataframe with stock metadata
    portfolio_history: dict
        Dictionary with portfolio history
    metrics_df: pd.DataFrame
        Dataframe with all metrics (returns, costs, exposures, etc.)
    pairs_df: pd.DataFrame
        Pairs information
        
    Returns:
    --------
    dict
        Dictionary with performance metrics
    """
    # Initialize variables
    initial_cash = config['INITIAL_CASH']
    num_pairs = config['NUM_PAIRS']
    transaction_cost = config['TRANSACTION_COST']
    borrow_rate_daily = config['BORROW_RATE_DAILY']
    metrics = {}
    
    # Calculate basic metrics
    net_returns = metrics_df['net_returns']
    metrics['Net Total Return'] = (1 + net_returns).prod() - 1
    metrics['Net Annualized Return'] = (1 + metrics['Net Total Return']) ** (252 / len(net_returns)) - 1
    metrics['Net Volatility'] = net_returns.std() * np.sqrt(252)
    metrics['Net Sharpe Ratio'] = metrics['Net Annualized Return'] / metrics['Net Volatility'] if metrics['Net Volatility'] > 0 else 0
    metrics['Gross Total Return'] = (1 + metrics_df['gross_returns']).prod() - 1
    metrics['Gross Annualized Return'] = (1 + metrics['Gross Total Return']) ** (252 / len(metrics_df['gross_returns'])) - 1
    metrics['Gross Volatility'] = metrics_df['gross_returns'].std() * np.sqrt(252)
    metrics['Gross Sharpe Ratio'] = metrics['Gross Annualized Return'] / metrics['Gross Volatility'] if metrics['Gross Volatility'] > 0 else 0
    
    # Calculate drawdown
    cum_returns = (1 + net_returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max) - 1
    metrics['Net Max Drawdown'] = drawdown.min()
    metrics['Net Calmar Ratio'] = metrics['Net Annualized Return'] / abs(metrics['Net Max Drawdown'])
    gross_cum_returns = (1 + metrics_df['gross_returns']).cumprod()
    running_max = gross_cum_returns.cummax()
    drawdown = (gross_cum_returns / running_max) - 1
    metrics['Gross Max Drawdown'] = drawdown.min()
    metrics['Gross Calmar Ratio'] = metrics['Gross Annualized Return'] / abs(metrics['Gross Max Drawdown'])
        
    # Calculate win rate
    win_days = (net_returns > 0).sum()
    total_days = len(net_returns)
    metrics['Net Win Rate'] = win_days / total_days if total_days > 0 else 0
    gross_win_days = (metrics_df['gross_returns'] > 0).sum()
    metrics['Gross Win Rate'] = gross_win_days / total_days if total_days > 0 else 0
    
    # Calculate skewness and kurtosis
    metrics['Net Skewness'] = net_returns.skew()
    metrics['Net Kurtosis'] = net_returns.kurtosis()
    metrics['Gross Skewness'] = metrics_df['gross_returns'].skew()
    metrics['Gross Kurtosis'] = metrics_df['gross_returns'].kurtosis()
        
    # Visualize cumulative returns
    plt.figure(figsize=(12, 6))
        
    # Plot net returns
    ((1 + metrics_df['net_returns']).cumprod() - 1).plot(label='Net Returns')
    
    # Plot gross returns if available
    ((1 + metrics_df['gross_returns']).cumprod() - 1).plot(label='Gross Returns')
    
    # TODO: Plot benchmark returns if available
        
    plt.title('Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.gca().yaxis.set_major_formatter(mpl_ticker.PercentFormatter(xmax=1))
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Visualize leverage if available
    plt.figure(figsize=(12, 6))
    metrics_df['leverage'].plot(label='Leverage')
    plt.title('Portfolio Leverage Over Time')
    plt.xlabel('Date')
    plt.ylabel('Leverage Ratio')
    plt.axhline(y=1, color='r', linestyle='--', label='Leverage = 1')
    plt.legend()
    plt.grid(True)
    plt.show()
        
    # Plot exposure
    plt.figure(figsize=(12, 6))
    metrics_df['long_exposure'].plot(label='Long Exposure', color='green')
    (-metrics_df['short_exposure']).plot(label='Short Exposure', color='red')
    plt.title('Long and Short Exposure Over Time')
    plt.xlabel('Date')
    plt.ylabel('Exposure')
    plt.gca().yaxis.set_major_formatter(mpl_ticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    plt.legend()
    plt.grid(True)
    plt.show()
        
    # Plot exposure
    plt.figure(figsize=(12, 6))
    (metrics_df['long_exposure'] + metrics_df['short_exposure']).plot(label='Total Exposure', color='black', alpha=0.7)
    (metrics_df['long_exposure'] - metrics_df['short_exposure']).plot(label='Net Exposure', color='purple', alpha=0.7)
    plt.title('Net and Total Exposure Over Time')
    plt.xlabel('Date')
    plt.ylabel('Exposure')
    plt.gca().yaxis.set_major_formatter(mpl_ticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    plt.legend()
    plt.grid(True)
    plt.show()
        
    # Plot portfolio value if available
    plt.figure(figsize=(12, 6))
    metrics_df['portfolio_value'].plot(label='Portfolio Value', color='blue')
    plt.title('Total Portfolio Value Over Time (Positions + Cash)')
    plt.gca().yaxis.set_major_formatter(mpl_ticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.grid(True)
    plt.show()
    
    # Visualize cash balance if available
    plt.figure(figsize=(12, 6))
    metrics_df['cash'].plot(label='Cash Balance', color='green')
    plt.title('Cash Balance Over Time')
    plt.gca().yaxis.set_major_formatter(mpl_ticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    plt.xlabel('Date')
    plt.ylabel('Cash Amount')
    plt.grid(True)
    plt.show()
        
    # Plot cash as percentage of portfolio value
    cash_percentage = metrics_df['cash'] / metrics_df['portfolio_value'] * 100
    plt.figure(figsize=(12, 6))
    cash_percentage.plot(label='Cash as % of Portfolio', color='blue')
    plt.title('Cash as Percentage of Portfolio Value')
    plt.xlabel('Date')
    plt.ylabel('Percentage (%)')
    plt.axhline(y=100, color='r', linestyle='--', label='100% Cash')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Visualize number of stocks in portfolio if available
    plt.figure(figsize=(12, 6))
    metrics_df['num_stocks'].plot(label='Number of Stocks', color='blue')
    plt.title('Number of Stocks in Portfolio Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Stocks')
    plt.grid(True)
    plt.show()
    
    # Visualize maximum position size as percentage of portfolio value
    plt.figure(figsize=(12, 6))
    metrics_df['max_position_pct'].plot(label='Max Position Size (%)', color='orange')
    plt.title('Maximum Position Size as % of Portfolio Value')
    plt.xlabel('Date')
    plt.ylabel('Percentage (%)')
    plt.grid(True)
    plt.show()
    
    # Visualize turnover
    plt.figure(figsize=(12, 6))
    metrics_df['turnover'].plot(label='Turnover', color='purple')
    plt.title('Portfolio Turnover Over Time')
    plt.xlabel('Date')
    plt.ylabel('Turnover')
    plt.ylim(0, min(10, metrics_df['turnover'].max()))
    plt.grid(True)
    plt.show()
    
    # Analyze pairs data if available
    if pairs_df is not None:
        # Count occurrences of each pair
        pair_counts = pairs_df['pair'].value_counts()
        
        # Plot top 10 most frequent pairs
        plt.figure(figsize=(12, 6))
        pair_counts.head(10).plot(kind='bar')
        plt.title('Top 10 Most Frequent Pairs')
        plt.xlabel('Pair')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Display the metadata and the of the top 10 pairs
        top_pairs_info = pair_counts.head(10).to_frame()
        top_pairs_info['ticker1'] = top_pairs_info.index.to_series().str.split('_').str[0]
        top_pairs_info['ticker2'] = top_pairs_info.index.to_series().str.split('_').str[1]
        top_pairs_info = pd.merge(top_pairs_info, metadata, left_on='ticker1', right_index=True, how='left')
        top_pairs_info = pd.merge(top_pairs_info, metadata, left_on='ticker2', right_index=True, how='left')
        top_pairs_info['ticker1_name'] = top_pairs_info['Company_x']
        top_pairs_info['ticker2_name'] = top_pairs_info['Company_y']
        top_pairs_info['ticker1_sector'] = top_pairs_info['Sector_x']
        top_pairs_info['ticker2_sector'] = top_pairs_info['Sector_y']
        top_pairs_info['ticker1_industry'] = top_pairs_info['Industry_x']
        top_pairs_info['ticker2_industry'] = top_pairs_info['Industry_y']
        
        top_pairs_info = top_pairs_info[[
            'ticker1', 'ticker1_name', 'ticker1_sector', 'ticker1_industry', 
            'ticker2', 'ticker2_name', 'ticker2_sector', 'ticker2_industry', 
            'count',
        ]]
        
        print("\nTop 10 Most Frequent Pairs:")
        with pd.option_context('display.max_columns', None, 'display.max_rows', None):
            display(top_pairs_info)
        
        # Count hedge ratios
        def categorize_hedge_ratio(x):
            normalized = x if x >= 1 else 1/x
            if normalized < 10:
                return f"{int(normalized)}-{int(normalized)+1}"
            else:
                return "and 10+"
            
        categorized_ratios = pairs_df['hedge_ratio'].apply(categorize_hedge_ratio)
        plt.figure(figsize=(12, 6))
        ratio_counts = categorized_ratios.value_counts().sort_index()
        ratio_counts.plot(kind='bar')
        plt.title('Distribution of Hedge Ratios')
        plt.xlabel('Hedge Ratio')
        plt.ylabel('Frequency')
        plt.xticks(rotation=0)
        plt.grid(True)
        plt.show()
        
        # # Analyze half-lives
        # plt.figure(figsize=(12, 6))
        # plt.hist(pairs_df['half_life'], bins=20)
        # plt.title('Distribution of Half-Lives')
        # plt.xlabel('Half-Life (days)')
        # plt.ylabel('Frequency')
        # plt.grid(True)
        # plt.show()
        
        # Add pair statistics to metrics
        metrics['Total Unique Pairs'] = len(pair_counts)
        metrics['Most Frequent Pair'] = pair_counts.index[0] if len(pair_counts) > 0 else "None"
        metrics['Avg Hedge Ratio'] = pairs_df['hedge_ratio'].mean()
        # metrics['Avg Half-Life'] = pairs_df['half_life'].mean()
        
        # Print summary of pairs
        print("\nPairs Trading Summary:")
        print(f"Total unique pairs: {len(pair_counts)}")
        print(f"Average hedge ratio: {pairs_df['hedge_ratio'].mean():.2f}")
        # print(f"Average half-life: {pairs_df['half_life'].mean():.2f} days")
    
    # Analyze costs if available
    # Initialize cost variables
    total_transaction_costs = 0
    avg_transaction_cost_per_day = 0
    transaction_cost_total_impact = 0
    transaction_cost_annualized_impact = 0
    
    total_borrowing_costs = 0
    avg_borrowing_cost_per_day = 0
    borrowing_cost_total_impact = 0
    borrowing_cost_annualized_impact = 0
    
    # Calculate transaction cost metrics if available
    if 'transaction_cost' in metrics_df.columns and not metrics_df['transaction_cost'].isna().all():
        # Handle NaN values safely
        clean_tc = metrics_df['transaction_cost'].fillna(0)
        total_transaction_costs = clean_tc.sum()
        avg_transaction_cost_per_day = total_transaction_costs / len(metrics_df) if len(metrics_df) > 0 else 0
        
        # Calculate impact on returns
        if 'gross_returns' in metrics_df.columns:
            # Use safe calculation methods to avoid overflow
            try:
                # Calculate gross return
                gross_returns = metrics_df['gross_returns'].fillna(0)
                gross_return = (1 + gross_returns).prod() - 1
                
                # Convert transaction costs to percentage of portfolio value
                portfolio_values = metrics_df['portfolio_value'].fillna(0)
                tc_pct = clean_tc / portfolio_values.shift(1)
                tc_pct = tc_pct.fillna(0)
                
                # Calculate net return after transaction costs
                net_returns_after_tc = gross_returns - tc_pct
                net_return_after_tc = (1 + net_returns_after_tc).prod() - 1
                
                # Calculate impact
                transaction_cost_total_impact = gross_return - net_return_after_tc
                
                # Annualized impact - only calculate if we have reasonable values
                if np.isfinite(gross_return) and np.isfinite(net_return_after_tc) and len(metrics_df) > 0:
                    gross_annual = (1 + gross_return) ** (252 / len(metrics_df)) - 1
                    net_annual_after_tc = (1 + net_return_after_tc) ** (252 / len(metrics_df)) - 1
                    transaction_cost_annualized_impact = gross_annual - net_annual_after_tc
            except Exception as e:
                print(f"Warning: Error calculating transaction cost impact: {e}")
                transaction_cost_total_impact = 0
                transaction_cost_annualized_impact = 0        
    
    # Calculate borrowing cost metrics if available
    if 'borrowing_cost' in metrics_df.columns and not metrics_df['borrowing_cost'].isna().all():
        # Handle NaN values safely
        clean_bc = metrics_df['borrowing_cost'].fillna(0)
        total_borrowing_costs = clean_bc.sum()
        avg_borrowing_cost_per_day = total_borrowing_costs / len(metrics_df) if len(metrics_df) > 0 else 0
        
        # Calculate impact on returns
        if 'gross_returns' in metrics_df.columns and 'transaction_cost' in metrics_df.columns:
            try:
                # Use the already cleaned values from above
                gross_returns = metrics_df['gross_returns'].fillna(0)
                portfolio_values = metrics_df['portfolio_value'].fillna(0)
                
                # Convert transaction costs to percentage of portfolio value
                tc_pct = clean_tc / portfolio_values.shift(1)
                tc_pct = tc_pct.fillna(0)
                
                # Convert borrowing costs to percentage of portfolio value
                bc_pct = clean_bc / portfolio_values.shift(1)
                bc_pct = bc_pct.fillna(0)
                
                # Calculate net return after transaction costs
                net_returns_after_tc = gross_returns - tc_pct
                net_return_after_tc = (1 + net_returns_after_tc).prod() - 1
                
                # Calculate net return after transaction costs and borrowing costs
                net_returns_after_tc_bc = gross_returns - tc_pct - bc_pct
                net_return_after_tc_bc = (1 + net_returns_after_tc_bc).prod() - 1
                
                # Calculate impact
                borrowing_cost_total_impact = net_return_after_tc - net_return_after_tc_bc
                
                # Annualized impact - only calculate if we have reasonable values
                if np.isfinite(net_return_after_tc) and np.isfinite(net_return_after_tc_bc) and len(metrics_df) > 0:
                    net_annual_after_tc = (1 + net_return_after_tc) ** (252 / len(metrics_df)) - 1
                    net_annual_after_tc_bc = (1 + net_return_after_tc_bc) ** (252 / len(metrics_df)) - 1
                    borrowing_cost_annualized_impact = net_annual_after_tc - net_annual_after_tc_bc
            except Exception as e:
                print(f"Warning: Error calculating borrowing cost impact: {e}")
                borrowing_cost_total_impact = 0
                borrowing_cost_annualized_impact = 0
        
    if 'leverage_cost' in metrics_df.columns and not metrics_df['leverage_cost'].isna().all():
        clean_lc = metrics_df['leverage_cost'].fillna(0)  
        total_leverage_cost = clean_lc.sum()
        avg_leverage_cost_per_day = total_leverage_cost / len(metrics_df) if len(metrics_df) > 0 else 0
        
    if 'margin_cost' in metrics_df.columns and not metrics_df['margin_cost'].isna().all():
        clean_mc = metrics_df['margin_cost'].fillna(0)
        total_margin_cost = clean_mc.sum()
        avg_margin_cost_per_day = total_margin_cost / len(metrics_df) if len(metrics_df) > 0 else 0
        
        
    # Calculate/plot total costs
    # Plot costs
    plt.figure(figsize=(12, 6))
    clean_tc.cumsum().plot(label='Cumulative Transaction Costs') if 'transaction_cost' in metrics_df.columns and not metrics_df['transaction_cost'].isna().all() else None
    clean_bc.cumsum().plot(label='Cumulative Borrowing Costs') if 'borrowing_cost' in metrics_df.columns and not metrics_df['borrowing_cost'].isna().all() else None
    clean_lc.cumsum().plot(label='Cumulative Leverage Costs') if 'leverage_cost' in metrics_df.columns and not metrics_df['leverage_cost'].isna().all() else None
    clean_mc.cumsum().plot(label='Cumulative Margin Costs') if 'margin_cost' in metrics_df.columns and not metrics_df['margin_cost'].isna().all() else None
        
    if (
        'transaction_cost' in metrics_df.columns and not metrics_df['transaction_cost'].isna().all()
        and 'borrowing_cost' in metrics_df.columns and not metrics_df['borrowing_cost'].isna().all()
        and 'leverage_cost' in metrics_df.columns and not metrics_df['leverage_cost'].isna().all()
        and 'margin_cost' in metrics_df.columns and not metrics_df['margin_cost'].isna().all()
    ):
        # Create a total cost series
        total_costs = pd.DataFrame({
                'transaction_cost': clean_tc,
                'borrowing_cost': clean_bc,
                'leverage_cost': clean_lc,
                'margin_cost': clean_mc
            }).sum(axis=1)
        total_costs.cumsum().plot(label='Cumulative Total Costs', linewidth=2)
    
    plt.title('Cumulative Costs Over Time')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Cost')
    plt.gca().yaxis.set_major_formatter(mpl_ticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Add leverage metrics if available
    if 'leverage' in metrics_df.columns:
        metrics['Average Leverage'] = metrics_df['leverage'].mean()
        metrics['Max Leverage'] = metrics_df['leverage'].max()
        metrics['Min Leverage'] = metrics_df['leverage'].min()
    
    # Add profit
    metrics['Total Gross Profit'] = initial_cash * ((1 + metrics_df['gross_returns']).prod() - 1)
    metrics['Total Net Profit'] = initial_cash * ((1 + metrics_df['net_returns']).prod() - 1)
    
    # Add transaction cost metrics
    metrics['Total Transaction Costs'] = total_transaction_costs
    metrics['Avg Transaction Cost/Day'] = avg_transaction_cost_per_day
    metrics['Total Transaction Cost Impact'] = transaction_cost_total_impact
    metrics['Annual Transaction Cost Impact'] = transaction_cost_annualized_impact
    
    # Add borrowing cost metrics
    metrics['Total Borrowing Costs'] = total_borrowing_costs
    metrics['Avg Borrowing Cost/Day'] = avg_borrowing_cost_per_day
    metrics['Total Borrowing Cost Impact'] = borrowing_cost_total_impact
    metrics['Annual Borrowing Cost Impact'] = borrowing_cost_annualized_impact
    
    # Add margin and leverage cost metrics
    metrics['Total Leverage Costs'] = total_leverage_cost
    metrics['Avg Leverage Cost/Day'] = avg_leverage_cost_per_day
    metrics['Total Margin Costs'] = total_margin_cost
    metrics['Avg Margin Cost/Day'] = avg_margin_cost_per_day
    
    # Combined cost metrics
    metrics['Total Costs'] = total_transaction_costs + total_borrowing_costs + total_leverage_cost + total_margin_cost
    metrics['Annual Txn+Borrow Cost Impact'] = transaction_cost_annualized_impact + borrowing_cost_annualized_impact
    
    # Basic monthly returns heatmap (using net returns)
    try:
        monthly_returns = metrics_df['net_returns'].resample('ME').apply(lambda x: (1 + x).prod() - 1)
        
        # Create a DataFrame with the right structure for pivot
        monthly_df = pd.DataFrame({
            'month': monthly_returns.index.month,
            'year': monthly_returns.index.year,
            'returns': monthly_returns.values
        })
        
        # Now create the pivot table
        monthly_table = pd.pivot_table(
            data=monthly_df,
            index='month',
            columns='year',
            values='returns'
        )
        
        # Plot monthly returns as a heatmap
        plt.figure(figsize=(14, 8))
        sns.heatmap(monthly_table.T, annot=True, cmap='RdYlGn', center=0, fmt='.2%')
        plt.title('Monthly Returns Heatmap')
        plt.xlabel('Month')
        plt.ylabel('Year')
        plt.show()
        
        # Annual returns
        annual_returns = metrics_df['net_returns'].resample('YE').apply(lambda x: (1 + x).prod() - 1)
        # Remove time part of date (otherwise it looks bad)
        annual_returns.index = annual_returns.index.year
        
        # Plot annual returns bar chrat
        plt.figure(figsize=(12, 6))
        annual_returns.plot(kind='bar', color=annual_returns.map(lambda x: 'green' if x > 0 else 'red'))
        plt.title('Annual Returns')
        plt.xlabel('Year')
        plt.ylabel('Return')
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
        plt.axhline(0, color='black', linewidth=1)
        
        # Add value labels on top of bars
        for i, v in enumerate(annual_returns):
            # Position labels based on the range of values in the data
            value_range = annual_returns.max() - annual_returns.min()
            label_position = v + np.sign(v) * value_range * (0.05 if v < 0 else 0.02)
            plt.text(i, label_position, f'{v:.2%}', ha='center')
            
        plt.show()
        
    except Exception as e:
        print(f"Could not generate monthly/annual return tables: {e}")
        raise e
    
    # TODO: Add more metrics/tests from quantstats and Gatev et al. 2006
    
    # Display performance metrics as a table
    summary_metrics_df = pd.DataFrame({
        'Metric': list(metrics.keys()),
        'Value': list(metrics.values())
    })
    summary_metrics_df['Formatted Value'] = summary_metrics_df.apply(
        lambda row: (
            f'{row["Value"]:.2%}' if any(x in row["Metric"] for x in ["Return", "Volatility", "Drawdown", "Win Rate"]) else
            f'{row["Value"]:.2f}' if any(x in row["Metric"] for x in ["Sharpe Ratio", "Calmar Ratio", "Skewness", "Kurtosis", "Hedge Ratio"]) or (row["Metric"] == "Leverage") else
            f'${row["Value"]:,.2f}' if any(x in row["Metric"] for x in ["Profit", "Total Transaction Costs", "Avg Transaction Cost/Day", "Total Borrowing Costs", "Avg Borrowing Cost/Day", "Total Costs", "Total Leverage Costs", "Total Margin Costs", "Avg Leverage Cost/Day", "Avg Margin Cost/Day"]) else
            f'{row["Value"]:.2%}' if any(x in row["Metric"] for x in ["Transaction Cost Impact", "Annual Transaction Cost Impact", "Borrowing Cost Impact", "Annual Borrowing Cost Impact", "Annual Cost Impact"]) else
            f'{int(row["Value"])}' if any(x in row["Metric"] for x in ["Pairs"]) else
            f'{row["Value"]:.2%}' if isinstance(row["Value"], (int, float)) and abs(row["Value"]) <= 1 else
            f'{row["Value"]:.2f}' if isinstance(row["Value"], (int, float)) else str(row["Value"])
        ), axis=1
    )
    
    try:
        with pd.option_context('display.max_columns', None, 'display.max_rows', None):
            display(summary_metrics_df[['Metric', 'Value', 'Formatted Value']])
    except Exception as e:
        print(f"Could not display summary metrics table: {e}")
        print(summary_metrics_df[['Metric', 'Value', 'Formatted Value']])
        raise e
    
    return metrics


def save_results(
    unique_id: str,
    config: dict,
    portfolio_history: dict,
    metrics_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
) -> None:
    """
    Saves all results with the unique identifier.
    
    Parameters:
    -----------
    unique_id: str
        Unique identifier for the results
    config: dict
        Configuration parameters
    portfolio_history: dict
        Portfolio history
    metrics_df: pd.DataFrame
        Dataframe with all metrics (returns, costs, exposures, etc.)
    pairs_df: pd.DataFrame
        Pairs information
        
    Returns:
    --------
    None
    """    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Save metrics dataframe
    metrics_df.to_csv(f'results/{unique_id}_metrics.csv')
    
    # Save pairs dataframe
    pairs_df.to_csv(f'results/{unique_id}_pairs.csv')
    
    # Save portfolio history and config
    import pickle
    with open(f'results/{unique_id}_portfolio_history.pkl', 'wb') as f:
        pickle.dump(portfolio_history, f)
    with open(f'results/{unique_id}_config.json', 'w') as f:
        json.dump(config, f)
    
    print(f"Results saved with ID: {unique_id}")


def get_unique_id(config: dict) -> str:
    """
    Generates a unique identifier from the config.
    
    Parameters:
    -----------
    config: dict
        Configuration parameters
        
    Returns:
    --------
    str
        Unique identifier
    """
    # Normalize the config (convert all numbers to the same format); 
    # Could perhaps use json.dumps(config) instead after sorting
    normalized_config = {}
    for k, v in config.items():
        if isinstance(v, (int, float)):
            normalized_config[k] = float(v)
        elif isinstance(v, (list, tuple)):
            for i, item in enumerate(v):
                if not isinstance(item, (int, float, str, bool)):
                    raise ValueError("Unsupported type for config value: "
                                     f"'{type(item)}'; value: {item}; parent: {k}")
                v[i] = float(item) if isinstance(item, int) else item
            normalized_config[k] = tuple(v)
        elif isinstance(v, set):
            for item in v:
                if not isinstance(item, (int, float, str, bool)):
                    raise ValueError("Unsupported type for config value: "
                                     f"'{type(item)}'; value: {item}; parent: {k}")
                if isinstance(item, int):
                    v.remove(item)
                    v.add(float(item))
            normalized_config[k] = tuple(v)
        elif isinstance(v, dict):
            for k, vv in v.items():
                if not isinstance(vv, (int, float, str, bool)):
                    raise ValueError("Unsupported type for config value: "
                                     f"'{type(vv)}'; value: {vv}; parent: {k}")
                v[k] = float(vv) if isinstance(vv, int) else vv
            normalized_config[k] = tuple(v.items())
        elif isinstance(v, (bool, str)):
            normalized_config[k] = v
        else:
            raise ValueError("Unsupported type for config value: "
                             f"'{type(v)}'; value: {v}")
    
    # Sort config        
    sorted_config = sorted(normalized_config.items(), key=lambda item: item[0])
    config_hash = hashlib.sha256(str(sorted_config).encode()).hexdigest()
    
    # Break hash into three numbers
    hash_len = len(config_hash)
    substring_len = hash_len // 3
    lookups = [
        int(config_hash[0:substring_len], 16),
        int(config_hash[substring_len:2*substring_len], 16),
        int(config_hash[2*substring_len:], 16)
    ]
    
    # Generaet petname
    tables = [
        petname.english.adverbs,
        petname.english.adjectives,
        petname.english.names,
    ]
    part_ids = [
        tables[i][lookup % len(tables[i])] for i, lookup in enumerate(lookups)
    ]
    unique_id = '-'.join(part_ids)
    
    print(f"Generated unique ID: {unique_id}")
    
    return unique_id


def load_results(unique_id: str):
    if os.path.exists(f'results/{unique_id}_metrics.csv'):
        print(f"Results already exist for '{unique_id}'; loading")
        # load results
        with open(f'results/{unique_id}_portfolio_history.pkl', 'rb') as f:
            portfolio_history = pickle.load(f)
        # load CSVs
        metrics_df = pd.read_csv(f'results/{unique_id}_metrics.csv', index_col=0, parse_dates=True, date_format='%Y-%m-%d')
        pairs_df = pd.read_csv(f'results/{unique_id}_pairs.csv', index_col=0, parse_dates=True, date_format='%Y-%m-%d')
        return (
            portfolio_history,
            metrics_df,
            pairs_df
        )
    else:
        return None 


def validate_config(config: dict) -> None:
    """
    Validates the configuration parameters.
    
    Parameters:
    -----------
    config: dict
        Configuration parameters
    
    Returns:
    --------
    None
    """
    assert config['NIKKEI_CSV_PATH'], "Nikkei CSV path must be specified"
    assert config['FILTER_MAX_ABS_RETURN'] >= 0, "Filter max absolute return must be non-negative"
    assert config['FORMATION_PERIOD'] > 0, "Formation period must be positive"
    assert config['TRADING_PERIOD'] > 0, "Trading period must be positive"
    assert config['NUM_PAIRS'] > 0, "Number of pairs must be positive"
    assert config['DISTANCE_METRIC'] in ['ssd'], "Distance metric must be 'ssd'"
    assert config['HEDGE_RATIO_METHOD'] in ['ols', 'unit'], "Hedge ratio method must be 'ols' or 'unit'"
    assert 0 < config['COINT_THRESHOLD'] < 1, "Cointegration threshold must be in (0, 1)"
    assert config['ESTIMATION_PERIOD'] > 0, "Estimation period must be positive"
    assert config['ESTIMATION_PERIOD'] < config['TRADING_PERIOD'], "Estimation period must fit inside trading period"
    assert config['Z_THRESHOLD'] > 0, "Z-score threshold must be positive"
    assert config['TRANSACTION_COST'] >= 0, "Transaction cost must be non-negative"
    assert config['BORROW_RATE_DAILY'] >= 0, "Borrowing rate must be non-negative"
    assert config['LEVERAGE_RATE_DAILY'] >= 0, "Leverage rate must be non-negative"
    assert config['MARGIN_RATE_DAILY'] >= 0, "Margin rate must be non-negative"
    assert config['INITIAL_CASH'] > 0, "Initial cash must be positive"
    assert config['MAX_LEVERAGE'] > 0, "Max leverage must be positive"
    # assert config['CAPITAL_PER_PAIR'] > 0, "Capital per pair must be positive"
    # assert config['CLOSE_OLD_POSITIONS'] in [True, False], "Close old positions must be True or False"


def run_backtest(config=None):
    """
    Runs the full strategy and analyzes performance.
    """
    # Parameters
    config = config or {
        # Save results
        'SAVE_RESULTS': True,
        # Path to the Nikkei 225 CSV file
        'NIKKEI_CSV_PATH': 'N225.csv',
        # The maximum absolute return allowed, e.g. 0.5
        'FILTER_MAX_ABS_RETURN': 0.5,
        # Length of formation period, in trading days; also used as period to select asset universe based on data availability and returns
        'FORMATION_PERIOD': 252,
        # Length of trading period, in trading days
        'TRADING_PERIOD': 126,
        # Top number of pairs to select for trading
        'NUM_PAIRS': 50,
        # Distance metric to use for selecting pairs
        'DISTANCE_METRIC': 'ssd',
        # Method to use for hedge ratio estimation ("ols" or "unit" or your own)
        'HEDGE_RATIO_METHOD': 'unit',
        # Cointegration p-value threshold for choosing pairs (set to 0.05 for 5% significance level)
        'COINT_THRESHOLD': 0.05,
        # Length of rolling mean/volatility estimation period for spreads, in trading days
        'ESTIMATION_PERIOD': 20,
        # Z-score threshold for generating signals
        'Z_THRESHOLD': 2.0,
        # Transaction cost as a fraction of trade value in percentage (0.0001 = 1 bps)
        'TRANSACTION_COST': 0.0005,
        # Daily borrowing cost for leverage in percentage (0.0001 = 1 bps); multiply by 252 to convert to annual rate
        'BORROW_RATE_DAILY': 0.03 / 252,
        # Leverage rate daily on balanced leverage in percentage (0.0001 = 1 bps); multiply by 252 to convert to annual rate
        'LEVERAGE_RATE_DAILY': 0.01 / 252,
        # Margin rate daily on unbalanced (excess) leverage in percentage (0.0001 = 1 bps); multiply by 252 to convert to annual rate
        'MARGIN_RATE_DAILY': 0.05 / 252,
        # Initial cash amount
        'INITIAL_CASH': 1_000_000.0,
        # Maximum leverage to target
        'MAX_LEVERAGE': 0.05,
    }
    
    print(f"Starting from '{os.getcwd()}'")
    
    # Validate configuration
    validate_config(config)
    
    # Generate unique identifier for config
    unique_id = get_unique_id(config)
    
    # Load the data
    prices, returns, tickers, metadata = load_data(config)
    
    # Load results if extant, otherwise run strategy
    results = load_results(unique_id)
    if not results:
        results = implement_strategy(
            prices,
            returns,
            config,
        )
        if config['SAVE_RESULTS']:
            save_results(
                unique_id,
                config,
                *results,
            )
    
    # Analyze performance and plot/print results
    performance_metrics = analyze_performance(
        config,
        prices, 
        returns, 
        tickers, 
        metadata,
        *results,
    )
        
    return (
        prices,
        returns,
        tickers,
        metadata,
        *results
    )


if __name__ == "__main__":  
    run_backtest()