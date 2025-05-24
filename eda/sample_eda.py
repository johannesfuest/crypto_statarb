# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
prices = pd.read_csv("/share/data/jfuest_crypto/sample/prices.csv")
# prices = pd.read_csv("/Users/pinkyvicky/Desktop/mse244/crypto_statarb/prices.csv")
print(f"Prices data shape: {prices.shape}")
print(f"Prices data columns: {prices.columns}")
print(f"Prices data types: {prices.dtypes}")
print(f"Prices data head: {prices.head()}")

# %%
# Check for missing values
missing_values = prices.isnull().sum()/ prices.shape[0] * 100
print(f"Missing values in each column (%):\n{missing_values[missing_values > 0]}")
# check for missing values grouped by coin
missing_values_by_coin = prices.groupby("coin").apply(lambda x: x.isnull().sum()/ x.shape[0] * 100)
print(f"Missing values by coin (%):\n{missing_values_by_coin[missing_values_by_coin > 0]}")
# %%
# Check for number of unique coins
unique_coins = prices["coin"].nunique()
print(f"Number of unique coins: {unique_coins}")

# %%
print(f"coin values: {prices['coin'].unique()}")
# %%
funding_rates = pd.read_csv("/share/data/jfuest_crypto/sample/funding_rates.csv")
# funding_rates = pd.read_csv("/Users/pinkyvicky/Desktop/mse244/crypto_statarb/funding_rates.csv")
print(f"Funding rates data shape: {funding_rates.shape}")
print(f"Funding rates data columns: {funding_rates.columns}")
print(f"Funding rates data types: {funding_rates.dtypes}")
print(f"Funding rates data head: {funding_rates.head()}")
# %%
# Check for missing values
missing_values = funding_rates.isnull().sum()/ funding_rates.shape[0] * 100
print(f"Missing values in each column (%):\n{missing_values[missing_values > 0]}")  
# %%
# Convert timestamps to datetime
prices['open_time'] = pd.to_datetime(prices['open_time'], unit='ms')
prices['close_time'] = pd.to_datetime(prices['close_time'], unit='ms')

# %%
def hourly_amihud(df):
    df['return'] = df.groupby('coin')['close'].pct_change()
    df['amihud'] = np.abs(df['return']) / df['quote_volume'].replace(0, np.nan)
    
    amihud_records = df.groupby(['coin', pd.Grouper(key='open_time', freq='H')])['amihud'].mean().reset_index()
    return amihud_records

def hourly_roll(df):
    roll_records = []
    for (coin, time), group in df.groupby(['coin', pd.Grouper(key='open_time', freq='H')]):
        if group['close'].count() < 3:
            roll_val = np.nan
        else:
            price_diff = group['close'].diff().dropna()
            if len(price_diff) >= 2:
                cov = np.cov(price_diff[1:], price_diff[:-1])[0, 1]
                roll_val = 2 * np.sqrt(-cov) if cov < 0 else np.nan
            else:
                roll_val = np.nan
        roll_records.append({'open_time': time, 'coin': coin, 'roll_estimate': roll_val})
    return pd.DataFrame(roll_records)

def hourly_kyle(df):
    df['return'] = df.groupby('coin')['close'].pct_change()
    df['signed_volume'] = df['taker_buy_volume'] - (df['volume'] - df['taker_buy_volume'])

    kyle_records = []
    for (coin, time), group in df.groupby(['coin', pd.Grouper(key='open_time', freq='H')]):
        group = group.dropna(subset=['return', 'signed_volume'])
        if len(group) < 2:
            kyle_val = np.nan
        else:
            cov = np.cov(group['return'], group['signed_volume'])[0, 1]
            var = np.var(group['signed_volume'])
            kyle_val = cov / var if var > 0 else np.nan
        kyle_records.append({'open_time': time, 'coin': coin, 'kyle_lambda': kyle_val})
    return pd.DataFrame(kyle_records)

# %%
amihud_hourly_df = hourly_amihud(prices)
roll_hourly_df = hourly_roll(prices)
kyle_hourly_df = hourly_kyle(prices)

# %%
# Pivot all to common format
amihud_pivot = amihud_hourly_df.pivot(index='open_time', columns='coin', values='amihud')
roll_pivot = roll_hourly_df.pivot(index='open_time', columns='coin', values='roll_estimate')
kyle_pivot = kyle_hourly_df.pivot(index='open_time', columns='coin', values='kyle_lambda')
# %%
amihud_pivot.head()
# %%
prices.set_index('open_time', inplace=True)
# %%
prices.tail()
# %%
