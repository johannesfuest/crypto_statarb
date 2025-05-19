# %%
import pandas as pd

prices = pd.read_csv("/share/data/jfuest_crypto/sample/prices.csv")
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
print(f"Funding rates data shape: {funding_rates.shape}")
print(f"Funding rates data columns: {funding_rates.columns}")
print(f"Funding rates data types: {funding_rates.dtypes}")
print(f"Funding rates data head: {funding_rates.head()}")
# %%
# Check for missing values
missing_values = funding_rates.isnull().sum()/ funding_rates.shape[0] * 100
print(f"Missing values in each column (%):\n{missing_values[missing_values > 0]}")  
# %%
