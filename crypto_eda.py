# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% load the data for January 2022
base_dir = "/share/data/jfuest_crypto/test/"

coins = ["BTC", "ETH", "BNB", "SOL", "DOGE"]
dfs = {}
for coin in coins:
    dfs[coin] = pd.read_csv(f"{base_dir}{coin}USDT-1m-2022-01.csv")
    print(f"Loaded {coin} data with shape: {dfs[coin].shape}")
    
# %%
print(f"Available columns: {dfs['BTC'].columns}")
print(f"BTC data types: {dfs['BTC'].dtypes}")
print(f"BTC data head: {dfs['BTC'].head()}")
# %% load the respective funding rates
funding_rates = {}
for coin in coins:
    funding_rates[coin] = pd.read_csv(f"{base_dir}{coin}USDT-fundingRate-2022-01.csv")

print(f"Available columns: {funding_rates['BTC'].columns}")
print(f"BTC funding rates data types: {funding_rates['BTC'].dtypes}")
print(f"BTC funding rates data head: {funding_rates['BTC'].head()}")
