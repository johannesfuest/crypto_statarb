# %%
%load_ext autoreload
%autoreload 2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from backtest import (
    load_data_crypto,
    select_asset_universe_crypto,
    form_pairs_crypto,
    estimate_hedge_ratio,
    compute_signal,
    allocate_positions,
    run_backtest,
)

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 40)
pd.set_option('display.precision', 4)

# %% 
config = {
    "CRYPTO_CSV_PATH": "prices.csv",
    "SAMPLE_PRICES_PATH": "/share/data/jfuest_crypto/sample/prices.csv",
    "FUNDING_RATES_PATH": "funding_rates.csv",
    "strategy": "forecast", # options: 'forecast', 'zscore'
    "criterion": "quote_volume", # criterion, must be one of 'quote_volume', 'amihud', 'kyle'
    "liquidity_interval": 60*24*7, # Length of intervals for aggergatuion of the liquidity measure 
    "liquidity_threshold": 500000000, # highest liquidity rank allowed for impact metric, min interval volume if quote volume
    "dist_metric": "ssd", # options: 'ssd', 'manhattan', 'euclidean', 'correlation', 'dtw', 'lcs'
    "n_pairs": 10, # Number of pairs to form
    'HEDGE_RATIO_METHOD': 'ols', # options (for now) are ols and unit
    'COINT_THRESHOLD': 0.05,
    'Z_THRESHOLD': 2.0, # z-score threshold for cointegration
    'ESTIMATION_PERIOD': 60*6, # Length of estimation period for z scores
    'EXIT_STRATEGY': 'convergence',
    'MAX_LEVERAGE': 0.05,
    #-------Below are backtest parameters that still need to get adjusted or understood-------
    # Transaction cost as a fraction of trade value in percentage (0.0001 = 1 bps)
    'TRANSACTION_COST': 0.0005,
    'BORROW_RATE_DAILY': 0,
    'LEVERAGE_RATE_DAILY': 0,
    'MARGIN_RATE_DAILY': 0,
    # Initial cash amount
    'INITIAL_CASH': 1000.0,
    'FORMATION_PERIOD': 60*24*7, # Used to filter by liquidity, and form pairs
    'TRADING_PERIOD': 60*24*7, # Muste be greaterthat formation period
    'SAVE_RESULTS': True, # Whether to save the results to a CSV file
    'update_frequency': 60*24, # How often to update the portfolio (in minutes)
}

# %% run select asset universe
prices_unselected, returns_unselected, tickers, amihud, kyle, dollar_vol, funding_rates = load_data_crypto(config, verbose=True)

# %% Run asset selection test and generate plot
sample_freq = config["TRADING_PERIOD"]
asset_filter = pd.DataFrame(index=prices_unselected.index[::sample_freq], columns=tickers, data=False)

for date in prices_unselected.index[::sample_freq]:
    selected_stocks, selected_rets, selected_uni = select_asset_universe_crypto(
        prices_unselected.copy(),
        returns_unselected.copy(),
        amihud.copy(),
        kyle.copy(),
        dollar_vol.copy(),
        date=date, 
        config=config,
    )
    asset_filter.loc[date, selected_stocks.columns] = True

plt.imshow(asset_filter, aspect='auto', cmap='viridis', interpolation=None)
plt.xlabel('Stock')
plt.ylabel('Date')
plt.title('Asset Filter Selection Over Time')
plt.grid(False)
plt.show()

# %% Run form_pairs test based on a test date
test_date = pd.to_datetime("2023-02-01 00:00:59.999000+00:00", utc=True)
selected_prices, selected_returns, coins = select_asset_universe_crypto(
    prices_unselected.copy(),
    returns_unselected.copy(),
    amihud,
    kyle,
    dollar_vol,
    date=test_date,
    config=config,
)
selected_prices_safe = selected_prices[test_date - pd.Timedelta(minutes=config["FORMATION_PERIOD"]):test_date].copy()
# Compute distances
pairs = form_pairs_crypto(selected_prices_safe, config)
print(pairs)

# Plot the top pair's normalized prices over the (approximate) lookback window
ticker1 = pairs.loc[0, 'stock1']
ticker2 = pairs.loc[0, 'stock2']

s1 = selected_prices_safe[ticker1]
s2 = selected_prices_safe[ticker2]
s1 = s1 / s1.iloc[0]
s2 = s2 / s2.iloc[0]
plt.plot(s1, label=f'Coin 1: {ticker1}')
plt.plot(s2, label=f'Coin 2: {ticker2}')
plt.legend()
plt.show()
# %% Estimate hedge ratios and visualize
hedge_ratios = estimate_hedge_ratio(selected_prices_safe, pairs, config)
print(hedge_ratios)

# Plot the first hedge ratio as a spread
pair_idx = 0

ticker1 = hedge_ratios['stock1'].iloc[pair_idx]
ticker2 = hedge_ratios['stock2'].iloc[pair_idx]
hr = hedge_ratios['hedge_ratio'].iloc[pair_idx]
s1 = selected_prices[ticker1]
s2 = selected_prices[ticker2]
spread = s1 - hr * s2
spread.plot()
plt.title(f'Spread: {ticker1} - {hr:.2f} * {ticker2}')
plt.show()

# %% Compute signal test
# Compute signals
signals = compute_signal(selected_prices_safe, hedge_ratios, config)
pair_idx = 0

# Get ticker info for the selected pair
ticker1 = hedge_ratios['stock1'].iloc[pair_idx]
ticker2 = hedge_ratios['stock2'].iloc[pair_idx]
hr = hedge_ratios['hedge_ratio'].iloc[pair_idx]

# Create figure with two subplots sharing x-axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot normalized prices on first subplot
s1 = selected_prices_safe[ticker1]
s2 = selected_prices_safe[ticker2]
s1_norm = s1 / s1.iloc[0]
s2_norm = s2 / s2.iloc[0]
ax1.plot(s1_norm.index, s1_norm, label=ticker1)
ax1.plot(s2_norm.index, s2_norm, label=ticker2)
ax1.set_title('Normalized Price')
ax1.legend()

# Plot spread and signals on second subplot
spread = s1 - hr * s2
ax2.plot(spread.index, spread, label='Spread')

# Add signals to the second subplot
pair_signals = signals[f"{ticker1}_{ticker2}"]
entry_long = np.where(pair_signals['signal'] == -1)[0]
entry_short = np.where(pair_signals['signal'] == 1)[0]
exit_points = np.where(pair_signals['signal'] == 0)[0]

ax2.scatter(spread.index[entry_long], spread.iloc[entry_long], 
            color='green', marker='^', s=100, label='Long Position')
ax2.scatter(spread.index[entry_short], spread.iloc[entry_short], 
            color='red', marker='v', s=100, label='Short Position')
ax2.scatter(spread.index[exit_points], spread.iloc[exit_points], 
            color='black', marker='o', s=50, label='Flat Position')
ax2.set_title(f'Spread and Signals: {ticker1} - {hr:.2f} * {ticker2}')
ax2.legend()

plt.tight_layout()
plt.show()

# %% Test allocate positions
pair_idx = 0

# Get ticker info for the selected pair
ticker1 = hedge_ratios['stock1'].iloc[pair_idx]
ticker2 = hedge_ratios['stock2'].iloc[pair_idx]
hr = hedge_ratios['hedge_ratio'].iloc[pair_idx]

# Compute the positions using allocate_positions
positions_t1 = {}
positions_t2 = {}
for date in selected_prices_safe.index:
    positions_t1[date] = allocate_positions(
        signals, config, selected_prices_safe, pd.Timestamp(date), 1_000_000
    ).get(ticker1, 0)
    positions_t2[date] = allocate_positions(
        signals, config, selected_prices_safe, pd.Timestamp(date), 1_000_000
    ).get(ticker2, 0)

positions = pd.DataFrame(
    {ticker1: positions_t1.values(), ticker2: positions_t2.values()},
    index=selected_prices_safe.index,
)
print(positions)

# Create figure with three subplots sharing x-axis
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Plot normalized prices on first subplot
s1 = selected_prices_safe[ticker1]
s2 = selected_prices_safe[ticker2]
s1_norm = s1 / s1.iloc[0]
s2_norm = s2 / s2.iloc[0]
ax1.plot(s1_norm.index, s1_norm, label=ticker1)
ax1.plot(s2_norm.index, s2_norm, label=ticker2)
ax1.set_title('Normalized Price')
ax1.legend()

# Plot spread and signals on second subplot
spread = s1 - hr * s2
ax2.plot(spread.index, spread, label='Spread')

# Add signals to the second subplot
pair_signals = signals[f"{ticker1}_{ticker2}"]
entry_long = np.where(pair_signals['signal'] == -1)[0]
entry_short = np.where(pair_signals['signal'] == 1)[0]
exit_points = np.where(pair_signals['signal'] == 0)[0]

ax2.scatter(spread.index[entry_long], spread.iloc[entry_long], 
            color='green', marker='^', s=100, label='Long Position')
ax2.scatter(spread.index[entry_short], spread.iloc[entry_short], 
            color='red', marker='v', s=100, label='Short Position')
ax2.scatter(spread.index[exit_points], spread.iloc[exit_points], 
            color='black', marker='o', s=50, label='Flat Position')
ax2.set_title(f'Spread and Signals: {ticker1} - {hr:.2f} * {ticker2}')
ax2.legend()

# Now plot the positions on the third subplot
ax3.plot(positions.index, positions[ticker1], label=ticker1)
ax3.plot(positions.index, positions[ticker2], label=ticker2)
ax3.set_title('Positions (Number of Shares)')
ax3.legend()

plt.tight_layout()
plt.show()

# %% Run backtest
(prices, returns, tickers, portfolio_history, metrics_df, pairs_df) = run_backtest(config=config)


