
# Binance 1-Minute BTC Futures Data – Column Descriptions

This document explains the meaning of each column in the Binance 1-minute candlestick (kline) data.

---

## 🕐 `open_time`
- **Type**: `int64` (milliseconds since Unix epoch)
- **Meaning**: The **start timestamp** of the 1-minute candlestick.
- **Example**: `1640995200000` → corresponds to **2022-01-01 00:00:00 UTC**
- **Usage**: This is the official time bin label for the row; all prices and volumes are for trades that occurred **within the 60-second interval starting at this time**.

---

## 📉 `open`
- **Type**: `float64`
- **Meaning**: The **price of the first trade** executed within this 1-minute interval.
- **Usage**: Represents the market opening price during this time window.

---

## 📈 `high`
- **Type**: `float64`
- **Meaning**: The **highest trade price** seen during the 1-minute interval.
- **Usage**: Reflects intraminute volatility or price spikes.

---

## 📉 `low`
- **Type**: `float64`
- **Meaning**: The **lowest trade price** observed during the 1-minute interval.

---

## 🏁 `close`
- **Type**: `float64`
- **Meaning**: The **price of the last trade** executed before the candle closed.

---

## 📊 `volume`
- **Type**: `float64`
- **Meaning**: The **total base asset volume** traded during the interval.
- **In BTC/USDT**, this is the **amount of BTC traded**.

---

## 🕔 `close_time`
- **Type**: `int64` (milliseconds since epoch)
- **Meaning**: The **end timestamp** of the candlestick.  
- This is always:
  ```
  close_time = open_time + 59999
  ```
  So for a 1-minute interval, it ends 1 millisecond before the next candle starts.
- **Example**:  
  If `open_time` = `1640995200000` (2022-01-01 00:00:00 UTC)  
  → `close_time` = `1640995259999` (2022-01-01 00:00:59.999 UTC)

---

## 💵 `quote_volume`
- **Type**: `float64`
- **Meaning**: The total traded volume **in quote currency** (e.g., USDT) during the interval.
- Calculated as:
  ```
  quote_volume = sum(price × trade size in BTC)
  ```

---

## 🔢 `count`
- **Type**: `int64`
- **Meaning**: The **number of individual trades** executed during the interval.

---

## 🤝 `taker_buy_volume`
- **Type**: `float64`
- **Meaning**: The amount of **BTC bought via market orders** (i.e., taker trades) during the interval.
- Reflects **aggressive buying behavior**.

---

## 💰 `taker_buy_quote_volume`
- **Type**: `float64`
- **Meaning**: The **USDT equivalent** of the `taker_buy_volume`, i.e.,
  ```
  sum(price × taker trade size)
  ```

---

## ❌ `ignore`
- **Type**: `int64` (always 0)
- **Meaning**: Placeholder column — has **no practical use**. Reserved by Binance API for potential future data.

---

## 🧠 Timestamp Conversion in Pandas
```python
df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
```


# BTC Perpetual Futures Funding Rate Data

## Dataset Overview

This dataset contains historical funding rate observations for BTC perpetual futures contracts.

### Columns

| Column Name               | Type     | Description |
|---------------------------|----------|-------------|
| `calc_time`               | `int64`  | Timestamp (in milliseconds since epoch) when the funding rate was calculated. |
| `funding_interval_hours`  | `int64`  | Length of each funding interval in hours (typically 8). |
| `last_funding_rate`       | `float64`| Funding rate applied during that interval (e.g., 0.0001 = 0.01%). |

### Example Row

| calc_time       | funding_interval_hours | last_funding_rate |
|------------------|------------------------|--------------------|
| 1640995200006    | 8                      | 0.0001             |

This corresponds approximately to:  
**Jan 1, 2022, 00:00 UTC**, with a **0.01% funding rate** applied over an **8-hour interval**.

---

## Funding Rate Mechanics

Perpetual futures have no expiry, so a **funding mechanism** is used to tether their price to the spot market. At fixed intervals (e.g., every 8 hours), **traders exchange payments** based on the funding rate:

- If the **funding rate is positive**, **longs pay shorts**.
- If the **funding rate is negative**, **shorts pay longs**.

### Formula

```text
Funding Payment = Position Size × Funding Rate