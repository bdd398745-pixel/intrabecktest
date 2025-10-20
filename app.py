import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator, StochRSIIndicator, ROCIndicator, UltimateOscillator
from ta.trend import MACD, ADXIndicator, CCIIndicator
from ta.volatility import AverageTrueRange
import pytz

# --- Prepare data with indicators ---
def prepare_data(df):
    df = df.copy()
    
    # Convert index to IST
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')

    close = df['Close'].astype(float)
    high = df['High'].astype(float)
    low = df['Low'].astype(float)

    df['RSI'] = RSIIndicator(close, window=9).rsi()
    df['Stoch'] = StochasticOscillator(high, low, close).stoch()
    df['StochRSI'] = StochRSIIndicator(close).stochrsi()
    df['MACD'] = MACD(close).macd_diff()
    df['ADX'] = ADXIndicator(high, low, close).adx()
    df['CCI'] = CCIIndicator(high, low, close).cci()
    df['UltimateOsc'] = UltimateOscillator(high, low, close).ultimate_oscillator()
    df['ROC'] = ROCIndicator(close).roc()
    df['ATR'] = AverageTrueRange(high, low, close).average_true_range()
    
    df['EMA13'] = close.ewm(span=13, adjust=False).mean()
    df['BullBear'] = high - df['EMA13']

    # --- Signal Logic ---
    def get_signal(row):
        score = 0
        score += 1 if row['RSI'] < 30 else -1 if row['RSI'] > 70 else 0
        score += 1 if row['Stoch'] < 20 else -1 if row['Stoch'] > 80 else 0
        score += 1 if row['StochRSI'] < 0.2 else -1 if row['StochRSI'] > 0.8 else 0
        score += 1 if row['MACD'] > 0 else -1 if row['MACD'] < 0 else 0
        score += 1 if row['ADX'] > 25 else 0
        score += 1 if row['CCI'] < -100 else -1 if row['CCI'] > 100 else 0
        score += 1 if row['UltimateOsc'] < 30 else -1 if row['UltimateOsc'] > 70 else 0
        score += 1 if row['ROC'] > 0 else -1 if row['ROC'] < 0 else 0
        score += 1 if row['BullBear'] > 0 else -1 if row['BullBear'] < 0 else 0

        if score > 0:
            return "BUY"
        elif score < 0:
            return "SELL"
        else:
            return "NEUTRAL"

    df['Signal'] = df.apply(get_signal, axis=1)
    return df

# --- Backtesting ---
def backtest(df):
    trades = []
    position = None

    for i in range(1, len(df)):
        price = float(df['Close'].iloc[i])
        atr = float(df['ATR'].iloc[i])
        time = df.index[i]

        if position is None:
            if df['Signal'].iloc[i-1] == "BUY" and df['Signal'].iloc[i] == "BUY":
                position = {
                    "type": "BUY",
                    "entry": price,
                    "sl": price - atr,
                    "target": price + 2 * atr,
                    "entry_time": time
                }
            elif df['Signal'].iloc[i-1] == "SELL" and df['Signal'].iloc[i] == "SELL":
                position = {
                    "type": "SELL",
                    "entry": price,
                    "sl": price + atr,
                    "target": price - 2 * atr,
                    "entry_time": time
                }
        else:
            # --- Exit logic ---
            if position["type"] == "BUY":
                if price <= float(position["sl"]):
                    pnl = float(position["sl"]) - float(position["entry"])
                    trades.append({"Exit": "Stoploss", "PnL": pnl, "Type": "BUY", "EntryTime": position["entry_time"], "ExitTime": time})
                    position = None
                elif price >= float(position["target"]):
                    pnl = float(position["target"]) - float(position["entry"])
                    trades.append({"Exit": "Target", "PnL": pnl, "Type": "BUY", "EntryTime": position["entry_time"], "ExitTime": time})
                    position = None
            elif position["type"] == "SELL":
                if price >= float(position["sl"]):
                    pnl = float(position["entry"]) - float(position["sl"])
                    trades.append({"Exit": "Stoploss", "PnL": pnl, "Type": "SELL", "EntryTime": position["entry_time"], "ExitTime": time})
                    position = None
                elif price <= float(position["target"]):
                    pnl = float(position["entry"]) - float(position["target"])
                    trades.append({"Exit": "Target", "PnL": pnl, "Type": "SELL", "EntryTime": position["entry_time"], "ExitTime": time})
                    position = None

    return pd.DataFrame(trades)
