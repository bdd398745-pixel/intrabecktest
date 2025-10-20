import yfinance as yf
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator, StochasticOscillator, StochRSIIndicator, ROCIndicator, UltimateOscillator
from ta.trend import MACD, ADXIndicator, CCIIndicator
from ta.volatility import AverageTrueRange

# --- Streamlit Page Config ---
st.set_page_config(page_title="📊 Intraday Backtesting", layout="wide")
st.title("📈 Intraday Signal Strategy Backtester")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("⚙️ Strategy Settings")
    ticker = st.text_input("Enter Stock Ticker (e.g., TCS.NS, INFY.NS, AAPL):", "TCS.NS")
    interval = st.selectbox("Select Interval", ["1m", "5m", "15m", "30m", "1h"], index=2)
    period = st.selectbox("Select Period", ["1d", "5d", "7d"], index=1)
    run_bt = st.button("🚀 Run Backtest")

# --- Function to fetch data ---
@st.cache_data
def fetch_data(ticker, interval, period):
    df = yf.download(ticker, interval=interval, period=period, progress=False)
    return df.dropna()

# --- Indicator and signal logic ---
def prepare_data(df):
    close, high, low = df['Close'], df['High'], df['Low']
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

# --- Backtest Logic ---
def backtest(df):
    trades = []
    position = None
    for i in range(1, len(df)):
        price = df['Close'].iloc[i]
        atr = df['ATR'].iloc[i]

        if position is None:
            if df['Signal'].iloc[i-1] == "BUY" and df['Signal'].iloc[i] == "BUY":
                entry = price
                sl = price - atr
                target = price + 2 * atr
                position = {"type": "BUY", "entry": entry, "sl": sl, "target": target, "entry_time": df.index[i]}
            elif df['Signal'].iloc[i-1] == "SELL" and df['Signal'].iloc[i] == "SELL":
                entry = price
                sl = price + atr
                target = price - 2 * atr
                position = {"type": "SELL", "entry": entry, "sl": sl, "target": target, "entry_time": df.index[i]}
        else:
            # Exit condition
            if position["type"] == "BUY":
                if price <= position["sl"]:
                    pnl = position["sl"] - position["entry"]
                    trades.append({"Exit": "Stoploss", "PnL": pnl, "Type": "BUY", "EntryTime": position["entry_time"], "ExitTime": df.index[i]})
                    position = None
                elif price >= position["target"]:
                    pnl = position["target"] - position["entry"]
                    trades.append({"Exit": "Target", "PnL": pnl, "Type": "BUY", "EntryTime": position["entry_time"], "ExitTime": df.index[i]})
                    position = None
            elif position["type"] == "SELL":
                if price >= position["sl"]:
                    pnl = position["entry"] - position["sl"]
                    trades.append({"Exit": "Stoploss", "PnL": pnl, "Type": "SELL", "EntryTime": position["entry_time"], "ExitTime": df.index[i]})
                    position = None
                elif price <= position["target"]:
                    pnl = position["entry"] - position["target"]
                    trades.append({"Exit": "Target", "PnL": pnl, "Type": "SELL", "EntryTime": position["entry_time"], "ExitTime": df.index[i]})
                    position = None
    return pd.DataFrame(trades)

# --- Main Logic ---
if run_bt:
    df = fetch_data(ticker, interval, period)
    df = prepare_data(df)
    bt = backtest(df)

    if bt.empty:
        st.warning("⚠️ No trades generated with current settings.")
    else:
        total_trades = len(bt)
        wins = len(bt[bt['PnL'] > 0])
        losses = len(bt[bt['PnL'] < 0])
        win_rate = (wins / total_trades) * 100
        total_pnl = bt['PnL'].sum()

        st.subheader(f"📊 Backtest Summary for {ticker}")
        st.metric("Total Trades", total_trades)
        st.metric("Win Rate (%)", round(win_rate, 2))
        st.metric("Total PnL", round(total_pnl, 2))

        st.dataframe(bt, use_container_width=True)

        # --- Equity Curve ---
        bt['Equity'] = bt['PnL'].cumsum()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(bt['Equity'], label='Equity Curve', color='green')
        ax.set_title(f"Equity Curve - {ticker}")
        ax.set_xlabel("Trade #")
        ax.set_ylabel("PnL (Cumulative)")
        ax.legend()
        st.pyplot(fig)
