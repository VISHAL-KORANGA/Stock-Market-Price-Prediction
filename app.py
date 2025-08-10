# Stock Market Prediction Dashboard v2.0 (Logo line removed to avoid error)
import numpy as np
import pandas as pd
import yfinance as yf
import random
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from datetime import datetime

# App Config
st.set_page_config(
    page_title="Stock Market Predictor",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded"
)

#App Header
col_title, = st.columns([6])
with col_title:
    st.title("📈 Stock Market Prediction Dashboard")
    st.markdown(
        "<h5 style='color:#888'>Accurate. Insightful. Attractive.</h5>", unsafe_allow_html=True
    )

with st.expander("ℹ️ How to Use This Dashboard"):
    st.markdown("""
    - 🎯 **Select a stock and date range** from the sidebar
    - 📊 **View real market data, or realistic synthetic data** if live data fails
    - 💡 Check **metrics, trends, candlestick price/volume, and predictions**
    - 💾 Download chart data for offline analysis
    - 🛠️ Advanced indicators coming soon!
    """)

# Sidebar 
st.sidebar.header("⚙️ Configuration Panel")
stocks = {
    "Nifty50": "^NSEI", "Tesla": "TSLA", "Facebook/Instagram": "META",
    "Bitcoin": "BTC-USD", "Google": "GOOG", "Apple": "AAPL",
    "Amazon": "AMZN", "Microsoft": "MSFT", "Netflix": "NFLX", "Twitter": "TWTR"
}
stock_selection = st.sidebar.selectbox(
    "Choose a stock or enter your own", list(stocks.keys()) + ["Custom"]
)
if stock_selection == "Custom":
    stock_symbol = st.sidebar.text_input("Enter stock symbol", "AAPL")
else:
    stock_symbol = stocks[stock_selection]

start_date = st.sidebar.date_input("Start Date", datetime.strptime("2012-01-01", "%Y-%m-%d"))
end_date = st.sidebar.date_input("End Date", datetime.strptime("2022-12-31", "%Y-%m-%d"))

st.sidebar.markdown("---")
with st.sidebar.expander("About/Project Info", expanded=False):
    st.write("""
    **Stock dashboard** app by VISHAL SINGH KORANGA
    - Built with Streamlit, Plotly & Tensorflow
    - Contact: vishalkoranga97@gmail.com
    """)

# Model Caching
MODEL_PATH = "GOOGL_Stock_Prediction_Model.keras"
@st.cache_resource
def load_trained_model(): return load_model(MODEL_PATH)
model = load_trained_model()

# Synthetic Data Fallback
@st.cache_data
def generate_synthetic_data(symbol, start, end):
    dates = pd.date_range(start=start, end=end)
    np.random.seed(0)
    proxy_symbol = symbol if symbol in stocks.values() else "AAPL"
    try:
        base_data = yf.download(proxy_symbol, start=start, end=end, progress=False, threads=True)
    except Exception:
        base_data = pd.DataFrame()
    if base_data.empty:
        base_price, base_vol = 150, 2
        mean_volume, std_volume = 1_000_000, 200_000
    else:
        base_price = base_data['Close'].mean()
        base_vol = base_data['Close'].std()
        mean_volume = base_data['Volume'].mean()
        std_volume = base_data['Volume'].std()
    trend = np.linspace(base_price * 0.95, base_price * 1.05, len(dates))
    noise = np.random.normal(0, base_vol * 0.5, len(dates))
    close = trend + noise
    open_ = close + np.random.normal(0, base_vol*0.05, len(close))
    high = close + np.abs(np.random.normal(0, base_vol*0.1, len(close)))
    low = close - np.abs(np.random.normal(0, base_vol*0.1, len(close)))
    volume = np.abs(np.random.normal(mean_volume, std_volume, len(close)))
    return pd.DataFrame({'Open': open_, 'High': high, 'Low': low, 'Close': close, 'Volume': volume}, index=dates)

@st.cache_data
def load_stock_data(symbol, start, end, max_retries=3):
    import time
    for attempt in range(max_retries):
        try:
            data = yf.download(symbol, start=start, end=end, progress=False, threads=True)
            if not data.empty:
                return data, False
            st.warning(f"Attempt {attempt+1} failed to fetch data for {symbol}. Retrying...")
            time.sleep(2)
        except Exception as e:
            st.warning(f"Attempt {attempt+1} error: {e}")
            time.sleep(2)
    st.warning("Failed to fetch real data, using synthetic data.")
    synthetic_data = generate_synthetic_data(symbol, start, end)
    return synthetic_data, True

#  Data Load 
data, is_synthetic = load_stock_data(stock_symbol, start_date, end_date)

if data is None or data.empty:
    st.error("No stock data found for this symbol and date range.")
    st.stop()

#  Key Metrics Tiles 
latest = data['Close'].iloc[-1]
prev = data['Close'].iloc[-2] if len(data['Close'])>1 else latest
high_52w = data['High'][-252:].max() if len(data['High'])>=252 else data['High'].max()
low_52w = data['Low'][-252:].min() if len(data['Low'])>=252 else data['Low'].min()
daily_change = ((latest - prev) / prev * 100) if pd.notna(prev) and prev != 0 else 0
volatility = data['Close'].pct_change().std()*100

col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest Price", f"${latest:.2f}", f"{daily_change:+.2f}%", delta_color="inverse" if daily_change<0 else "normal", help="Latest market close")
col2.metric("52W High", f"${high_52w:.2f}")
col3.metric("52W Low", f"${low_52w:.2f}")
col4.metric("Volatility", f"{volatility:.2f}%")

#  Synthetic Data Notice 
if is_synthetic:
    st.info(":warning: Shown data is realistically simulated due to data fetching issues. Trends/statistics are approximate!")

#  Calculate MAs & Technicals 
data["MA50"] = data["Close"].rolling(50).mean()
data["MA100"] = data["Close"].rolling(100).mean()
data["MA200"] = data["Close"].rolling(200).mean()
# Bollinger Bands Example
data["BB_up"] = data["MA50"] + 2 * data["Close"].rolling(50).std()
data["BB_dn"] = data["MA50"] - 2 * data["Close"].rolling(50).std()

# Candlestick Chart + Volume
st.subheader("Candlestick Price Chart & Volume")
fig1 = go.Figure()
fig1.add_trace(go.Candlestick(
    x=data.index,
    open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'],
    name='Price', increasing_line_color='lime', decreasing_line_color='orangered'
))
fig1.add_trace(go.Scatter(
    x=data.index, y=data["MA50"], line=dict(color="dodgerblue"), name="MA50"
))
fig1.add_trace(go.Scatter(
    x=data.index, y=data["MA200"], line=dict(color="orange"), name="MA200"
))
fig1.add_trace(go.Scatter(
    x=data.index, y=data["BB_up"], line=dict(color="lightgrey", dash="dot"), name="Bollinger Up", showlegend=False
))
fig1.add_trace(go.Scatter(
    x=data.index, y=data["BB_dn"], line=dict(color="lightgrey", dash="dot"), name="Bollinger Dn", showlegend=False
))
fig1.add_trace(go.Bar(
    x=data.index, y=data['Volume']/data['Volume'].max()*data['High'].max()*0.2, name='Volume', marker=dict(color='grey'),
    yaxis='y2', opacity=0.2
))
fig1.update_layout(
    yaxis=dict(title='Price'),
    yaxis2=dict(title='Volume', overlaying='y', side='right', showgrid=False, showticklabels=False),
    xaxis_title='Date',
    plot_bgcolor="#161a1d",
    legend=dict(x=0,y=1.05, orientation="h"),
    template='plotly_dark', height=600
)
st.plotly_chart(fig1, use_container_width=True)

# Predictions
data_train = data["Close"][: int(len(data) * 0.8)]
data_test = data["Close"][int(len(data) * 0.8):]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_test = scaler.fit_transform(data_test.values.reshape(-1, 1))
x_test, y_test = [], []
for i in range(100, len(scaled_test)):
    x_test.append(scaled_test[i - 100: i])
    y_test.append(scaled_test[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)
if x_test.shape[0] > 0:
    predictions = model.predict(x_test)
    pred = predictions * (1 / scaler.scale_)
    y_test = y_test * (1 / scaler.scale_)
    st.subheader("Original vs Predicted Price")
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(y=y_test, mode="lines", name="Original", line=dict(color="lime")))
    fig_pred.add_trace(go.Scatter(y=pred.flatten(), mode="lines", name="Predicted", line=dict(color="magenta")))
    fig_pred.update_layout(
        title="Original vs Predicted Price", xaxis_title="Test Time", yaxis_title="Price",
        template="plotly_dark", height=400
    )
    st.plotly_chart(fig_pred, use_container_width=True)
else:
    st.warning("Not enough data in selected range for predictions.")

# Additional & Expandable Metrics
with st.expander("📊 Show More Metrics and Explanations", expanded=False):
    st.write("Moving Averages (MA50/MA200), Bollinger Bands, and volume provide insights into trend and volatility. More advanced indicators (RSI, MACD, etc.) coming soon.")

# Data Download 
st.subheader("📥 Download Chart Data")
csv = data.to_csv().encode("utf-8")
st.download_button("Download CSV", csv, f"{stock_symbol}_data.csv", "text/csv")
