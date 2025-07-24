# Stock Market Price Prediction
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from datetime import datetime

# Constants
MODEL_PATH = "C:\\Users\\vishal\\Documents\\1.VISHAL\\PROJECTS\\Stock_Market_Prediction\\Stock Predictions Model.keras"
DEFAULT_START_DATE = "2012-01-01"
DEFAULT_END_DATE = "2024-12-31"

# Streamlit App Layout
st.set_page_config(page_title="Stock Market Predictor", layout="wide")
st.title("📈 Stock Market Prediction Dashboard")
st.sidebar.header("⚙️ Configuration Panel")

# Sidebar: Stock Selection with Additional Stocks
stocks = {
    "Nifty50": "^NSEI",
    "Tesla": "TSLA",
    "Facebook/Instagram": "META",
    "Bitcoin": "BTC-USD",
    "Google": "GOOG",
    "Apple": "AAPL",
    "Amazon": "AMZN",
    "Microsoft": "MSFT",
    "Netflix": "NFLX",
    "Twitter": "TWTR",
}

stock_selection = st.sidebar.selectbox(
    "Choose a stock or enter your own:", list(stocks.keys()) + ["Custom"]
)

# If user chooses "Custom", prompt them to enter a custom stock symbol
if stock_selection == "Custom":
    stock_symbol = st.sidebar.text_input("Enter stock symbol", "AAPL")

else:
    stock_symbol = stocks[stock_selection]

# Sidebar: Date Range Selection
start_date = st.sidebar.date_input(
    "Start Date", datetime.strptime(DEFAULT_START_DATE, "%Y-%m-%d")
)
end_date = st.sidebar.date_input(
    "End Date", datetime.strptime(DEFAULT_END_DATE, "%Y-%m-%d")
)


# Caching Model Loading
@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH)


model = load_trained_model()


# Caching Data Loading
@st.cache_data
def load_stock_data(symbol, start, end):
    try:
        data = yf.download(symbol, start=start, end=end)
        if data.empty:
            st.error("No data found for the selected stock symbol.")
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None


data = load_stock_data(stock_symbol, start_date, end_date)

if data is not None:
    # Stock Overview
    st.sidebar.subheader("📊 Stock Overview")
    st.sidebar.write(f"**Latest Close Price**: ${data['Close'].iloc[-1]:.2f}")
    #st.sidebar.write(f"**max value**:${data['close'].mean()}")
    st.sidebar.write(f"**Highest Price**: ${data['High'].max():.2f}")
    st.sidebar.write(f"**Lowest Price**: ${data['Low'].min():.2f}")
    st.sidebar.write(f"**Average Volume**: {data['Volume'].mean():,.0f}")

    # Data Splitting
    data_train = data["Close"][: int(len(data) * 0.8)]
    data_test = data["Close"][int(len(data) * 0.8) :]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_test = scaler.fit_transform(data_test.values.reshape(-1, 1))

    # Moving Averages
    data["MA50"] = data["Close"].rolling(50).mean()
    data["MA100"] = data["Close"].rolling(100).mean()
    data["MA200"] = data["Close"].rolling(200).mean()

    # Plot Price and Moving Averages
    st.subheader("Price and Moving Averages")
    data['stock code']=stock_symbol
    st.write(data)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Close"],
            mode="lines",
            name="Close Price",
            line=dict(color="green"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["MA50"],
            mode="lines",
            name="MA50",
            line=dict(color="red"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["MA100"],
            mode="lines",
            name="MA100",
            line=dict(color="blue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["MA200"],
            mode="lines",
            name="MA200",
            line=dict(color="orange"),
        )
    )
    fig.update_layout(
        title="Price and Moving Averages",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Prediction
    x_test, y_test = [], []
    for i in range(100, len(scaled_test)):
        x_test.append(scaled_test[i - 100 : i])
        y_test.append(scaled_test[i, 0])
    x_test, y_test = np.array(x_test), np.array(y_test)

    predictions = model.predict(x_test)
    scale = 1 / scaler.scale_
    predictions = predictions * scale
    y_test = y_test * scale

    # Plot Predictions vs Original
    st.subheader("Original Price vs Predicted Price")
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            y=y_test, mode="lines", name="Original Price", line=dict(color="green")
        )
    )
    fig2.add_trace(
        go.Scatter(
            y=predictions.flatten(),
            mode="lines",
            name="Predicted Price",
            line=dict(color="red"),
        )
    )
    fig2.update_layout(
        title="Original vs Predicted Price",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_dark",
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Data Download Option
    st.subheader("📥 Download Processed Data")
    csv = data.to_csv().encode("utf-8")
    st.download_button("Download CSV", csv, f"{stock_symbol}_data.csv", "text/csv")

else:
    st.error("Failed to load stock data. Please adjust the input parameters.")
