import os
import requests
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from transformers import pipeline, BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import xgboost as xgb
from ta.momentum import RSIIndicator
from ta.trend import MACD
import plotly.graph_objects as go
import plotly.express as px


# Disable tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# API Key
POLYGON_API_KEY = "Xfls6mGPBAT1aRAXMrpWs7vdiXUbSUv4"

# Load FinBERT for sentiment analysis
MODEL_NAME = "ProsusAI/finbert"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
sentiment_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)

# 📌 --- STOCK DATA ---
def get_stock_historical_data(ticker, start, end):
    start, end = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}?apiKey={POLYGON_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json().get("results", [])
        df = pd.DataFrame(data)
        df["t"] = pd.to_datetime(df["t"], unit="ms")
        return df
    except:
        return pd.DataFrame()

# 📌 --- SENTIMENT ANALYSIS ---
def get_stock_news(ticker):
    url = f"https://api.polygon.io/v2/reference/news?ticker={ticker}&apiKey={POLYGON_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return [{"headline": article.get('title', ''), "summary": article.get('description', '')}
                for article in response.json().get("results", [])]
    except:
        return []

def analyze_sentiment(text):
    result = sentiment_pipeline(text)
    return result[0]['label']

def get_sentiment_score(label):
    return {"POSITIVE": 1, "NEUTRAL": 0, "NEGATIVE": -1}.get(label.upper(), 0)

def process_sentiment(ticker):
    news = get_stock_news(ticker)
    sentiments = []
    for article in news[:10]:  # Limit to 10 articles
        combined = article["headline"] + " " + article["summary"]
        sentiment = analyze_sentiment(combined)
        score = get_sentiment_score(sentiment)
        sentiments.append({"date": datetime.now().date(), "headline": article["headline"], "sentiment": sentiment, "score": score})
    return pd.DataFrame(sentiments)

# 📌 --- FEATURE ENGINEERING ---
def add_technical_indicators(df):
    df['SMA'] = df['c'].rolling(window=10).mean()
    df['RSI'] = RSIIndicator(df['c']).rsi()
    macd = MACD(df['c'])
    df['MACD'] = macd.macd_diff()
    return df

def create_lag_features(df, lags=[1, 2, 3]):
    for lag in lags:
        df[f"lag_{lag}"] = df["c"].shift(lag)
    return df

# 📌 --- MODELING ---
def prepare_data(df, sentiment_df):
    df = add_technical_indicators(df)
    df = create_lag_features(df)
    df = df.dropna()

    # Merge sentiment
    avg_sentiment = sentiment_df.groupby("date")["score"].mean().reset_index()
    df["date"] = df["t"].dt.date
    df = df.merge(avg_sentiment, how="left", left_on="date", right_on="date")
    df["score"] = df["score"].fillna(0)

    # Features and targets
    features = ["SMA", "RSI", "MACD", "lag_1", "lag_2", "lag_3", "v", "score"]
    target = "c"

    X = df[features]
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

    return X_scaled, y_scaled, df["t"].values, scaler, scaler_y, df

def train_lstm(X, y):
    X_lstm = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    model = Sequential()
    model.add(LSTM(64, input_shape=(X_lstm.shape[1], X_lstm.shape[2]), return_sequences=False))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_lstm, y, epochs=10, batch_size=8, verbose=0)
    return model

def train_xgboost(X, y):
    xgb_model = xgb.XGBRegressor(n_estimators=100)
    xgb_model.fit(X, y)
    return xgb_model

# 📌 --- STREAMLIT UI ---
st.set_page_config("📈 Stock Hybrid Model", layout="wide")
st.title("📊 Stock Forecasting with LSTM + XGBoost & Sentiment Analysis")

ticker = st.text_input("Enter Stock Ticker:", "AAPL")
end_date = datetime.now()
start_date = end_date - timedelta(days=180)
start = st.date_input("Start Date", start_date)
end = st.date_input("End Date", end_date)

if st.button("Run Hybrid Model"):
    sentiment_df = process_sentiment(ticker)
    hist_df = get_stock_historical_data(ticker, start, end)

    if hist_df.empty:
        st.error("No stock data found.")
    else:
        X, y, dates, scaler, scaler_y, processed_df = prepare_data(hist_df, sentiment_df)
        lstm_model = train_lstm(X, y)
        xgb_model = train_xgboost(X, y)

        # 📰 Sentiment Analysis
        st.subheader("📰 Sentiment Analysis")
        st.dataframe(sentiment_df)

        # Make Predictions
        lstm_preds = lstm_model.predict(np.reshape(X, (X.shape[0], 1, X.shape[1]))).flatten()

        # Inverse transform predictions for display
        lstm_preds = scaler_y.inverse_transform(lstm_preds.reshape(-1, 1)).flatten()

        hybrid_input = np.concatenate([X, lstm_preds.reshape(-1, 1)], axis=1)
        hybrid_model = train_xgboost(hybrid_input, y)
        hybrid_preds = hybrid_model.predict(hybrid_input).reshape(-1, 1)

        # Inverse transform hybrid predictions for display
        hybrid_preds = scaler_y.inverse_transform(hybrid_preds).flatten()


        # 📊 STOCK PRICE & TECHNICAL INDICATORS PLOTS
        st.subheader("📈 Historical Stock Data & Technical Indicators")

        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=processed_df["t"], y=processed_df["c"], 
                                        mode='lines', name='Closing Price', line=dict(color='blue')))
        fig_price.add_trace(go.Scatter(x=processed_df["t"], y=processed_df["SMA"], 
                                        mode='lines', name='SMA', line=dict(color='orange', dash='dot')))
        fig_price.update_layout(title=f"{ticker} Stock Price & SMA",
                                xaxis_title="Date", yaxis_title="Price", template="plotly_dark")
        st.plotly_chart(fig_price)

        # 📊 Volume Traded
        st.subheader("📉 Trading Volume Over Time")
        fig_volume = px.bar(processed_df, x="t", y="v", labels={"v": "Volume"}, title=f"{ticker} Trading Volume")
        st.plotly_chart(fig_volume)

        # 📊 RSI & MACD Trend Plots
        st.subheader("📊 RSI & MACD Trends")

        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=processed_df["t"], y=processed_df["RSI"], mode='lines', name='RSI', line=dict(color='purple')))
        fig_rsi.update_layout(title="RSI Indicator", xaxis_title="Date", yaxis_title="RSI", template="plotly_dark")
        st.plotly_chart(fig_rsi)

        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=processed_df["t"], y=processed_df["MACD"], mode='lines', name='MACD', line=dict(color='green')))
        fig_macd.update_layout(title="MACD Trend", xaxis_title="Date", yaxis_title="MACD", template="plotly_dark")
        st.plotly_chart(fig_macd)

        # 📉 ACTUAL vs HYBRID MODEL PREDICTIONS
        st.subheader("Historic Stock Prices")

        fig_comparison = go.Figure()
        fig_comparison.add_trace(go.Scatter(x=dates, y=processed_df["c"], mode='lines', name='Actual Price', line=dict(color='blue')))
        fig_comparison.update_layout(title="Historic Stock Prices",
                                     xaxis_title="Date", yaxis_title="Stock Price", template="plotly_dark")
        st.plotly_chart(fig_comparison)

        # 📉 Future Predictions
        future_days = 10
        future_predictions = []
        last_known_X = X[-1]

        for _ in range(future_days):
            # Predict with LSTM
            lstm_pred = lstm_model.predict(np.reshape(last_known_X.reshape(1, 1, -1), (1, 1, -1))).flatten()[0]
            
            # Update features for next prediction
            new_features = np.roll(last_known_X, -1)
            new_features[-1] = lstm_pred  # Assuming the last feature is the target
            
            # Predict with Hybrid Model
            hybrid_input = np.concatenate([new_features.reshape(1, -1), np.array([[lstm_pred]])], axis=1)
            hybrid_pred = hybrid_model.predict(hybrid_input).flatten()[0]  # Get prediction
            future_predictions.append(hybrid_pred)  # Append corrected hybrid prediction
            
            # Update last_known_X for next iteration
            last_known_X = new_features

        # Inverse transform future predictions
        future_predictions = scaler_y.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

        future_dates = [(pd.to_datetime(dates[-1]) + timedelta(days=i)).date() for i in range(1, future_days + 1)]
        future_df = pd.DataFrame({"Date": future_dates, "Predicted Price": future_predictions})

        # 📊 Display Predictions
        st.write("📅 **Future Predicted Prices**")
        st.dataframe(future_df)

        # 📈 FUTURE PRICE PREDICTIONS PLOT
        st.subheader("📈 Future Price Predictions")

        fig_future = go.Figure()
        fig_future.add_trace(go.Scatter(x=future_df["Date"], y=future_df["Predicted Price"],
                                mode='lines+markers', name='Future Predictions', line=dict(color='red')))
        fig_future.update_layout(title="Future Stock Price Predictions",
                         xaxis_title="Date", yaxis_title="Predicted Price", template="plotly_dark")
        st.plotly_chart(fig_future)

