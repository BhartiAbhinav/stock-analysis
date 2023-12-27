import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Import our existing modules
from models.lstm_model import StockLSTM
from models.trend_predictor import TrendPredictor
from models.traditional_model import TechnicalModel
from data.preprocessor import DataPreprocessor
from features.technical_indicators import add_technical_indicators
from features.advanced_indicators import add_advanced_indicators

def main():
    st.set_page_config(page_title="Stock Market Predictor", layout="wide")
    st.title("Live Stock Market Analysis & Prediction")
    
    # Sidebar inputs
    st.sidebar.header("Settings")
    ticker = st.sidebar.text_input("Stock Symbol", value="AAPL")
    prediction_days = st.sidebar.slider("Prediction Days", 1, 30, 7)
    
    # Fetch live data
    if st.sidebar.button("Analyze"):
        with st.spinner("Fetching and analyzing data..."):
            # Get historical data
            stock = yf.Ticker(ticker)
            df = stock.history(period="2y")
            
            if len(df) == 0:
                st.error("No data found for this ticker symbol!")
                return
            
            # Process data
            df = add_technical_indicators(df)
            df = add_advanced_indicators(df)
            
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["Price Analysis", "Predictions", "Technical Indicators"])
            
            with tab1:
                plot_stock_data(df, ticker)
            
            with tab2:
                make_predictions(df, prediction_days)
            
            with tab3:
                plot_technical_indicators(df)

def plot_stock_data(df, ticker):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='OHLC'
    ))
    fig.update_layout(title=f"{ticker} Stock Price", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

def make_predictions(df, prediction_days):
    # Prepare data
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.prepare_data(df, sequence_length=60)
    
    # Train and predict using LSTM
    lstm_model = StockLSTM(
        sequence_length=60,
        n_features=len(processed_data['feature_columns'])
    )
    
    lstm_model.train(
        processed_data['X_train'],
        processed_data['y_train'],
        epochs=10  # Reduced epochs for demo
    )
    
    # Make predictions
    last_sequence = processed_data['X_test'][-1:]
    lstm_pred = lstm_model.predict(last_sequence)[0][0]  # Extract the single value
    
    # Trend prediction
    trend_predictor = TrendPredictor()
    X_trend, y_trend = trend_predictor.prepare_trend_data(df)
    trend_predictor.train(X_trend[:-prediction_days], y_trend[:-prediction_days])
    trend_pred = trend_predictor.predict(X_trend[-prediction_days:])
    
    # Display predictions
    st.subheader("Predictions")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Next Day Price Prediction",
            value=f"${lstm_pred:.2f}",
            delta=f"{((lstm_pred - df['Close'].iloc[-1])/df['Close'].iloc[-1]*100):.2f}%"
        )
    
    with col2:
        trend_direction = "Upward" if trend_pred[-1] == 1 else "Downward"
        st.metric(
            label="Trend Direction",
            value=trend_direction
        )

def plot_technical_indicators(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'))
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=30, line_dash="dash", line_color="green")
    fig.update_layout(title="RSI Indicator", xaxis_title="Date", yaxis_title="RSI")
    st.plotly_chart(fig, use_container_width=True)
    
    # MACD Plot
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'))
    fig2.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], name='Signal Line'))
    fig2.update_layout(title="MACD", xaxis_title="Date", yaxis_title="Value")
    st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    main() 