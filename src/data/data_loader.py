import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_stock_data(ticker, period, interval='1d'):
    """
    Fetch stock data from Yahoo Finance
    
    Args:
        ticker (str): Stock ticker symbol
        period (str): Time period (e.g., '2y', '5y')
        interval (str): Data interval (default: '1d')
        
    Returns:
        pd.DataFrame: Historical stock data
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    return df

def load_multiple_stocks(tickers, periods):
    """
    Load data for multiple stocks and periods
    
    Args:
        tickers (list): List of stock ticker symbols
        periods (list): List of time periods
        
    Returns:
        dict: Nested dictionary containing stock data
    """
    stock_data = {}
    
    for ticker in tickers:
        stock_data[ticker] = {}
        for period in periods:
            stock_data[ticker][period] = get_stock_data(ticker, period)
            
    return stock_data 