import pandas as pd
import numpy as np

def add_advanced_indicators(df):
    """Add advanced technical indicators"""
    
    # Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
    
    # Stochastic Oscillator
    low_min = df['Low'].rolling(14).min()
    high_max = df['High'].rolling(14).max()
    df['%K'] = ((df['Close'] - low_min) / (high_max - low_min)) * 100
    df['%D'] = df['%K'].rolling(3).mean()
    
    # Price Rate of Change (ROC)
    df['ROC'] = df['Close'].pct_change(periods=12) * 100
    
    # Commodity Channel Index (CCI)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    tp_sma = typical_price.rolling(window=20).mean()
    mad = abs(typical_price - tp_sma).rolling(window=20).mean()
    df['CCI'] = (typical_price - tp_sma) / (0.015 * mad)
    
    return df 