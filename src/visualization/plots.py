import matplotlib.pyplot as plt
import seaborn as sns

def plot_stock_analysis(df, title):
    """
    Create visualization of stock data with technical indicators
    
    Args:
        df (pd.DataFrame): DataFrame with stock data and indicators
        title (str): Plot title
    """
    fig = plt.figure(figsize=(15, 10))
    
    # Price and Moving Averages
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(df.index, df['Close'], label='Close Price')
    ax1.plot(df.index, df['MA20'], label='20 Day MA')
    ax1.plot(df.index, df['MA50'], label='50 Day MA')
    ax1.set_title(f'{title} - Stock Price and Moving Averages')
    ax1.set_ylabel('Price')
    ax1.legend()
    
    # RSI
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(df.index, df['RSI'], label='RSI')
    ax2.axhline(y=70, color='r', linestyle='--')
    ax2.axhline(y=30, color='g', linestyle='--')
    ax2.set_title('Relative Strength Index')
    ax2.set_ylabel('RSI')
    ax2.legend()
    
    plt.tight_layout()
    plt.show() 