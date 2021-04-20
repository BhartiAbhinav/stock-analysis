import pandas as pd
import numpy as np

class Backtester:
    def __init__(self, initial_capital=100000.0):
        self.initial_capital = initial_capital
        self.positions = []
        self.portfolio_value = []
        
    def generate_signals(self, df, technical_predictions, lstm_predictions):
        """Generate trading signals based on model predictions"""
        # Reshape LSTM predictions to match technical predictions
        lstm_predictions = lstm_predictions.reshape(-1)
        
        # Ensure all arrays have the same length
        min_length = min(len(technical_predictions), len(lstm_predictions))
        technical_predictions = technical_predictions[-min_length:]
        lstm_predictions = lstm_predictions[-min_length:]
        
        # Get the correct window of close prices
        close_prices = df['Close'].values[-min_length:]
        
        # Create signals DataFrame with the correct index
        signals = pd.DataFrame(index=df.index[-min_length:])
        
        # Generate signals
        signals['Technical_Signal'] = np.where(technical_predictions > close_prices, 1, -1)
        signals['LSTM_Signal'] = np.where(lstm_predictions > close_prices, 1, -1)
        signals['Combined_Signal'] = np.where(
            (signals['Technical_Signal'] + signals['LSTM_Signal']) > 0, 1, -1
        )
        return signals
    
    def backtest_strategy(self, df, signals, transaction_cost=0.001):
        """
        Backtest trading strategy
        
        Args:
            df: DataFrame with price data
            signals: DataFrame with trading signals
            transaction_cost: Transaction cost as percentage
        """
        portfolio = pd.DataFrame(index=signals.index)
        portfolio['Position'] = signals['Combined_Signal']
        portfolio['Close'] = df['Close']
        
        # Calculate returns
        portfolio['Returns'] = portfolio['Close'].pct_change()
        portfolio['Strategy_Returns'] = portfolio['Position'].shift(1) * portfolio['Returns']
        
        # Account for transaction costs
        portfolio['Trade'] = portfolio['Position'].diff().abs()
        portfolio['Transaction_Costs'] = portfolio['Trade'] * transaction_cost
        portfolio['Strategy_Returns'] = portfolio['Strategy_Returns'] - portfolio['Transaction_Costs']
        
        # Calculate portfolio value
        portfolio['Portfolio_Value'] = (1 + portfolio['Strategy_Returns']).cumprod() * self.initial_capital
        
        # Calculate metrics
        total_return = (portfolio['Portfolio_Value'].iloc[-1] - self.initial_capital) / self.initial_capital
        sharpe_ratio = np.sqrt(252) * (portfolio['Strategy_Returns'].mean() / portfolio['Strategy_Returns'].std())
        max_drawdown = (portfolio['Portfolio_Value'] / portfolio['Portfolio_Value'].cummax() - 1).min()
        
        return {
            'portfolio': portfolio,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        } 