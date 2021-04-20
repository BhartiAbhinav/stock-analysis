from data.data_loader import load_multiple_stocks
from features.technical_indicators import add_technical_indicators
from data.preprocessor import DataPreprocessor
from models.lstm_model import StockLSTM
from models.traditional_model import TechnicalModel
from visualization.plots import plot_stock_analysis
from models.trend_predictor import TrendPredictor
from models.model_tuner import ModelTuner
from features.advanced_indicators import add_advanced_indicators
from analysis.backtester import Backtester
from models.lstm_model import StockLSTM
from models.traditional_model import TechnicalModel
from models.trend_predictor import TrendPredictor
from models.model_tuner import ModelTuner
from features.advanced_indicators import add_advanced_indicators
from analysis.backtester import Backtester

def main():
    # Initialize parameters
    tickers = ['AAPL', 'MSFT']
    periods = ['2y', '5y']
    sequence_length = 60
    
    # Load and process data
    stock_data = load_multiple_stocks(tickers, periods)
    preprocessor = DataPreprocessor()
    
    for ticker in stock_data:
        for period in stock_data[ticker]:
            print(f"\nProcessing {ticker} - {period}")
            
            # Get and process data
            df = stock_data[ticker][period]
            df = add_technical_indicators(df)
            
            # Add advanced indicators
            df = add_advanced_indicators(df)
            
            # Prepare data for both models
            processed_data = preprocessor.prepare_data(df, sequence_length)
            
            # Train and evaluate LSTM model
            lstm_model = StockLSTM(
                sequence_length=sequence_length,
                n_features=len(processed_data['feature_columns'])
            )
            
            lstm_history = lstm_model.train(
                processed_data['X_train'],
                processed_data['y_train'],
                validation_data=(processed_data['X_test'], processed_data['y_test'])
            )
            
            # Get LSTM predictions
            lstm_predictions = lstm_model.predict(processed_data['X_test'])
            
            # Train and evaluate Traditional model
            tech_model = TechnicalModel()
            X_tech, y_tech = tech_model.prepare_features(df)
            split_idx = int(len(X_tech) * 0.8)
            
            tech_model.train(
                X_tech[:split_idx],
                y_tech[:split_idx]
            )
            
            tech_predictions = tech_model.predict(X_tech[split_idx:])
            tech_metrics = tech_model.evaluate(y_tech[split_idx:], tech_predictions)
            
            # Trend prediction
            trend_predictor = TrendPredictor()
            X_trend, y_trend = trend_predictor.prepare_trend_data(df)
            split_idx_trend = int(len(X_trend) * 0.8)
            
            # Make sure we use the same indices for training and testing
            trend_predictor.train(X_trend[:split_idx_trend], y_trend[:split_idx_trend])
            trend_predictions = trend_predictor.predict(X_trend[split_idx_trend:])
            trend_metrics = trend_predictor.evaluate(y_trend[split_idx_trend:], trend_predictions)
            
            # Model tuning
            tuned_tech_model, tech_params = ModelTuner.tune_traditional_model(
                tech_model.model, X_tech[:split_idx], y_tech[:split_idx]
            )
            
            # Backtesting
            backtester = Backtester()

            # Get the minimum length of predictions
            min_length = min(len(tech_predictions), len(lstm_predictions))

            # Use the correct window of test data
            test_data = df.iloc[-min_length:]

            signals = backtester.generate_signals(
                test_data,
                tech_predictions[-min_length:],
                lstm_predictions[-min_length:]
            )
            backtest_results = backtester.backtest_strategy(test_data, signals)
            
            # Plot results
            plot_stock_analysis(df, f'{ticker} - {period}')
            
            print(f"Technical Model Metrics:")
            for metric, value in tech_metrics.items():
                print(f"{metric}: {value:.4f}")
            
            print("\nTrend Prediction Metrics:")
            print(trend_metrics['classification_report'])
            
            print("\nBacktesting Results:")
            print(f"Total Return: {backtest_results['total_return']:.2%}")
            print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
            print(f"Maximum Drawdown: {backtest_results['max_drawdown']:.2%}")

if __name__ == "__main__":
    main() 