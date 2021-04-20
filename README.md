# Stock Market Analysis and Trading System

A comprehensive Python-based system for stock market analysis, prediction, and automated trading strategy backtesting.

## Features

- **Data Loading and Processing**
  - Multiple stock symbols and time periods support
  - Technical indicators calculation
  - Advanced market indicators
  - Data preprocessing and normalization

- **Multiple Models**
  - LSTM (Long Short-Term Memory) for sequence prediction
  - Traditional ML model using technical indicators
  - Trend direction prediction
  - Model hyperparameter tuning

- **Trading Strategy**
  - Combined signals from multiple models
  - Backtesting functionality
  - Transaction costs consideration
  - Portfolio performance metrics

## Project Structure
src/

├── data/

│ ├── data_loader.py # Stock data loading

│ └── preprocessor.py # Data preprocessing

├── features/

│ ├── technical_indicators.py # Basic indicators

│ └── advanced_indicators.py # Advanced indicators

├── models/

│ ├── lstm_model.py # LSTM implementation

│ ├── traditional_model.py # Technical analysis model

│ ├── trend_predictor.py # Trend prediction

│ └── model_tuner.py # Hyperparameter tuning

├── analysis/

│ └── backtester.py # Strategy backtesting

└── visualization/

└── plots.py # Data visualization


## Installation

1. Clone the repository:

bash
git clone https://github.com/yourusername/stock-analysis.git
cd stock-analysis

2. Create a virtual environment:

bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

3. Install the required packages:

bash
pip install -r requirements.txt
## Usage

1. Basic usage:

python
from src.main import main
main()

2. Custom parameters:
python
Initialize parameters
tickers = ['AAPL', 'MSFT'] # Stock symbols
periods = ['2y', '5y'] # Time periods
sequence_length = 60 # For LSTM prediction


## Models

### LSTM Model
- Sequence-based prediction
- Multiple LSTM layers with dropout
- Dense layers for final prediction

### Traditional Model
- Random Forest Regressor
- Technical indicators as features
- Price prediction based on market indicators

### Trend Predictor
- Classification model for trend direction
- Feature engineering for trend prediction
- Accuracy and classification metrics

## Technical Indicators

The system calculates various technical indicators including:
- Moving Averages (MA5, MA20, MA50)
- Relative Strength Index (RSI)
- MACD
- Bollinger Bands
- Volume indicators
- Advanced indicators (ATR, OBV, Stochastic Oscillator, etc.)

## Backtesting

The backtesting system includes:
- Signal generation from multiple models
- Transaction cost consideration
- Portfolio value tracking
- Performance metrics calculation

## Requirements

- Python 3.8+
- TensorFlow 2.x
- Pandas
- NumPy
- Scikit-learn
- Keras

## License

MIT License

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request