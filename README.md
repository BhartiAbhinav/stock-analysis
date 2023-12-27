# Stock Market Analysis & Prediction

A comprehensive stock market analysis tool with live predictions and interactive visualization.

## Project Structure
│ ├── trend_predictor.py # Trend prediction
│ └── model_tuner.py # Hyperparameter tuning
├── analysis/
│ └── backtester.py # Strategy backtesting
└── visualization/
└── plots.py # Data visualization

## Features
- Live stock data analysis and prediction
- Interactive web interface using Streamlit
- Multiple prediction models:
  - LSTM for price prediction
  - Random Forest for trend prediction
  - Technical indicators analysis
- Real-time visualization with Plotly
- Customizable prediction timeframes

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

1. Run the Streamlit interface:

bash
streamlit run src/app.py

2. Using the web interface:
   - Enter a stock symbol (e.g., 'AAPL' for Apple)
   - Adjust prediction days using the slider
   - Click "Analyze" to view:
     - Live price charts
     - Price predictions
     - Technical indicators
     - Trend analysis

3. For programmatic usage:

python
from src.main import main
main()

4. Custom parameters:

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
- Classification-based approach
- Predicts market direction
- Uses technical indicators as features

## Web Interface
The Streamlit interface provides:
- Real-time stock data visualization
- Interactive charts and indicators
- Price predictions and trend analysis
- Customizable analysis parameters
- Three main views:
  1. Price Analysis
  2. Predictions
  3. Technical Indicators

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
Feel free to submit issues, fork the repository, and create pull requests for any improvements.