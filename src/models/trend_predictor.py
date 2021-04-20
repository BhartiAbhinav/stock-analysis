import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

class TrendPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        
    def prepare_trend_data(self, df, lookback_period=1):
        """
        Prepare data for trend prediction
        Returns 1 for upward trend, 0 for downward trend
        """
        # Calculate price changes
        df = df.copy()  # Create a copy to avoid SettingWithCopyWarning
        df['Price_Change'] = df['Close'].diff(lookback_period)
        df['Trend'] = (df['Price_Change'] > 0).astype(int)
        
        # Create feature set
        features = [
            'MA5', 'MA20', 'MA50',
            'RSI', 'MACD', 'Signal_Line',
            'BB_upper', 'BB_lower',
            'Volume_MA5'
        ]
        
        # Make sure we have the same length for features and target
        feature_df = df[features].dropna()
        trend_df = df['Trend'].dropna()
        
        # Align the indices
        common_idx = feature_df.index.intersection(trend_df.index)
        return feature_df.loc[common_idx], trend_df.loc[common_idx]
    
    def train(self, X_train, y_train):
        """Train the trend prediction model"""
        self.model.fit(X_train, y_train)
        
    def predict(self, X):
        """Predict trend direction"""
        return self.model.predict(X)
    
    def evaluate(self, y_true, y_pred):
        """Evaluate trend prediction performance"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred)
        } 