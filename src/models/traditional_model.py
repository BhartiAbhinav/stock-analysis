from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class TechnicalModel:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
        
    def prepare_features(self, df):
        """Prepare features for traditional model"""
        features = [
            'MA5', 'MA20', 'MA50',
            'RSI', 'MACD', 'Signal_Line',
            'BB_upper', 'BB_lower',
            'Volume_MA5'
        ]
        
        return df[features], df['Close']
    
    def train(self, X_train, y_train):
        """Train the model"""
        self.model.fit(X_train, y_train)
        
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def evaluate(self, y_true, y_pred):
        """Evaluate model performance"""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        } 