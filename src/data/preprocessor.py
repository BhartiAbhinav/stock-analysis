import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class DataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def prepare_data(self, df, sequence_length=60, target_column='Close', train_split=0.8):
        """
        Prepare data for model training
        
        Args:
            df: DataFrame with features
            sequence_length: Number of time steps for LSTM
            target_column: Column to predict
            train_split: Train/test split ratio
            
        Returns:
            Dictionary containing processed datasets
        """
        # Drop any missing values
        df = df.dropna()
        
        # Scale the features
        scaled_data = self.scaler.fit_transform(df)
        scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
        
        # Create sequences for LSTM
        X, y = self._create_sequences(scaled_df, sequence_length, target_column)
        
        # Split into train and test sets
        split_idx = int(len(X) * train_split)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': self.scaler,
            'feature_columns': df.columns
        }
    
    def _create_sequences(self, df, sequence_length, target_column):
        """Create sequences for LSTM model"""
        X = []
        y = []
        
        for i in range(len(df) - sequence_length):
            X.append(df.iloc[i:(i + sequence_length)].values)
            y.append(df.iloc[i + sequence_length][target_column])
            
        return np.array(X), np.array(y) 