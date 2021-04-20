import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam

class StockLSTM:
    def __init__(self, sequence_length, n_features):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = self._build_model()
        
    def _build_model(self):
        """Build LSTM model architecture"""
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(units=50, 
                      return_sequences=True,
                      input_shape=(self.sequence_length, self.n_features)))
        model.add(Dropout(0.2))
        
        # Second LSTM layer
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        
        # Dense layers
        model.add(Dense(units=25))
        model.add(Dense(units=1))
        
        # Compile model
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='mean_squared_error')
        return model
    
    def train(self, X_train, y_train, validation_data=None, epochs=50, batch_size=32):
        """Train the model"""
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        return history
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X) 