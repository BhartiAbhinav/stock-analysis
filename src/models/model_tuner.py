from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.optimizers.legacy import Adam

class ModelTuner:
    @staticmethod
    def tune_traditional_model(model, X_train, y_train):
        """Tune traditional model hyperparameters"""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_, grid_search.best_params_
    
    @staticmethod
    def create_lstm_model(hp):
        """Create LSTM model with hyperparameters"""
        model = Sequential([
            LSTM(units=hp['lstm_units'], 
                 return_sequences=True,
                 input_shape=(hp['sequence_length'], hp['n_features'])),
            Dropout(hp['dropout_rate']),
            LSTM(units=hp['lstm_units'], return_sequences=False),
            Dropout(hp['dropout_rate']),
            Dense(units=hp['dense_units']),
            Dense(units=1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=hp['learning_rate']),
            loss='mean_squared_error'
        )
        return model
    
    @staticmethod
    def tune_lstm_model(X_train, y_train, sequence_length, n_features):
        """Tune LSTM model hyperparameters"""
        param_grid = {
            'lstm_units': [32, 50, 64],
            'dense_units': [16, 25, 32],
            'dropout_rate': [0.1, 0.2, 0.3],
            'learning_rate': [0.001, 0.0001],
            'sequence_length': [sequence_length],
            'n_features': [n_features]
        }
        
        model = KerasRegressor(build_fn=ModelTuner.create_lstm_model)
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=3,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_, grid_search.best_params_ 