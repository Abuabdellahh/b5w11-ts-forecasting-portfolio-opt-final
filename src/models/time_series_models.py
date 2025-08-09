"""
Time series forecasting models for financial data.
"""
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesForecaster:
    """Base class for time series forecasting."""
    
    def __init__(self, train_data: pd.Series, test_data: pd.Series):
        """
        Initialize the forecaster with training and test data.
        
        Args:
            train_data: Training data (pandas Series)
            test_data: Test data (pandas Series)
        """
        self.train_data = train_data
        self.test_data = test_data
        self.model = None
        self.forecast = None
        self.history = None
    
    def train(self, **kwargs):
        """Train the model. To be implemented by subclasses."""
        raise NotImplementedError
    
    def forecast_future(self, steps: int) -> np.ndarray:
        """
        Forecast future values.
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Array of forecasted values
        """
        raise NotImplementedError
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Returns:
            Dictionary with MAE, RMSE, and MAPE
        """
        if self.forecast is None or len(self.forecast) != len(self.test_data):
            raise ValueError("Model not trained or forecast doesn't match test data length")
        
        y_true = self.test_data.values
        y_pred = self.forecast
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
    
    def plot_forecast(self, title: str = 'Forecast vs Actuals'):
        """Plot the forecast against actual values."""
        if self.forecast is None:
            raise ValueError("No forecast available. Train the model first.")
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.train_data.index, self.train_data, label='Training Data')
        plt.plot(self.test_data.index, self.test_data, label='Actual Test Data')
        plt.plot(self.test_data.index, self.forecast, label='Forecast', alpha=0.7)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        # Save the figure
        plt.savefig(f'reports/figures/{title.lower().replace(" ", "_")}.png')
        plt.close()


class ARIMAForecaster(TimeSeriesForecaster):
    """ARIMA/SARIMA based time series forecaster."""
    
    def __init__(self, train_data: pd.Series, test_data: pd.Series, 
                 seasonal_period: int = 5):
        """
        Initialize ARIMA forecaster.
        
        Args:
            train_data: Training data
            test_data: Test data
            seasonal_period: Seasonal period for SARIMA (default: 5 for weekly seasonality)
        """
        super().__init__(train_data, test_data)
        self.seasonal_period = seasonal_period
        self.best_params = None
    
    def _find_best_arima_params(self) -> Tuple[int, int, int]:
        """Find the best ARIMA parameters using auto_arima."""
        model = auto_arima(
            self.train_data,
            seasonal=True,
            m=self.seasonal_period,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            trace=False
        )
        return model.order, model.seasonal_order
    
    def train(self, **kwargs):
        """Train the ARIMA/SARIMA model."""
        # Find best parameters if not provided
        order = kwargs.get('order')
        seasonal_order = kwargs.get('seasonal_order')
        
        if order is None or seasonal_order is None:
            (p, d, q), (sp, sd, sq, sm) = self._find_best_arima_params()
            order = (p, d, q)
            seasonal_order = (sp, sd, sq, sm)
        
        self.best_params = {
            'order': order,
            'seasonal_order': seasonal_order
        }
        
        # Fit the model
        self.model = SARIMAX(
            self.train_data,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        self.history = self.model.fit(disp=0)
        
        # Forecast on test data
        self.forecast = self.history.get_forecast(steps=len(self.test_data)).predicted_mean
        
        return self.history
    
    def forecast_future(self, steps: int) -> np.ndarray:
        """Forecast future values."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get forecast for future steps
        forecast_result = self.history.get_forecast(steps=steps)
        return forecast_result.predicted_mean.values


class LSTMForecaster(TimeSeriesForecaster):
    """LSTM based time series forecaster."""
    
    def __init__(self, train_data: pd.Series, test_data: pd.Series, 
                 look_back: int = 30, epochs: int = 50, batch_size: int = 32):
        """
        Initialize LSTM forecaster.
        
        Args:
            train_data: Training data
            test_data: Test data
            look_back: Number of previous time steps to use for prediction
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        super().__init__(train_data, test_data)
        self.look_back = look_back
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None
    
    def _create_dataset(self, data: np.ndarray, look_back: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Create dataset for LSTM."""
        X, y = [], []
        for i in range(len(data) - look_back - 1):
            X.append(data[i:(i + look_back), 0])
            y.append(data[i + look_back, 0])
        return np.array(X), np.array(y)
    
    def _build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Build LSTM model."""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def train(self, **kwargs):
        """Train the LSTM model."""
        # Scale the data
        train_data = self.train_data.values.reshape(-1, 1)
        scaled_train = self.scaler.fit_transform(train_data)
        
        # Create training dataset
        X_train, y_train = self._create_dataset(scaled_train, self.look_back)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        
        # Store for later use
        self.X_train, self.y_train = X_train, y_train
        
        # Build and train the model
        self.model = self._build_model((X_train.shape[1], 1))
        
        # Early stopping to prevent overfitting
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=1
        )
        
        # Prepare test data for evaluation
        test_data = self.test_data.values.reshape(-1, 1)
        scaled_test = self.scaler.transform(test_data)
        
        # Create test dataset
        X_test, y_test = self._create_dataset(scaled_test, self.look_back)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        # Store for later use
        self.X_test, self.y_test = X_test, y_test
        
        # Make predictions
        test_predict = self.model.predict(X_test)
        
        # Invert scaling for forecast
        test_predict = self.scaler.inverse_transform(test_predict)
        self.forecast = test_predict.flatten()
        
        return self.history
    
    def forecast_future(self, steps: int) -> np.ndarray:
        """Forecast future values."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Start with the last look_back values from training data
        last_data = self.train_data.values[-self.look_back:].reshape(-1, 1)
        scaled_last_data = self.scaler.transform(last_data)
        
        future_forecast = []
        
        for _ in range(steps):
            # Reshape for prediction
            X = scaled_last_data.reshape(1, self.look_back, 1)
            
            # Predict next value
            scaled_pred = self.model.predict(X, verbose=0)
            
            # Store prediction
            future_forecast.append(scaled_pred[0][0])
            
            # Update last_data for next prediction
            scaled_last_data = np.append(scaled_last_data[1:], scaled_pred, axis=0)
        
        # Invert scaling for final forecast
        future_forecast = np.array(future_forecast).reshape(-1, 1)
        future_forecast = self.scaler.inverse_transform(future_forecast)
        
        return future_forecast.flatten()
