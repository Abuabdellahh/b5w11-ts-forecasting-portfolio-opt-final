"""
Data loading and preprocessing module for financial time series data.
"""
import yfinance as yf
import pandas as pd
from typing import Dict, List, Tuple
import numpy as np

class DataLoader:
    """Class for loading and preprocessing financial data."""
    
    def __init__(self, tickers: List[str], start_date: str, end_date: str):
        """
        Initialize the DataLoader with tickers and date range.
        
        Args:
            tickers: List of ticker symbols (e.g., ['TSLA', 'SPY', 'BND'])
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = {}
    
    def fetch_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for all tickers.
        
        Returns:
            Dictionary with tickers as keys and DataFrames as values
        """
        print(f"Fetching data from {self.start_date} to {self.end_date}...")
        for ticker in self.tickers:
            try:
                df = yf.download(ticker, start=self.start_date, end=self.end_date)
                if not df.empty:
                    self.data[ticker] = df
                    print(f"Successfully fetched {ticker} data")
                else:
                    print(f"No data found for {ticker}")
            except Exception as e:
                print(f"Error fetching {ticker}: {str(e)}")
        
        return self.data
    
    def preprocess_data(self) -> Dict[str, pd.DataFrame]:
        """
        Preprocess the fetched data.
        
        Returns:
            Dictionary with preprocessed DataFrames
        """
        if not self.data:
            print("No data to preprocess. Fetch data first.")
            return {}
        
        processed_data = {}
        
        for ticker, df in self.data.items():
            # Create a copy to avoid SettingWithCopyWarning
            df_processed = df.copy()
            
            # Handle missing values
            df_processed = self._handle_missing_values(df_processed)
            
            # Add returns and other features
            df_processed = self._add_features(df_processed)
            
            processed_data[ticker] = df_processed
        
        return processed_data
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the DataFrame."""
        # Forward fill for missing values (carry forward last known value)
        df = df.ffill()
        
        # If there are still missing values at the beginning, backward fill
        df = df.bfill()
        
        return df
    
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators and features to the DataFrame."""
        # Calculate daily returns
        df['returns'] = df['Adj Close'].pct_change()
        
        # Calculate moving averages
        df['MA_20'] = df['Adj Close'].rolling(window=20).mean()
        df['MA_50'] = df['Adj Close'].rolling(window=50).mean()
        
        # Calculate volatility (20-day rolling standard deviation of returns)
        df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)  # Annualized
        
        # Calculate daily high-low percentage
        df['HL_PCT'] = (df['High'] - df['Low']) / df['Adj Close'] * 100.0
        
        # Calculate percentage change
        df['PCT_change'] = df['Adj Close'].pct_change() * 100.0
        
        # Drop any remaining NaN values
        df = df.dropna()
        
        return df
    
    def get_merged_returns(self) -> pd.DataFrame:
        """
        Merge returns of all tickers into a single DataFrame.
        
        Returns:
            DataFrame with returns for all tickers
        """
        returns_data = {}
        for ticker, df in self.data.items():
            if 'returns' in df.columns:
                returns_data[ticker] = df['returns']
        
        if not returns_data:
            return pd.DataFrame()
            
        return pd.DataFrame(returns_data)
