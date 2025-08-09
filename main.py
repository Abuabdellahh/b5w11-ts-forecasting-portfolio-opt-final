"""
Main script for Time Series Forecasting and Portfolio Optimization.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from src.data.data_loader import DataLoader
from src.models.time_series_models import ARIMAForecaster, LSTMForecaster
from src.models.portfolio_optimizer import PortfolioOptimizer
from src.utils.visualization import (
    plot_price_series, 
    plot_returns_distribution,
    plot_rolling_statistics,
    plot_correlation_heatmap,
    plot_forecast_comparison,
    plot_portfolio_performance,
    plot_drawdown
)

# Ensure reports/figures directory exists
os.makedirs('reports/figures', exist_ok=True)

def main():
    """Main function to run the analysis."""
    print("Starting Time Series Forecasting and Portfolio Optimization Analysis...")
    
    # 1. Data Loading and Preprocessing
    print("\n1. Loading and preprocessing data...")
    
    # Define parameters
    TICKERS = ['TSLA', 'SPY', 'BND']
    START_DATE = '2015-07-01'
    END_DATE = '2025-07-31'
    
    # Initialize and fetch data
    data_loader = DataLoader(TICKERS, START_DATE, END_DATE)
    raw_data = data_loader.fetch_data()
    
    # Preprocess data
    processed_data = data_loader.preprocess_data()
    
    # Save processed data
    for ticker, df in processed_data.items():
        df.to_csv(f'data/{ticker}_processed.csv')
    
    # 2. Exploratory Data Analysis
    print("\n2. Performing Exploratory Data Analysis...")
    
    # Plot price series
    plot_price_series(
        {ticker: df for ticker, df in processed_data.items()},
        title='Adjusted Close Prices',
        filename='adjusted_close_prices.png'
    )
    
    # Get merged returns for correlation analysis
    returns_df = data_loader.get_merged_returns()
    
    # Plot returns distribution
    plot_returns_distribution(
        returns_df,
        title='Distribution of Daily Returns',
        filename='returns_distribution.png'
    )
    
    # Plot rolling statistics for TSLA
    if 'TSLA' in processed_data:
        plot_rolling_statistics(
            processed_data['TSLA']['Adj Close'],
            window=30,
            title='TSLA - Rolling Statistics',
            filename='tsla_rolling_stats.png'
        )
    
    # Plot correlation heatmap
    plot_correlation_heatmap(
        returns_df,
        title='Asset Returns Correlation',
        filename='returns_correlation.png'
    )
    
    # 3. Time Series Forecasting
    print("\n3. Time Series Forecasting for TSLA...")
    
    if 'TSLA' in processed_data:
        # Prepare data for forecasting
        tsla_data = processed_data['TSLA']['Adj Close']
        
        # Split into train and test sets (80-20 split)
        train_size = int(len(tsla_data) * 0.8)
        train_data, test_data = tsla_data[:train_size], tsla_data[train_size:]
        
        # ARIMA Model
        print("\nTraining ARIMA model...")
        arima_model = ARIMAForecaster(train_data, test_data)
        arima_model.train()
        arima_metrics = arima_model.evaluate()
        print(f"ARIMA Model Metrics: {arima_metrics}")
        
        # Plot ARIMA forecast
        arima_model.plot_forecast(
            title='ARIMA Forecast vs Actuals',
            filename='arima_forecast.png'
        )
        
        # LSTM Model
        print("\nTraining LSTM model...")
        lstm_model = LSTMForecaster(
            train_data, 
            test_data,
            look_back=30,
            epochs=50,
            batch_size=32
        )
        lstm_model.train()
        lstm_metrics = lstm_model.evaluate()
        print(f"LSTM Model Metrics: {lstm_metrics}")
        
        # Plot LSTM forecast
        lstm_model.plot_forecast(
            title='LSTM Forecast vs Actuals',
            filename='lstm_forecast.png'
        )
        
        # Select best model based on RMSE
        best_model = arima_model if arima_metrics['RMSE'] < lstm_metrics['RMSE'] else lstm_model
        best_model_name = 'ARIMA' if arima_metrics['RMSE'] < lstm_metrics['RMSE'] else 'LSTM'
        print(f"\nBest model based on RMSE: {best_model_name}")
        
        # Generate 6-month forecast
        print("\nGenerating 6-month forecast...")
        forecast_steps = 126  # ~6 months of trading days
        future_forecast = best_model.forecast_future(steps=forecast_steps)
        
        # Create date range for forecast
        last_date = tsla_data.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=forecast_steps,
            freq='B'  # Business days
        )
        
        # Create forecast series
        forecast_series = pd.Series(
            future_forecast,
            index=forecast_dates,
            name='Forecast'
        )
        
        # Plot forecast
        plt.figure(figsize=(14, 7))
        plt.plot(tsla_data.index[-100:], tsla_data[-100:], label='Historical Data')
        plt.plot(forecast_series.index, forecast_series, label='Forecast', color='red')
        plt.title(f'6-Month {best_model_name} Price Forecast', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price (USD)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('reports/figures/6_month_forecast.png')
        plt.close()
        
        # 4. Portfolio Optimization
        print("\n4. Portfolio Optimization...")
        
        # Prepare returns data for optimization
        returns_df = data_loader.get_merged_returns().dropna()
        
        # Add forecasted returns for TSLA
        forecast_returns = pd.Series(
            future_forecast / tsla_data[-1] - 1,  # Calculate daily returns
            index=forecast_dates,
            name='TSLA'
        )
        
        # Combine historical and forecasted returns
        combined_returns = returns_df.copy()
        
        # Initialize portfolio optimizer
        optimizer = PortfolioOptimizer(combined_returns)
        
        # Calculate expected returns and covariance matrix
        expected_returns = optimizer.calculate_expected_returns(method='mean_historical_return')
        cov_matrix = optimizer.calculate_covariance_matrix(method='ledoit_wolf')
        
        # Optimize for maximum Sharpe ratio
        print("\nOptimizing portfolio for maximum Sharpe ratio...")
        weights = optimizer.optimize_portfolio(objective='sharpe')
        
        # Get portfolio performance
        port_return, port_volatility, sharpe_ratio = optimizer.get_portfolio_performance()
        
        print("\nOptimal Portfolio Weights:")
        for asset, weight in weights.items():
            print(f"{asset}: {weight:.2%}")
        
        print(f"\nExpected Annual Return: {port_return:.2%}")
        print(f"Expected Annual Volatility: {port_volatility:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        
        # Plot efficient frontier
        optimizer.plot_efficient_frontier(
            filename='efficient_frontier.png'
        )
        
        # 5. Backtesting
        print("\n5. Backtesting Portfolio Strategy...")
        
        # Backtest the optimized portfolio
        backtest_results = optimizer.backtest_portfolio(
            initial_investment=10000,
            rebalance_freq='M',
            start_date='2024-01-01',
            end_date=END_DATE
        )
        
        # Get benchmark (60% SPY, 40% BND)
        if 'SPY' in processed_data and 'BND' in processed_data:
            # Calculate benchmark returns
            spy_returns = processed_data['SPY']['returns'].loc['2024-01-01':END_DATE]
            bnd_returns = processed_data['BND']['returns'].loc['2024-01-01':END_DATE]
            benchmark_returns = 0.6 * spy_returns + 0.4 * bnd_returns
            benchmark_values = (1 + benchmark_returns).cumprod() * 10000
            
            # Plot portfolio vs benchmark
            plot_portfolio_performance(
                backtest_results,
                benchmark=benchmark_values,
                title='Portfolio vs Benchmark Performance',
                filename='portfolio_vs_benchmark.png'
            )
            
            # Plot drawdown analysis
            portfolio_returns = backtest_results['Portfolio'].pct_change().dropna()
            plot_drawdown(
                portfolio_returns,
                title='Portfolio Drawdown Analysis',
                filename='portfolio_drawdown.png'
            )
        
        print("\nAnalysis complete! Check the 'reports/figures' directory for visualizations.")

if __name__ == "__main__":
    main()
