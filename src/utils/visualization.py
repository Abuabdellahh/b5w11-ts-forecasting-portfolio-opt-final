""
Visualization utilities for financial analysis.
"""
from typing import Optional, List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

# Set the style for plots
plt.style.use('seaborn')
sns.set_palette("viridis")

def plot_price_series(data: Dict[str, pd.DataFrame], 
                    column: str = 'Adj Close', 
                    title: str = 'Asset Prices Over Time',
                    filename: Optional[str] = None) -> None:
    """
    Plot price series for multiple assets.
    
    Args:
        data: Dictionary with asset names as keys and DataFrames as values
        column: Column to plot (default: 'Adj Close')
        title: Plot title
        filename: If provided, save the plot to this file
    """
    plt.figure(figsize=(14, 7))
    
    for asset, df in data.items():
        if column in df.columns:
            plt.plot(df.index, df[column], label=asset)
    
    plt.title(title, fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Format x-axis for dates
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(f'reports/figures/{filename}')
        plt.close()
    else:
        plt.show()

def plot_returns_distribution(returns: pd.DataFrame, 
                            title: str = 'Distribution of Returns',
                            filename: Optional[str] = None) -> None:
    """
    Plot distribution of returns for multiple assets.
    
    Args:
        returns: DataFrame with asset returns (columns: asset names, index: dates)
        title: Plot title
        filename: If provided, save the plot to this file
    """
    plt.figure(figsize=(14, 7))
    
    for column in returns.columns:
        sns.histplot(returns[column], kde=True, alpha=0.5, label=column)
    
    plt.title(title, fontsize=16)
    plt.xlabel('Daily Returns', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(f'reports/figures/{filename}')
        plt.close()
    else:
        plt.show()

def plot_rolling_statistics(series: pd.Series, 
                          window: int = 21, 
                          title: str = 'Rolling Statistics',
                          filename: Optional[str] = None) -> None:
    """
    Plot rolling mean and standard deviation of a time series.
    
    Args:
        series: Time series data
        window: Window size for rolling calculations
        title: Plot title
        filename: If provided, save the plot to this file
    """
    plt.figure(figsize=(14, 10))
    
    # Calculate rolling statistics
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    
    # Plot rolling statistics
    plt.subplot(2, 1, 1)
    plt.plot(series.index, series, label='Original')
    plt.plot(rolling_mean.index, rolling_mean, 'r', label=f'Rolling Mean ({window}d)')
    plt.title(f'{title} - Rolling Mean', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(rolling_std.index, rolling_std, 'g', label=f'Rolling Std ({window}d)')
    plt.title(f'{title} - Rolling Standard Deviation', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(f'reports/figures/{filename}')
        plt.close()
    else:
        plt.show()

def plot_correlation_heatmap(returns: pd.DataFrame, 
                           title: str = 'Correlation Heatmap',
                           filename: Optional[str] = None) -> None:
    """
    Plot correlation heatmap for asset returns.
    
    Args:
        returns: DataFrame with asset returns (columns: asset names, index: dates)
        title: Plot title
        filename: If provided, save the plot to this file
    """
    plt.figure(figsize=(10, 8))
    
    # Calculate correlation matrix
    corr = returns.corr()
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, 
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    
    plt.title(title, fontsize=16)
    
    if filename:
        plt.savefig(f'reports/figures/{filename}')
        plt.close()
    else:
        plt.show()

def plot_forecast_comparison(train: pd.Series, 
                           test: pd.Series, 
                           forecast: np.ndarray,
                           title: str = 'Forecast vs Actuals',
                           model_name: str = 'Model',
                           filename: Optional[str] = None) -> None:
    """
    Plot forecasted values against actual values.
    
    Args:
        train: Training data (pandas Series)
        test: Test data (pandas Series)
        forecast: Forecasted values (numpy array)
        title: Plot title
        model_name: Name of the forecasting model (for legend)
        filename: If provided, save the plot to this file
    """
    plt.figure(figsize=(14, 7))
    
    # Plot training data
    plt.plot(train.index, train, label='Training Data', color='blue', alpha=0.6)
    
    # Plot test data
    plt.plot(test.index, test, label='Actual Test Data', color='green', alpha=0.8)
    
    # Plot forecast
    plt.plot(test.index, forecast, label=f'{model_name} Forecast', 
             color='red', linestyle='--', alpha=0.8)
    
    # Add confidence interval (if available)
    if isinstance(forecast, dict) and 'mean' in forecast and 'lower' in forecast and 'upper' in forecast:
        plt.fill_between(test.index, 
                        forecast['lower'], 
                        forecast['upper'], 
                        color='red', alpha=0.1,
                        label='95% Confidence Interval')
    
    plt.title(title, fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Format x-axis for dates
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(f'reports/figures/{filename}')
        plt.close()
    else:
        plt.show()

def plot_portfolio_performance(portfolio_values: pd.DataFrame, 
                             benchmark: Optional[pd.Series] = None,
                             title: str = 'Portfolio Performance',
                             filename: Optional[str] = None) -> None:
    """
    Plot portfolio performance over time.
    
    Args:
        portfolio_values: DataFrame with portfolio values over time
        benchmark: Optional benchmark series for comparison
        title: Plot title
        filename: If provided, save the plot to this file
    """
    plt.figure(figsize=(14, 7))
    
    # Plot portfolio values
    for column in portfolio_values.columns:
        plt.plot(portfolio_values.index, portfolio_values[column], label=column)
    
    # Plot benchmark if provided
    if benchmark is not None:
        # Normalize benchmark to start at the same point as the portfolio
        benchmark_normalized = benchmark / benchmark.iloc[0] * portfolio_values.iloc[0, 0]
        plt.plot(benchmark_normalized.index, benchmark_normalized, 
                label='Benchmark', linestyle='--', alpha=0.8)
    
    plt.title(title, fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value (USD)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Format x-axis for dates
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(f'reports/figures/{filename}')
        plt.close()
    else:
        plt.show()

def plot_drawdown(returns: pd.Series, 
                 window: int = 21,
                 title: str = 'Drawdown Analysis',
                 filename: Optional[str] = None) -> None:
    """
    Plot drawdown analysis for a return series.
    
    Args:
        returns: Series of returns
        window: Window size for rolling maximum
        title: Plot title
        filename: If provided, save the plot to this file
    """
    # Calculate cumulative returns
    cum_returns = (1 + returns).cumprod()
    
    # Calculate running maximum
    running_max = cum_returns.cummax()
    
    # Calculate drawdown
    drawdown = (cum_returns - running_max) / running_max
    
    # Calculate maximum drawdown
    max_drawdown = drawdown.min()
    max_drawdown_date = drawdown.idxmin()
    
    # Create figure
    plt.figure(figsize=(14, 7))
    
    # Plot drawdown
    plt.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
    plt.plot(drawdown.index, drawdown, color='red', alpha=0.8, label='Drawdown')
    
    # Add max drawdown annotation
    plt.axhline(y=max_drawdown, color='black', linestyle='--', alpha=0.5)
    plt.annotate(f'Max Drawdown: {max_drawdown:.2%}', 
                 xy=(max_drawdown_date, max_drawdown),
                 xytext=(10, 10), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
    
    plt.title(title, fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Drawdown', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Format x-axis for dates
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(f'reports/figures/{filename}')
        plt.close()
    else:
        plt.show()
