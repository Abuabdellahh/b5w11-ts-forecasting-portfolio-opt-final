"""
Portfolio optimization using Modern Portfolio Theory.
"""
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.cla import CLA
import warnings
warnings.filterwarnings('ignore')

class PortfolioOptimizer:
    """Class for portfolio optimization using Modern Portfolio Theory."""
    
    def __init__(self, returns_df: pd.DataFrame):
        """
        Initialize the portfolio optimizer with historical returns.
        
        Args:
            returns_df: DataFrame with asset returns (columns: asset names, index: dates)
        """
        self.returns_df = returns_df.dropna()
        self.expected_returns = None
        self.cov_matrix = None
        self.ef = None
        self.optimal_weights = None
        
    def calculate_expected_returns(self, method: str = 'mean_historical_return', 
                                 **kwargs) -> pd.Series:
        """
        Calculate expected returns using the specified method.
        
        Args:
            method: Method to calculate expected returns ('mean_historical_return' or 'capm')
            **kwargs: Additional arguments for the expected returns calculation
            
        Returns:
            Series of expected returns for each asset
        """
        if method == 'mean_historical_return':
            self.expected_returns = expected_returns.mean_historical_return(
                self.returns_df,
                frequency=252,  # Trading days in a year
                **kwargs
            )
        elif method == 'capm':
            self.expected_returns = expected_returns.capm_return(
                self.returns_df,
                risk_free_rate=kwargs.get('risk_free_rate', 0.02),
                frequency=252,
                **{k: v for k, v in kwargs.items() if k != 'risk_free_rate'}
            )
        else:
            raise ValueError(f"Unsupported expected returns method: {method}")
            
        return self.expected_returns
    
    def calculate_covariance_matrix(self, method: str = 'ledoit_wolf', 
                                  **kwargs) -> pd.DataFrame:
        """
        Calculate the covariance matrix using the specified method.
        
        Args:
            method: Method to calculate covariance ('sample_cov', 'ledoit_wolf', etc.)
            **kwargs: Additional arguments for the covariance calculation
            
        Returns:
            DataFrame of the covariance matrix
        """
        if method == 'sample_cov':
            self.cov_matrix = risk_models.sample_cov(self.returns_df, **kwargs)
        elif method == 'ledoit_wolf':
            self.cov_matrix = risk_models.CovarianceShrinkage(self.returns_df, **kwargs).ledoit_wolf()
        elif method == 'oracle_approximating':
            self.cov_matrix = risk_models.CovarianceShrinkage(self.returns_df, **kwargs).oracle_approximating()
        else:
            raise ValueError(f"Unsupported covariance method: {method}")
            
        return self.cov_matrix
    
    def optimize_portfolio(self, objective: str = 'sharpe', 
                         target_return: Optional[float] = None,
                         target_risk: Optional[float] = None,
                         weight_bounds: Tuple[float, float] = (0, 1),
                         **kwargs) -> dict:
        """
        Optimize the portfolio weights.
        
        Args:
            objective: Optimization objective ('sharpe', 'min_volatility', 'efficient_risk', 'efficient_return')
            target_return: Target return for 'efficient_risk' objective
            target_risk: Target risk (volatility) for 'efficient_return' objective
            weight_bounds: Bounds for asset weights (min, max)
            **kwargs: Additional arguments for the optimization
            
        Returns:
            Dictionary of optimized weights
        """
        if self.expected_returns is None:
            self.calculate_expected_returns()
        if self.cov_matrix is None:
            self.calculate_covariance_matrix()
        
        # Create the efficient frontier
        self.ef = EfficientFrontier(
            self.expected_returns,
            self.cov_matrix,
            weight_bounds=weight_bounds,
            **kwargs
        )
        
        # Add constraints if provided
        if 'constraints' in kwargs:
            for constr in kwargs['constraints']:
                self.ef.add_constraint(constr)
        
        # Optimize for the specified objective
        if objective == 'sharpe':
            self.ef.max_sharpe()
        elif objective == 'min_volatility':
            self.ef.min_volatility()
        elif objective == 'efficient_risk':
            if target_return is None:
                raise ValueError("target_return must be provided for 'efficient_risk' objective")
            self.ef.efficient_risk(target_risk, target_return)
        elif objective == 'efficient_return':
            if target_risk is None:
                raise ValueError("target_risk must be provided for 'efficient_return' objective")
            self.ef.efficient_return(target_return, target_risk)
        else:
            raise ValueError(f"Unsupported optimization objective: {objective}")
        
        # Get the optimized weights
        self.optimal_weights = self.ef.clean_weights()
        return self.optimal_weights
    
    def get_portfolio_performance(self, weights: Optional[dict] = None) -> Tuple[float, float, float]:
        """
        Calculate portfolio performance metrics.
        
        Args:
            weights: Dictionary of asset weights. If None, uses optimal_weights
            
        Returns:
            Tuple of (expected return, volatility, Sharpe ratio)
        """
        if weights is None:
            if self.optimal_weights is None:
                raise ValueError("No weights provided and no optimal weights available.")
            weights = self.optimal_weights
        
        # Convert weights to numpy array in the same order as returns_df columns
        weights_array = np.array([weights[asset] for asset in self.returns_df.columns])
        
        # Calculate portfolio return
        port_return = np.sum(self.expected_returns * weights_array)
        
        # Calculate portfolio volatility
        port_volatility = np.sqrt(np.dot(weights_array.T, np.dot(self.cov_matrix, weights_array)))
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0 for simplicity)
        sharpe_ratio = port_return / port_volatility if port_volatility > 0 else 0
        
        return port_return, port_volatility, sharpe_ratio
    
    def plot_efficient_frontier(self, show_assets: bool = True, show_cml: bool = True,
                              filename: str = 'efficient_frontier.png'):
        """
        Plot the efficient frontier and capital market line.
        
        Args:
            show_assets: Whether to show individual assets on the plot
            show_cml: Whether to show the Capital Market Line
            filename: Name of the file to save the plot
        """
        if self.ef is None:
            raise ValueError("Efficient frontier not calculated. Run optimize_portfolio first.")
        
        # Use CLA for efficient frontier plotting
        cla = CLA(self.expected_returns, self.cov_matrix)
        cla.max_sharpe()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot efficient frontier
        ef = cla.efficient_frontier
        ax.scatter(ef['variance'] ** 0.5, ef['returns'], c='blue', marker='.', label='Efficient Frontier')
        
        # Plot individual assets
        if show_assets:
            std = np.sqrt(np.diag(self.cov_matrix))
            ax.scatter(std, self.expected_returns, marker='*', s=200, c='r', label='Assets')
            
            # Annotate asset names
            for i, txt in enumerate(self.returns_df.columns):
                ax.annotate(txt, (std[i], self.expected_returns[i]))
        
        # Plot Capital Market Line if available
        if show_cml and hasattr(cla, 'max_sharpe_ratio'):
            cml_x = [0, cla.max_sharpe_ratio['variance'] ** 0.5 * 1.5]
            cml_y = [0.02, 0.02 + cla.max_sharpe_ratio['sharpe'] * cml_x[1]]
            ax.plot(cml_x, cml_y, 'g--', linewidth=2, label='Capital Market Line')
            
            # Plot max Sharpe ratio portfolio
            ax.scatter(cla.max_sharpe_ratio['variance'] ** 0.5, 
                      cla.max_sharpe_ratio['returns'], 
                      marker='*', s=300, c='g', label='Max Sharpe Ratio')
        
        # Plot minimum volatility portfolio
        min_vol_idx = cla.weights.std(axis=1).idxmin()
        min_vol_ret = cla.expected_return[min_vol_idx]
        min_vol_std = cla.volatility[min_vol_idx]
        ax.scatter(min_vol_std, min_vol_ret, marker='*', s=300, c='y', label='Min Volatility')
        
        # Add labels and legend
        ax.set_title('Portfolio Optimization with Efficient Frontier')
        ax.set_xlabel('Volatility (Standard Deviation)')
        ax.set_ylabel('Expected Return')
        ax.legend()
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(f'reports/figures/{filename}')
        plt.close()
        
        return fig, ax
    
    def backtest_portfolio(self, initial_investment: float = 10000,
                         rebalance_freq: str = 'M',
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Backtest the portfolio strategy.
        
        Args:
            initial_investment: Initial investment amount
            rebalance_freq: Rebalancing frequency ('D', 'W', 'M', 'Q', 'Y')
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            
        Returns:
            DataFrame with portfolio values over time
        """
        # Filter data for backtest period
        if start_date is not None:
            returns = self.returns_df.loc[start_date:]
        else:
            returns = self.returns_df
            
        if end_date is not None:
            returns = returns.loc[:end_date]
        
        # Initialize portfolio value
        portfolio_value = pd.DataFrame(index=returns.index)
        portfolio_value['Portfolio'] = initial_investment
        
        # Get rebalancing dates
        rebalance_dates = pd.date_range(
            start=returns.index[0],
            end=returns.index[-1],
            freq=rebalance_freq
        )
        
        # Initialize weights (equal weight for first period)
        current_weights = {asset: 1/len(returns.columns) for asset in returns.columns}
        
        for i in range(1, len(portfolio_value)):
            # Check if it's time to rebalance
            if portfolio_value.index[i] in rebalance_dates:
                # Calculate new optimal weights based on data up to this point
                historical_returns = returns.loc[:portfolio_value.index[i-1]]
                
                # Re-optimize portfolio
                self.returns_df = historical_returns
                self.calculate_expected_returns()
                self.calculate_covariance_matrix()
                current_weights = self.optimize_portfolio(objective='sharpe')
            
            # Calculate daily returns based on current weights
            daily_return = sum(returns.iloc[i][asset] * current_weights[asset] 
                             for asset in returns.columns)
            
            # Update portfolio value
            portfolio_value.iloc[i] = portfolio_value.iloc[i-1] * (1 + daily_return)
        
        return portfolio_value
