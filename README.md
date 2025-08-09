# Time Series Forecasting for Portfolio Management Optimization

## Project Overview
This project implements a comprehensive pipeline for financial time series forecasting and portfolio optimization. It includes data loading, exploratory data analysis, time series forecasting (ARIMA and LSTM models), portfolio optimization using Modern Portfolio Theory, and backtesting of investment strategies.

## Features

- **Data Loading & Preprocessing**
  - Fetch historical financial data using YFinance
  - Handle missing values and outliers
  - Calculate technical indicators and returns

- **Exploratory Data Analysis**
  - Visualize price series and returns
  - Analyze volatility and correlation
  - Detect trends and seasonality

- **Time Series Forecasting**
  - Implement ARIMA/SARIMA models
  - Implement LSTM neural networks
  - Compare model performance
  - Generate future price forecasts

- **Portfolio Optimization**
  - Calculate expected returns and covariance
  - Generate efficient frontier
  - Identify optimal portfolios (max Sharpe ratio, min volatility)

- **Backtesting**
  - Simulate portfolio performance
  - Compare against benchmarks
  - Analyze risk metrics

## Project Structure
```
.
├── data/                    # Raw and processed data
├── notebooks/               # Jupyter notebooks for analysis
├── reports/                 # Generated reports and visualizations
│   └── figures/             # Saved plots and charts
├── src/                     # Source code
│   ├── data/                # Data loading and preprocessing
│   ├── models/              # Forecasting and optimization models
│   └── utils/               # Utility functions
├── main.py                  # Main script to run the analysis
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/b5w11-ts-forecasting-portfolio-opt-final.git
cd b5w11-ts-forecasting-portfolio-opt-final
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the main analysis script:
```bash
python main.py
```

This will:
1. Download and preprocess the data
2. Perform exploratory data analysis
3. Train and compare forecasting models
4. Optimize the portfolio
5. Generate visualizations in the `reports/figures` directory

## Customization

### Data
- Modify the tickers and date range in `main.py`:
```python
TICKERS = ['TSLA', 'SPY', 'BND']
START_DATE = '2015-07-01'
END_DATE = '2025-07-31'
```

### Models
Adjust model parameters in the respective model classes:
- ARIMA/SARIMA: `src/models/time_series_models.py`
- LSTM: `src/models/time_series_models.py`
- Portfolio Optimization: `src/models/portfolio_optimizer.py`

## Results

The analysis will generate several visualizations in the `reports/figures` directory, including:
- Price series and returns distributions
- Model forecasts and performance metrics
- Efficient frontier and optimal portfolios
- Backtesting results and performance comparison

## Dependencies

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- yfinance
- statsmodels
- pmdarima
- tensorflow
- scikit-learn
- PyPortfolioOpt

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Data provided by [Yahoo Finance](https://finance.yahoo.com/)
- Built with [PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt)
- Inspired by Modern Portfolio Theory (Harry Markowitz, 1952)