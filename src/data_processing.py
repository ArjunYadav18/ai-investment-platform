# src/data_processing.py

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

TICKERS = ['AAPL', 'MSFT', 'JPM', 'PG', 'JNJ', 'XOM']
YEARS_OF_DATA = 5

def fetch_stock_data(tickers, years):
    """
    Fetches historical adjusted closing prices for a list of stock tickers.
    """
    print(f"Fetching {years} years of data for: {', '.join(tickers)}...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)

    try:
        data_df = yf.download(tickers, start=start_date, end=end_date)
        adj_close_df = data_df['Close']
        
        if adj_close_df.empty:
            print("Error: No data fetched.")
            return None
            
        print("Data fetched successfully.")
        return adj_close_df.dropna()

    except Exception as e:
        print(f"An error occurred during data fetching: {e}")
        return None

def calculate_portfolio_inputs(price_data):
    """
    Calculates annualized mean returns and the covariance matrix from price data.
    """
    if price_data is None:
        return None, None
        
    print("Calculating portfolio inputs...")
    monthly_returns = price_data.resample('M').ffill().pct_change().dropna()
    annualized_mean_returns = monthly_returns.mean() * 12
    annualized_cov_matrix = monthly_returns.cov() * 12
    
    print("Calculations complete.")
    return annualized_mean_returns, annualized_cov_matrix

def save_data_to_csv(mean_returns, cov_matrix, directory="data"):
    """
    Saves the calculated mean returns and covariance matrix to CSV files.
    """
    if mean_returns is None or cov_matrix is None:
        print("Error: No data to save.")
        return
        
    # Correctly join paths to handle the script being in 'src'
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, directory)
    
    print(f"Saving data to '{output_dir}' directory...")
    os.makedirs(output_dir, exist_ok=True)
    
    mean_returns.to_csv(os.path.join(output_dir, "annualized_mean_returns.csv"))
    cov_matrix.to_csv(os.path.join(output_dir, "annualized_cov_matrix.csv"))
    
    print("Data saved successfully.")

if __name__ == "__main__":
    price_df = fetch_stock_data(tickers=TICKERS, years=YEARS_OF_DATA)
    mean_returns, cov_matrix = calculate_portfolio_inputs(price_df)
    save_data_to_csv(mean_returns, cov_matrix)

    if mean_returns is not None:
        print("\n--- Annualized Mean Returns ---")
        print(mean_returns)
        print("\n--- Annualized Covariance Matrix ---")
        print(cov_matrix)