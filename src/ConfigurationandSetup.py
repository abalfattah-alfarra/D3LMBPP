import requests
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta

# -------------------------------
# Configuration and Setup
# -------------------------------

# Define the output directory
output_dir = 'D:/bitcoin/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")
else:
    print(f"Using existing directory: {output_dir}")

# Configure logging
log_file = os.path.join(output_dir, 'bitcoin_data.log')
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# -------------------------------
# Function Definitions
# -------------------------------

def fetch_market_data(coin_id='bitcoin', vs_currency='usd', days=365, interval='daily'):
    """
    Fetch historical market data from CoinGecko API.
    """
    print("Fetching market data from CoinGecko...")
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart'
    params = {
        'vs_currency': vs_currency,
        'days': days,
        'interval': interval
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Process prices
        df_prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        df_prices['date'] = pd.to_datetime(df_prices['timestamp'], unit='ms')
        df_prices.set_index('date', inplace=True)
        df_prices.drop('timestamp', axis=1, inplace=True)
        
        # Process market caps
        df_market_cap = pd.DataFrame(data['market_caps'], columns=['timestamp', 'market_cap'])
        df_market_cap['date'] = pd.to_datetime(df_market_cap['timestamp'], unit='ms')
        df_market_cap.set_index('date', inplace=True)
        df_market_cap.drop('timestamp', axis=1, inplace=True)
        
        # Process total volumes
        df_total_volume = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'total_volume'])
        df_total_volume['date'] = pd.to_datetime(df_total_volume['timestamp'], unit='ms')
        df_total_volume.set_index('date', inplace=True)
        df_total_volume.drop('timestamp', axis=1, inplace=True)
        
        # Merge all market data
        df_market = df_prices.join(df_market_cap).join(df_total_volume)
        df_market.ffill(inplace=True)
        df_market.bfill(inplace=True)
        
        logging.info("Successfully fetched market data from CoinGecko.")
        print("Market data fetched successfully.")
        return df_market
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred while fetching market data: {http_err}")
        print(f"HTTP error occurred while fetching market data: {http_err}")
    except Exception as err:
        logging.error(f"An error occurred while fetching market data: {err}")
        print(f"An error occurred while fetching market data: {err}")
    return pd.DataFrame()

def fetch_onchain_data_hashrate():
    """
    Fetch hash rate data from Blockchain.com's public API.
    """
    print("Fetching hash rate data from Blockchain.com...")
    url = 'https://api.blockchain.info/charts/hash-rate'
    params = {
        'timespan': '365days',
        'format': 'json',
        'cors': 'true'
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Extract and process data
        values = data.get('values', [])
        if not values:
            logging.error("No hash rate data found in the response.")
            print("No hash rate data found in the response.")
            return pd.DataFrame()
        
        # Create DataFrame
        df_hashrate = pd.DataFrame(values)
        df_hashrate['date'] = pd.to_datetime(df_hashrate['x'], unit='s')
        df_hashrate.set_index('date', inplace=True)
        df_hashrate.rename(columns={'y': 'hash_rate_ehs'}, inplace=True)  # EH/s
        
        # Convert EH/s to TH/s (1 EH/s = 1e6 TH/s)
        df_hashrate['hash_rate_ths'] = df_hashrate['hash_rate_ehs'] * 1e6
        df_hashrate.drop(['hash_rate_ehs', 'x'], axis=1, inplace=True)
        
        logging.info("Successfully fetched hash rate data from Blockchain.com.")
        print("Hash rate data fetched successfully.")
        return df_hashrate
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred while fetching hash rate data: {http_err}")
        print(f"HTTP error occurred while fetching hash rate data: {http_err}")
    except Exception as err:
        logging.error(f"An error occurred while fetching hash rate data: {err}")
        print(f"An error occurred while fetching hash rate data: {err}")
    return pd.DataFrame()

def fetch_onchain_data_transactions():
    """
    Fetch transaction volume data from Blockchain.com's public API.
    """
    print("Fetching transaction volume data from Blockchain.com...")
    url = 'https://api.blockchain.info/charts/n-transactions'
    params = {
        'timespan': '365days',
        'format': 'json',
        'cors': 'true'
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Extract and process data
        values = data.get('values', [])
        if not values:
            logging.error("No transaction volume data found in the response.")
            print("No transaction volume data found in the response.")
            return pd.DataFrame()
        
        # Create DataFrame
        df_transactions = pd.DataFrame(values)
        df_transactions['date'] = pd.to_datetime(df_transactions['x'], unit='s')
        df_transactions.set_index('date', inplace=True)
        df_transactions.rename(columns={'y': 'transaction_volume'}, inplace=True)
        
        # Drop 'x' column as it's no longer needed
        df_transactions.drop('x', axis=1, inplace=True)
        
        logging.info("Successfully fetched transaction volume data from Blockchain.com.")
        print("Transaction volume data fetched successfully.")
        return df_transactions
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred while fetching transaction volume data: {http_err}")
        print(f"HTTP error occurred while fetching transaction volume data: {http_err}")
    except Exception as err:
        logging.error(f"An error occurred while fetching transaction volume data: {err}")
        print(f"An error occurred while fetching transaction volume data: {err}")
    return pd.DataFrame()

def fetch_all_data(coin_id='bitcoin'):
    """
    Fetch all relevant data and merge into a single DataFrame.
    """
    # Fetch market data
    df_market = fetch_market_data(coin_id=coin_id)
    if df_market.empty:
        logging.error("Market data is empty. Exiting data fetching process.")
        print("Market data is empty. Please check the logs for more details.")
        return pd.DataFrame()
    
    # Fetch hash rate data
    df_hashrate = fetch_onchain_data_hashrate()
    if df_hashrate.empty:
        logging.warning("Hash rate data is empty. Proceeding without it.")
        print("Hash rate data is empty. Proceeding without it.")
    else:
        # Merge hash rate data with market data
        print("Merging hash rate data with market data...")
        df_market = df_market.join(df_hashrate, how='left')
        # Handle any remaining missing values
        df_market['hash_rate_ths'] = df_market['hash_rate_ths'].fillna(method='ffill')
        df_market['hash_rate_ths'] = df_market['hash_rate_ths'].fillna(method='bfill')
    
    # Fetch transaction volume data
    df_transactions = fetch_onchain_data_transactions()
    if df_transactions.empty:
        logging.warning("Transaction volume data is empty. Proceeding without it.")
        print("Transaction volume data is empty. Proceeding without it.")
    else:
        # Merge transaction volume data with market data
        print("Merging transaction volume data with market data...")
        df_market = df_market.join(df_transactions, how='left')
        # Handle any remaining missing values
        df_market['transaction_volume'] = df_market['transaction_volume'].fillna(method='ffill')
        df_market['transaction_volume'] = df_market['transaction_volume'].fillna(method='bfill')
    
    logging.info("All data fetched and merged successfully.")
    print("All data fetched and merged successfully.")
    
    return df_market

# -------------------------------
# Main Execution
# -------------------------------

if __name__ == "__main__":
    # Fetch and merge all data
    combined_data = fetch_all_data(coin_id='bitcoin')
    
    if not combined_data.empty:
        # Save to CSV
        csv_file = os.path.join(output_dir, 'combined_data.csv')
        combined_data.to_csv(csv_file)
        logging.info(f"Combined data saved successfully to {csv_file}.")
        print(f"Combined data saved successfully to {csv_file}.")
    else:
        logging.error("Combined data is empty. No CSV file will be saved.")
        print("Combined data is empty. Please check the logs for more details.")
