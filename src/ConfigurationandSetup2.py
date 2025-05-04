import requests
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import pickle

# -------------------------------
# Configuration and Setup
# -------------------------------

# Define the output directory
OUTPUT_DIR = 'D:/bitcoin/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created directory: {OUTPUT_DIR}")
else:
    print(f"Using existing directory: {OUTPUT_DIR}")

# Configure logging
LOG_FILE = os.path.join(OUTPUT_DIR, 'bitcoin_data.log')
logging.basicConfig(
    filename=LOG_FILE,
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
        df_market['hash_rate_ths'] = df_market['hash_rate_ths'].ffill()
        df_market['hash_rate_ths'] = df_market['hash_rate_ths'].bfill()
    
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
        df_market['transaction_volume'] = df_market['transaction_volume'].ffill()
        df_market['transaction_volume'] = df_market['transaction_volume'].bfill()
    
    logging.info("All data fetched and merged successfully.")
    print("All data fetched and merged successfully.")
    
    return df_market

def preprocess_data():
    """
    Perform data preprocessing including EDA, handling missing values, and feature scaling.
    """
    print("\n--- Data Preprocessing Started ---")
    
    # Define the path to the combined CSV file
    combined_csv = os.path.join(OUTPUT_DIR, 'combined_data.csv')
    
    # Load the data into a DataFrame
    try:
        df = pd.read_csv(combined_csv, parse_dates=['date'], index_col='date')
        print("\nFirst 5 rows of the dataset:")
        print(df.head())
    except FileNotFoundError:
        logging.error(f"Combined data file not found at {combined_csv}.")
        print(f"Combined data file not found at {combined_csv}. Please ensure the data fetching step was successful.")
        return
    except Exception as err:
        logging.error(f"An error occurred while loading the combined data: {err}")
        print(f"An error occurred while loading the combined data: {err}")
        return
    
    # Exploratory Data Analysis (EDA)
    print("\n--- Exploratory Data Analysis (EDA) ---")
    print("\nDataset Information:")
    print(df.info())
    
    print("\nStatistical Summary:")
    print(df.describe())
    
    duplicate_rows = df.duplicated().sum()
    print(f"\nNumber of duplicate rows: {duplicate_rows}")
    if duplicate_rows > 0:
        df.drop_duplicates(inplace=True)
        logging.info(f"Dropped {duplicate_rows} duplicate rows.")
        print(f"Dropped {duplicate_rows} duplicate rows.")
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Handle Missing Values
    print("\n--- Handling Missing Values ---")
    missing_values = df.isnull().sum()
    print("\nMissing Values After Initial Fill:")
    print(missing_values)
    
    if missing_values.any():
        # Option 1: Drop remaining missing values
        df.dropna(inplace=True)
        logging.info("Dropped remaining missing values.")
        print("\nDropped remaining missing values.")
        
        # Option 2: Impute missing values (Uncomment if preferred)
        # df.fillna(df.mean(), inplace=True)
        # logging.info("Imputed missing values with column means.")
        # print("\nImputed missing values with column means.")
    else:
        print("\nNo remaining missing values.")
    
    # Feature Scaling
    print("\n--- Feature Scaling ---")
    scaler = StandardScaler()
    features_to_scale = ['price', 'market_cap', 'total_volume', 'hash_rate_ths', 'transaction_volume']
    
    # Check if all features exist in the DataFrame
    existing_features = [feature for feature in features_to_scale if feature in df.columns]
    missing_features = [feature for feature in features_to_scale if feature not in df.columns]
    
    if missing_features:
        for feature in missing_features:
            logging.warning(f"Feature '{feature}' not found in the DataFrame. Skipping scaling for this feature.")
            print(f"Warning: Feature '{feature}' not found in the DataFrame. Skipping scaling for this feature.")
    
    if existing_features:
        df_scaled = df.copy()
        df_scaled[existing_features] = scaler.fit_transform(df_scaled[existing_features])
        print("\nFirst 5 rows of the scaled dataset:")
        print(df_scaled.head())
        
        # Save the scaler object for future use
        scaler_file = os.path.join(OUTPUT_DIR, 'scaler_standard.pkl')
        with open(scaler_file, 'wb') as f:
            pickle.dump(scaler, f)
        logging.info(f"Scaler object saved successfully to {scaler_file}.")
        print(f"\nScaler object saved successfully to {scaler_file}.")
    else:
        df_scaled = df.copy()
        print("\nNo features were scaled as none were found.")
    
    # Save the preprocessed data
    preprocessed_csv = os.path.join(OUTPUT_DIR, 'preprocessed_combined_data.csv')
    try:
        df_scaled.to_csv(preprocessed_csv)
        logging.info(f"Preprocessed data saved successfully to {preprocessed_csv}.")
        print(f"\nPreprocessed data saved successfully to {preprocessed_csv}.")
    except Exception as err:
        logging.error(f"An error occurred while saving the preprocessed data: {err}")
        print(f"An error occurred while saving the preprocessed data: {err}")
    
    print("\n--- Data Preprocessing Completed ---\n")
