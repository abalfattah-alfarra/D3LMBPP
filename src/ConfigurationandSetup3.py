import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import pickle
import logging
from ta import trend, momentum, volatility, volume

# -------------------------------
# Configuration and Setup
# -------------------------------

# Define the output directory
OUTPUT_DIR = 'D:/bitcoin/'
FEATURES_DIR = os.path.join(OUTPUT_DIR, 'features')
if not os.path.exists(FEATURES_DIR):
    os.makedirs(FEATURES_DIR)
    print(f"Created features directory: {FEATURES_DIR}")
else:
    print(f"Using existing features directory: {FEATURES_DIR}")

# Configure logging
LOG_FILE = os.path.join(OUTPUT_DIR, 'bitcoin_feature_engineering.log')
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# -------------------------------
# Function Definitions
# -------------------------------

def add_technical_indicators(df):
    """
    Add technical indicators to the DataFrame.
    """
    print("\n--- Adding Technical Indicators ---")
    
    # Ensure the DataFrame is sorted by date
    df = df.sort_index()
    
    # Moving Averages
    df['MA_10'] = trend.SMAIndicator(close=df['price'], window=10).sma_indicator()
    df['MA_50'] = trend.SMAIndicator(close=df['price'], window=50).sma_indicator()
    df['EMA_10'] = trend.EMAIndicator(close=df['price'], window=10).ema_indicator()
    df['EMA_50'] = trend.EMAIndicator(close=df['price'], window=50).ema_indicator()
    
    # Relative Strength Index (RSI)
    df['RSI_14'] = momentum.RSIIndicator(close=df['price'], window=14).rsi()
    
    # Bollinger Bands
    bollinger = volatility.BollingerBands(close=df['price'], window=20, window_dev=2)
    df['Bollinger_High'] = bollinger.bollinger_hband()
    df['Bollinger_Low'] = bollinger.bollinger_lband()
    df['Bollinger_Middle'] = bollinger.bollinger_mavg()
    
    # Moving Average Convergence Divergence (MACD)
    macd = trend.MACD(close=df['price'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()
    
    # Exponential Moving Average (EMA) Difference
    df['EMA_Diff'] = df['EMA_10'] - df['EMA_50']
    
    logging.info("Technical indicators added successfully.")
    print("Technical indicators added successfully.")
    return df

def add_time_features(df):
    """
    Add time-based features to the DataFrame.
    """
    print("\n--- Adding Time-Based Features ---")
    
    df['Day'] = df.index.day
    df['Month'] = df.index.month
    df['Weekday'] = df.index.weekday  # Monday=0, Sunday=6
    df['Quarter'] = df.index.quarter
    
    # Cyclical Encoding for Weekday
    df['Weekday_Sin'] = np.sin(2 * np.pi * df['Weekday']/6)
    df['Weekday_Cos'] = np.cos(2 * np.pi * df['Weekday']/6)
    
    # Cyclical Encoding for Month
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month']/12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month']/12)
    
    logging.info("Time-based features added successfully.")
    print("Time-based features added successfully.")
    return df

def add_lagged_features(df, lags=[1,2,3,5,7]):
    """
    Add lagged features to the DataFrame.
    """
    print("\n--- Adding Lagged Features ---")
    
    for lag in lags:
        df[f'Price_Lag_{lag}'] = df['price'].shift(lag)
        df[f'MarketCap_Lag_{lag}'] = df['market_cap'].shift(lag)
        df[f'TotalVolume_Lag_{lag}'] = df['total_volume'].shift(lag)
        df[f'HashRate_Lag_{lag}'] = df['hash_rate_ths'].shift(lag)
        df[f'TransactionVolume_Lag_{lag}'] = df['transaction_volume'].shift(lag)
    
    logging.info("Lagged features added successfully.")
    print("Lagged features added successfully.")
    return df

def handle_missing_values(df):
    """
    Handle missing values resulting from feature engineering.
    """
    print("\n--- Handling Missing Values After Feature Engineering ---")
    
    missing_before = df.isnull().sum().sum()
    print(f"Total missing values before handling: {missing_before}")
    
    # Option 1: Drop rows with any missing values
    df.dropna(inplace=True)
    
    # Option 2: Alternatively, you can choose to fill missing values
    # df.fillna(method='ffill', inplace=True)
    # df.fillna(method='bfill', inplace=True)
    
    missing_after = df.isnull().sum().sum()
    print(f"Total missing values after handling: {missing_after}")
    
    logging.info("Missing values after feature engineering handled successfully.")
    print("Missing values after feature engineering handled successfully.")
    return df

def scale_features(df, features_to_scale):
    """
    Scale selected features using StandardScaler.
    """
    print("\n--- Scaling Features ---")
    scaler = StandardScaler()
    
    df_scaled = df.copy()
    df_scaled[features_to_scale] = scaler.fit_transform(df_scaled[features_to_scale])
    
    # Save the scaler object for future use
    scaler_file = os.path.join(FEATURES_DIR, 'feature_scaler.pkl')
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    logging.info(f"Feature scaler saved successfully to {scaler_file}.")
    print(f"Feature scaler saved successfully to {scaler_file}.")
    
    return df_scaled

def feature_engineering():
    """
    Perform feature engineering on the preprocessed data.
    """
    print("\n=== Step 5.1: Feature Engineering ===")
    
    # Define the path to the preprocessed CSV file
    preprocessed_csv = os.path.join(OUTPUT_DIR, 'preprocessed_combined_data.csv')
    
    # Load the preprocessed data
    try:
        df = pd.read_csv(preprocessed_csv, parse_dates=['date'], index_col='date')
        print("\nFirst 5 rows of the preprocessed dataset:")
        print(df.head())
    except FileNotFoundError:
        logging.error(f"Preprocessed data file not found at {preprocessed_csv}.")
        print(f"Preprocessed data file not found at {preprocessed_csv}. Please ensure the preprocessing step was successful.")
        return
    except Exception as err:
        logging.error(f"An error occurred while loading the preprocessed data: {err}")
        print(f"An error occurred while loading the preprocessed data: {err}")
        return
    
    # Add Technical Indicators
    df = add_technical_indicators(df)
    
    # Add Time-Based Features
    df = add_time_features(df)
    
    # Add Lagged Features
    df = add_lagged_features(df, lags=[1,2,3,5,7])
    
    # Handle Missing Values
    df = handle_missing_values(df)
    
    # Define features to scale (excluding target variable if any)
    features_to_scale = [
        'price', 'market_cap', 'total_volume', 'hash_rate_ths', 'transaction_volume',
        'MA_10', 'MA_50', 'EMA_10', 'EMA_50', 'RSI_14',
        'Bollinger_High', 'Bollinger_Low', 'Bollinger_Middle',
        'MACD', 'MACD_Signal', 'MACD_Diff', 'EMA_Diff',
        'Price_Lag_1', 'Price_Lag_2', 'Price_Lag_3', 'Price_Lag_5', 'Price_Lag_7',
        'MarketCap_Lag_1', 'MarketCap_Lag_2', 'MarketCap_Lag_3', 'MarketCap_Lag_5', 'MarketCap_Lag_7',
        'TotalVolume_Lag_1', 'TotalVolume_Lag_2', 'TotalVolume_Lag_3', 'TotalVolume_Lag_5', 'TotalVolume_Lag_7',
        'HashRate_Lag_1', 'HashRate_Lag_2', 'HashRate_Lag_3', 'HashRate_Lag_5', 'HashRate_Lag_7',
        'TransactionVolume_Lag_1', 'TransactionVolume_Lag_2', 'TransactionVolume_Lag_3', 'TransactionVolume_Lag_5', 'TransactionVolume_Lag_7',
        'Weekday_Sin', 'Weekday_Cos', 'Month_Sin', 'Month_Cos'
    ]
    
    # Scale Features
    df_scaled = scale_features(df, features_to_scale)
    
    # Save the feature-engineered data
    engineered_csv = os.path.join(FEATURES_DIR, 'feature_engineered_data.csv')
    try:
        df_scaled.to_csv(engineered_csv)
        logging.info(f"Feature-engineered data saved successfully to {engineered_csv}.")
        print(f"\nFeature-engineered data saved successfully to {engineered_csv}.")
    except Exception as err:
        logging.error(f"An error occurred while saving the feature-engineered data: {err}")
        print(f"An error occurred while saving the feature-engineered data: {err}")
    
    print("\n=== Step 5.1: Feature Engineering Completed ===\n")

# -------------------------------
# Main Execution
# -------------------------------

if __name__ == "__main__":
    feature_engineering()
    
    print("All steps completed successfully!")
    print("You can find the feature-engineered data in the 'features' directory within the output directory.")
    print("Logs are saved in 'bitcoin_feature_engineering.log'.")
