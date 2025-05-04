import os
import logging
import sys
from datetime import datetime
from san import ApiConfig, get
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
import inspect

# Define the path to the san_key.env file
env_path = Path('')  # Use forward slashes or raw string

# Check if san_key.env file exists
if not env_path.exists():
    logging.error(f"san_key.env file not found at {env_path}.")
    sys.exit(1)

# Load environment variables from san_key.env file
load_dotenv(dotenv_path=env_path)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Print Current Working Directory
print(f"Current Working Directory: {os.getcwd()}")

# Retrieve the SAN_API_KEY
san_api_key = os.getenv('SAN_API_KEY')
print(f"SAN_API_KEY: {san_api_key}")  # Remove or comment out after verification

# Set your API key from environment variable
ApiConfig.api_key = san_api_key

if not ApiConfig.api_key:
    logging.error("API key not found. Please set the SAN_API_KEY environment variable.")
    sys.exit(1)

# Define the parameters within allowed limits
dataset = "sentiment_volume_consumed_total"  # Updated metric
slug = "bitcoin"
from_date = "2024-04-01T00:00:00Z"  # April 1, 2024
to_date = "2024-12-09T00:00:00Z"    # December 9, 2024
interval = "1d"  # Daily intervals

# Validate dates
def validate_date(date_text):
    try:
        datetime.strptime(date_text, '%Y-%m-%dT%H:%M:%SZ')
    except ValueError:
        raise ValueError(f"Incorrect date format for '{date_text}'. Expected format: YYYY-MM-DDTHH:MM:SSZ.")

try:
    validate_date(from_date)
    validate_date(to_date)
except ValueError as ve:
    logging.error(ve)
    sys.exit(1)

try:
    # Fetch the data with dataset and slug as arguments
    data = get(
        dataset,             # Pass 'dataset' as the first positional argument
        slug=slug,           # Pass 'slug' as a keyword argument
        from_date=from_date,
        to_date=to_date,
        interval=interval
    )
    
    # Process the data based on its type
    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        df = pd.DataFrame(data)
        logging.info("First 5 rows of the data:")
        logging.info(df.head())
        df.to_csv("bitcoin_sentiment_data.csv", index=False)
        logging.info("Data saved to bitcoin_sentiment_data.csv.")
    elif isinstance(data, pd.DataFrame):
        logging.info("Data is a DataFrame.")
        logging.info(data.head())
        # Reset index to have 'datetime' as a column instead of index
        df = data.reset_index()
        df.to_csv("bitcoin_sentiment_data.csv", index=False)
        logging.info("Data saved to bitcoin_sentiment_data.csv.")
    else:
        logging.error("Unexpected data format received:")
        logging.error(data)
except Exception as e:
    logging.error(f"An error occurred: {e}")
