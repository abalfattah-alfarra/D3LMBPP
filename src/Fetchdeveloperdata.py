import requests
import pandas as pd
import numpy as np
import logging
import os

def fetch_developer_data(coin_id='bitcoin'):
    """
    Fetch developer data from CoinGecko API.
    """
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/developer_data'
    response = requests.get(url)
    if response.status_code != 200:
        logging.error(f"Error fetching developer data: {response.status_code}")
        return None
    data = response.json()
    
    developer_metrics = {
        'forks': data.get('forks', np.nan),
        'stars': data.get('stars', np.nan),
        'subscribers': data.get('subscribers', np.nan),
        'total_issues': data.get('total_issues', np.nan),
        'closed_issues': data.get('closed_issues', np.nan),
        'pull_requests_merged': data.get('pull_requests_merged', np.nan),
        'pull_request_contributors': data.get('pull_request_contributors', np.nan),
        'commit_count_4_weeks': data.get('commit_count_4_weeks', np.nan)
    }
    
    return developer_metrics

def fetch_community_data(coin_id='bitcoin'):
    """
    Fetch community data from CoinGecko API.
    """
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/community_data'
    response = requests.get(url)
    if response.status_code != 200:
        logging.error(f"Error fetching community data: {response.status_code}")
        return None
    data = response.json()
    
    community_metrics = {
        'facebook_likes': data.get('facebook_likes', np.nan),
        'twitter_followers': data.get('twitter_followers', np.nan),
        'reddit_average_posts_48h': data.get('reddit_average_posts_48h', np.nan),
        'reddit_average_comments_48h': data.get('reddit_average_comments_48h', np.nan),
        'reddit_subscribers': data.get('reddit_subscribers', np.nan),
        'telegram_channel_user_count': data.get('telegram_channel_user_count', np.nan)
    }
    
    return community_metrics

def fetch_market_data(coin_id='bitcoin', vs_currency='usd', days=365, interval='daily'):
    """
    Fetch historical market data from CoinGecko API.
    """
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart'
    params = {
        'vs_currency': vs_currency,
        'days': days,
        'interval': interval
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        logging.error(f"Error fetching market data: {response.status_code}")
        return None
    data = response.json()
    
    # Convert timestamps to datetime
    df_prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    df_prices['date'] = pd.to_datetime(df_prices['timestamp'], unit='ms')
    df_prices.set_index('date', inplace=True)
    df_prices.drop('timestamp', axis=1, inplace=True)
    
    df_market_cap = pd.DataFrame(data['market_caps'], columns=['timestamp', 'market_cap'])
    df_market_cap['date'] = pd.to_datetime(df_market_cap['timestamp'], unit='ms')
    df_market_cap.set_index('date', inplace=True)
    df_market_cap.drop('timestamp', axis=1, inplace=True)
    
    df_total_volume = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'total_volume'])
    df_total_volume['date'] = pd.to_datetime(df_total_volume['timestamp'], unit='ms')
    df_total_volume.set_index('date', inplace=True)
    df_total_volume.drop('timestamp', axis=1, inplace=True)
    
    # Merge all market data
    df_market = df_prices.join(df_market_cap).join(df_total_volume)
    df_market.fillna(method='ffill', inplace=True)
    df_market.fillna(method='bfill', inplace=True)
    
    return df_market

# Configure logging
output_dir = 'D:/bitcoin/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

logging.basicConfig(
    filename=os.path.join(output_dir, 'bitcoin_data.log'),
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

def fetch_all_data(coin_id='bitcoin'):
    """
    Fetch all relevant data and merge into a single DataFrame.
    """
    # Fetch market data
    market_data = fetch_market_data(coin_id=coin_id)
    if market_data is None:
        logging.error("Failed to fetch market data.")
        return None
    logging.info("Fetched market data successfully.")
    
    # Fetch developer data
    developer_data = fetch_developer_data(coin_id=coin_id)
    if developer_data:
        df_developer = pd.DataFrame([developer_data], index=market_data.index[:1]).reindex(market_data.index, method='ffill')
    else:
        df_developer = pd.DataFrame(index=market_data.index)
        logging.warning("Developer data is unavailable.")
    
    # Fetch community data
    community_data = fetch_community_data(coin_id=coin_id)
    if community_data:
        df_community = pd.DataFrame([community_data], index=market_data.index[:1]).reindex(market_data.index, method='ffill')
    else:
        df_community = pd.DataFrame(index=market_data.index)
        logging.warning("Community data is unavailable.")
    
    # Merge all data
    combined_df = market_data.join(df_developer).join(df_community)
    
    # Handle missing values
    combined_df.fillna(method='ffill', inplace=True)
    combined_df.fillna(method='bfill', inplace=True)
    
    return combined_df

# Fetch and save all data
combined_data = fetch_all_data()
if combined_data is not None:
    print(combined_data.head())
    combined_data.to_csv(os.path.join(output_dir, 'combined_data.csv'))
    logging.info("Combined data saved successfully.")
else:
    logging.error("No data to save.")
