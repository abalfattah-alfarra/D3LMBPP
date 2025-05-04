import requests
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np

def fetch_market_data(coin_id='bitcoin', vs_currency='usd', days=365, interval='daily'):
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart'
    params = {
        'vs_currency': vs_currency,
        'days': days,
        'interval': interval
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Error fetching market data: {response.status_code}")
        return None
    data = response.json()
    
    df = pd.DataFrame({
        'timestamp': [entry[0] for entry in data['prices']],
        'price': [entry[1] for entry in data['prices']],
        'market_cap': [entry[1] for entry in data['market_caps']],
        'total_volume': [entry[1] for entry in data['total_volumes']]
    })
    
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)
    df.drop('timestamp', axis=1, inplace=True)
    
    return df

def fetch_supply_data(coin_id='bitcoin'):
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}'
    params = {
        'localization': 'false',
        'tickers': 'false',
        'market_data': 'true',
        'community_data': 'false',
        'developer_data': 'false',
        'sparkline': 'false'
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Error fetching supply data: {response.status_code}")
        return None
    data = response.json()
    
    supply_data = {
        'max_supply': data['market_data'].get('max_supply', None),
        'total_supply': data['market_data'].get('total_supply', None),
        'circulating_supply': data['market_data'].get('circulating_supply', None)
    }
    
    return supply_data

def fetch_developer_data(coin_id='bitcoin'):
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/developer_data'
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error fetching developer data: {response.status_code}")
        return None
    data = response.json()
    
    developer_metrics = {
        'forks': data.get('forks', 0),
        'stars': data.get('stars', 0),
        'subscribers': data.get('subscribers', 0),
        'total_issues': data.get('total_issues', 0),
        'closed_issues': data.get('closed_issues', 0),
        'pull_requests_merged': data.get('pull_requests_merged', 0),
        'pull_request_contributors': data.get('pull_request_contributors', 0),
        'commit_count_4_weeks': data.get('commit_count_4_weeks', 0)
    }
    
    return developer_metrics

def fetch_community_data(coin_id='bitcoin'):
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/community_data'
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error fetching community data: {response.status_code}")
        return None
    data = response.json()
    
    community_metrics = {
        'facebook_likes': data.get('facebook_likes', 0),
        'twitter_followers': data.get('twitter_followers', 0),
        'reddit_average_posts_48h': data.get('reddit_average_posts_48h', 0),
        'reddit_average_comments_48h': data.get('reddit_average_comments_48h', 0),
        'reddit_subscribers': data.get('reddit_subscribers', 0),
        'telegram_channel_user_count': data.get('telegram_channel_user_count', 0)
    }
    
    return community_metrics

# Fetch all data
market_df = fetch_market_data()
supply_data = fetch_supply_data()
developer_data = fetch_developer_data()
community_data = fetch_community_data()

if market_df is not None:
    print("Market Data:")
    print(market_df.head())
    
    # Save raw market data
    raw_csv_path = 'D:/rate/bitcoin_data_raw.csv'
    market_df.to_csv(raw_csv_path, index=True)
    print(f"\nRaw market data saved to {raw_csv_path}")
    
    # Scaling
    scaler = MinMaxScaler()
    numerical_cols = ['price', 'market_cap', 'total_volume']
    market_df_scaled = market_df.copy()
    market_df_scaled[numerical_cols] = scaler.fit_transform(market_df_scaled[numerical_cols])
    print("\nScaled Market Data:")
    print(market_df_scaled.head())
    
    # Save scaled market data
    scaled_csv_path = 'D:/rate/bitcoin_data_scaled.csv'
    market_df_scaled.to_csv(scaled_csv_path, index=True)
    print(f"\nScaled market data saved to {scaled_csv_path}")
    
    # Initialize empty DataFrames for supply, developer, community data
    if supply_data:
        supply_df = pd.DataFrame([supply_data]*len(market_df_scaled), index=market_df_scaled.index)
    else:
        supply_df = pd.DataFrame(index=market_df_scaled.index)
    
    if developer_data:
        developer_df = pd.DataFrame([developer_data]*len(market_df_scaled), index=market_df_scaled.index)
    else:
        developer_df = pd.DataFrame(index=market_df_scaled.index)
    
    if community_data:
        community_df = pd.DataFrame([community_data]*len(market_df_scaled), index=market_df_scaled.index)
    else:
        community_df = pd.DataFrame(index=market_df_scaled.index)
    
    # Merge all DataFrames
    # To avoid overlapping columns, specify suffixes or ensure unique column names
    combined_df = market_df_scaled.join(supply_df, how='left') \
                                  .join(developer_df, how='left', lsuffix='_market', rsuffix='_dev') \
                                  .join(community_df, how='left', lsuffix='_market', rsuffix='_comm')
    
    # Alternatively, handle missing DataFrames
    # combined_df = market_df_scaled
    # if supply_df is not empty:
    #     combined_df = combined_df.join(supply_df, how='left')
    # if developer_df is not empty:
    #     combined_df = combined_df.join(developer_df, how='left', rsuffix='_dev')
    # if community_df is not empty:
    #     combined_df = combined_df.join(community_df, how='left', rsuffix='_comm')
    
    # Handle missing values if any
    combined_df.fillna(method='ffill', inplace=True)
    
    # Save the combined DataFrame
    combined_csv_path = 'D:/rate/bitcoin_data_combined.csv'
    combined_df.to_csv(combined_csv_path, index=True)
    print(f"\nCombined data saved to {combined_csv_path}")
    
    # Example: Calculate Technical Indicators
    # Simple Moving Average (SMA)
    combined_df['SMA_30'] = combined_df['price'].rolling(window=30).mean()
    
    # Relative Strength Index (RSI)
    def calculate_rsi(series, window=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    combined_df['RSI_14'] = calculate_rsi(combined_df['price'], window=14)
    
    # Moving Average Convergence Divergence (MACD)
    def calculate_macd(series, span_short=12, span_long=26, span_signal=9):
        ema_short = series.ewm(span=span_short, adjust=False).mean()
        ema_long = series.ewm(span=span_long, adjust=False).mean()
        macd = ema_short - ema_long
        signal = macd.ewm(span=span_signal, adjust=False).mean()
        histogram = macd - signal
        return macd, signal, histogram
    
    macd, signal, histogram = calculate_macd(combined_df['price'])
    combined_df['MACD'] = macd
    combined_df['MACD_Signal'] = signal
    combined_df['MACD_Histogram'] = histogram
    
    # Drop NaN values resulting from rolling calculations
    combined_df.dropna(inplace=True)
    
    # Save DataFrame with Technical Indicators
    indicators_csv_path = 'D:/rate/bitcoin_data_with_indicators.csv'
    combined_df.to_csv(indicators_csv_path, index=True)
    print(f"\nData with Technical Indicators saved to {indicators_csv_path}")
    
    # Example: Visualization of RSI
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12,6))
    plt.plot(combined_df.index, combined_df['RSI_14'], label='RSI 14')
    plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
    plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
    plt.title('Relative Strength Index (RSI)')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    plt.grid(True)
    plt.show()
    
else:
    print("Failed to fetch market data.")
