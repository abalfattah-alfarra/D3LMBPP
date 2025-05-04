import requests
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

def fetch_coingecko_data(coin_id='bitcoin', vs_currency='usd', days=365):
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart'
    params = {'vs_currency': vs_currency, 'days': days}
    
    try:
        response = requests.get(url, params=params)
        print(f"Request URL: {response.url}")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
        
        data = response.json()
        
        if 'prices' not in data or 'total_volumes' not in data:
            print("Error: 'prices' or 'total_volumes' key not found in the response.")
            print(f"Response Keys: {list(data.keys())}")
            print(f"Response Content: {data}")
            return None
        
        prices = data['prices']
        volumes = data['total_volumes']
        
        df_prices = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df_volumes = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
        
        df = pd.merge(df_prices, df_volumes, on='timestamp')
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('date', inplace=True)
        df = df[['price', 'volume']]
        
        return df
    
    except requests.exceptions.RequestException as e:
        print(f"Request Exception: {e}")
        return None
    except ValueError as ve:
        print(f"JSON Decode Error: {ve}")
        return None

# Fetch data
bitcoin_df = fetch_coingecko_data()
if bitcoin_df is not None:
    print(bitcoin_df.head())
    
    # Data Preprocessing
    scaler = MinMaxScaler()
    numerical_cols = ['price', 'volume']
    bitcoin_df[numerical_cols] = scaler.fit_transform(bitcoin_df[numerical_cols])
    print("\nScaled DataFrame:")
    print(bitcoin_df.head())
    
    # Save the DataFrame to CSV
    bitcoin_df.to_csv('D:/rate/bitcoin_data.csv', index=True)
    print("\nData has been saved to D:/rate/bitcoin_data.csv")
else:
    print("Failed to fetch Bitcoin data.")
