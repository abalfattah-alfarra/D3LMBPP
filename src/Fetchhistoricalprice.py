import yfinance as yf
import pandas as pd

def fetch_historical_price(symbol='BTC-USD', period='1y', interval='1d'):
    """
    Fetch historical price data for a given symbol.
    """
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period, interval=interval)
    hist.reset_index(inplace=True)
    hist.rename(columns={
        'Date': 'date',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }, inplace=True)
    return hist

# Fetch data
price_data = fetch_historical_price()
print(price_data.head())

# Save to CSV
price_data.to_csv('D:/rate/historical_price_data.csv', index=False)
