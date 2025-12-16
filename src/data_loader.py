import logging
import datetime
import requests
import pandas as pd
import yfinance as yf

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
# Henry Hub, Louisiana (Delivery point for NYMEX Natural Gas Futures)
HENRY_HUB_LAT = 29.89
HENRY_HUB_LON = -92.06
START_DATE = "2020-01-01"

def get_market_data(ticker: str) -> pd.DataFrame:
    """
    Fetches daily OHLCV data from Yahoo Finance for a given ticker.
    
    Args:
        ticker (str): Yahoo Finance ticker symbol (e.g., 'NG=F').

    Returns:
        pd.DataFrame: DataFrame containing 'price' and 'volume'.
    """
    logging.info(f"Downloading market data for {ticker}...")
    
    try:
        df = yf.download(ticker, start=START_DATE, progress=False)
        
        # Handle MultiIndex columns 
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Standardize columns
        df = df[['Close', 'Volume']].copy()
        df.rename(columns={'Close': 'price', 'Volume': 'volume'}, inplace=True)
        
        # Remove timezone information to enable merging with weather data
        df.index = df.index.tz_localize(None)
        
        return df

    except Exception as e:
        logging.error(f"Failed to fetch market data: {e}")
        return pd.DataFrame()

def get_weather_data(lat: float, lon: float) -> pd.DataFrame:
    """
    Fetches historical daily mean temperature from Open-Meteo API.
    
    Args:
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.

    Returns:
        pd.DataFrame: DataFrame indexed by date with 'temperature' column.
    """
    logging.info("Querying Open-Meteo API...")
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    # End date set to yesterday to ensure data availability
    end_date = (datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": end_date,
        "daily": "temperature_2m_mean",
        "timezone": "auto"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        df_weather = pd.DataFrame({
            'date': data['daily']['time'],
            'temperature': data['daily']['temperature_2m_mean']
        })
        
        df_weather['date'] = pd.to_datetime(df_weather['date'])
        df_weather.set_index('date', inplace=True)
        
        return df_weather
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Weather API request failed: {e}")
        return pd.DataFrame()

def load_data(ticker: str = "NG=F") -> pd.DataFrame:
    """
    Orchestrates the data ingestion pipeline.
    Merges financial data with local weather conditions at the asset's physical hub.
    """
    # 1. Fetch Market Data
    df_market = get_market_data(ticker)
    if df_market.empty:
        raise ValueError("Market data is empty. Check ticker or connection.")

    # 2. Fetch Weather Data (Henry Hub context)
    df_weather = get_weather_data(HENRY_HUB_LAT, HENRY_HUB_LON)
    
    # 3. Merge Datasets
    # Left join ensures we only keep trading days
    df = df_market.join(df_weather, how='left')
    
    # Forward fill temperature for weekends/holidays if necessary
    df['temperature'] = df['temperature'].ffill()
    
    df.dropna(inplace=True)
    
    logging.info(f"Data Pipeline completed. Rows: {len(df)}")
    return df

if __name__ == "__main__":
    # Smoke test
    data = load_data("NG=F")
    print(data.head())