from alpaca_trade_api.rest import REST, TimeFrame
from asset_fetch.iex_symbols import fetch_iex_symbols 
import pandas as pd
import asyncio
import aiohttp
import backoff
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv('AlPACA_API_KEY') 
secret_key = os.getenv('AlPACA_SECRET_KEY')
base_url = "https://paper-api.alpaca.markets"  # Use the appropriate URL for paper or live trading

client = REST(api_key, secret_key, base_url)  # Initializing the REST client

async def robust_fetch(session, url, params=None, headers=None):
    """ Asynchronously fetch data from a URL with error handling and exponential backoff. """
    @backoff.on_exception(backoff.expo, aiohttp.ClientError, max_tries=5)
    async def get_request():
        async with session.get(url, params=params, headers=headers) as response:
            response.raise_for_status()  # Will raise HTTPError for bad responses
            return await response.json()

    async with aiohttp.ClientSession() as session:
        try:
            return await get_request()
        except aiohttp.ClientError as http_err:
            print(f"HTTP error occurred: {http_err}")
        except Exception as err:
            print(f"Other error occurred: {err}")
        return None


async def fetch_market_data(symbols, timeframe=TimeFrame.Minute, chunk_size=100):
    """Fetch market data asynchronously for given symbols using Alpaca SDK in real-time."""
    responses = []
    now = pd.Timestamp.utcnow()
    start = (now - pd.DateOffset(minutes=5)).isoformat()
    end = now.isoformat()
    headers = {
        'APCA-API-KEY-ID': api_key,
        'APCA-API-SECRET-KEY': secret_key 
    }

    async with aiohttp.ClientSession() as session:
        for i in range(0, len(symbols), chunk_size):
            symbol_chunk = symbols[i:i + chunk_size]
            url = f"https://data.alpaca.markets/v2/stocks/bars"
            params = {
                'symbols': ','.join(symbol_chunk),
                'start': start,
                'end': end,
                'timeframe': timeframe.value
            }
            data = await robust_fetch(session, url, params=params, headers=headers)
            if data:
                responses.append(pd.DataFrame(data))
            await asyncio.sleep(60)  # Sleep to respect rate limit

    return pd.concat(responses, ignore_index=True)

async def fetch_news_data(api_key, secret_key, symbols=None, max_requests_per_minute=10):
    """Fetches news data for specified symbols from Alpaca's News API."""
    base_url = "https://data.alpaca.markets/v1beta1/news"
    headers = {
        'Apca-Api-Key-Id': api_key,
        'Apca-Api-Secret-Key': secret_key
    }
    params = {'symbols': symbols} if symbols else {}

    async with aiohttp.ClientSession() as session:
        news_articles = await robust_fetch(session, base_url, params, headers)
        return news_articles.get('news', []) if news_articles else []

# Usage of asyncio to run the async functions
async def main():
    symbols = fetch_iex_symbols()
    market_data = await fetch_market_data(symbols)
    news_data = await fetch_news_data(api_key, secret_key)
    print(market_data, news_data)

if __name__ == "__main__":
    asyncio.run(main())