import aiohttp
import asyncio
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
api_token = os.getenv('IEX_API_TOKEN')

async def fetch_iex_symbols():
    api_url = 'https://api.iex.cloud/v1/data/CORE/REF_DATA_IEX_SYMBOLS?token=' + api_token
    headers = {'Authorization': f'Token {api_token}'}
    async with aiohttp.ClientSession() as session:
        async with session.get(api_url, headers=headers) as response:
            if response.status == 200:
                # Assuming the response returns CSV data
                df = pd.read_csv(await response.text())
                symbols = df['symbol'].tolist()  # Assuming the file has a column named 'symbol'
                return symbols
            else:
                print(f"Failed to fetch symbols, status code: {response.status}")
                return []


if __name__ == "__main__":
    symbols = fetch_iex_symbols()
    print(symbols)
