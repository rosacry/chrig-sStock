import aiohttp
import asyncio
import pandas as pd
import os
import json  # Import JSON for parsing the JSON response
from dotenv import load_dotenv

load_dotenv()
api_token = os.getenv('IEX_API_TOKEN')

async def fetch_iex_symbols():
    api_url = 'https://api.iex.cloud/v1/data/CORE/REF_DATA_IEX_SYMBOLS?token=' + api_token
    headers = {'Authorization': f'Token {api_token}'}
    async with aiohttp.ClientSession() as session:
        async with session.get(api_url, headers=headers) as response:
            if response.status == 200:
                data = await response.text()
                data_json = json.loads(data)  # Convert JSON string to a Python list
                df = pd.json_normalize(data_json)  # Convert list of dictionaries to DataFrame
                return df
            else:
                print(f"Failed to fetch symbols, status code: {response.status}")
                return pd.DataFrame()

def save_symbols_to_csv(df, filename="asset_fetch/tickers/iex-tickers.csv"):
    if not df.empty:
        df.to_csv(filename, index=False)
        print(f"Saved IEX tickers to {filename}")
    else:
        print("No data to save.")

async def main():
    df = await fetch_iex_symbols()
    save_symbols_to_csv(df)

if __name__ == "__main__":
    asyncio.run(main())

