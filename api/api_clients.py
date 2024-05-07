import os
import aiohttp
import cachetools.func
import asyncio

# Market API keys
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
IEX_CLOUD_API_KEY = os.getenv('IEX_CLOUD_API_KEY')
QUANDL_API_KEY = os.getenv('QUANDL_API_KEY')
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
EOD_HISTORICAL_API_KEY = os.getenv('EOD_HISTORICAL_API_KEY')

# Cache for 10 minutes for less volatile data, 2 minutes for high-frequency updates
@cachetools.func.ttl_cache(maxsize=100, ttl=600)
def get_cached_response(url, params):
    async def fetch(url, params):
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                return await response.json()

    return asyncio.run(fetch(url, params))

async def fetch_market_data():
    all_market_data = {}

    # Alpha Vantage
    params_alpha = {"apikey": ALPHA_VANTAGE_API_KEY}
    try:
        all_market_data["alpha_vantage"] = await get_cached_response("https://www.alphavantage.co/query", params_alpha)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Alpha Vantage: {str(e)}")
        all_market_data["alpha_vantage"] = None

    # IEX Cloud
    headers_iex = {"Authorization": f"Bearer {IEX_CLOUD_API_KEY}"}
    try:
        all_market_data["iex_cloud"] = await get_cached_response("https://cloud.iexapis.com/stable", headers_iex)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from IEX Cloud: {str(e)}")
        all_market_data["iex_cloud"] = None

    # Quandl
    params_quandl = {"api_key": QUANDL_API_KEY}
    try:
        all_market_data["quandl"] = await get_cached_response("https://www.quandl.com/api/v3/datasets", params_quandl)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Quandl: {str(e)}")
        all_market_data["quandl"] = None

    # Polygon
    headers_polygon = {"Authorization": f"Bearer {POLYGON_API_KEY}"}
    try:
        all_market_data["polygon"] = await get_cached_response("https://api.polygon.io/v2", headers_polygon)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Polygon: {str(e)}")
        all_market_data["polygon"] = None

    # EOD Historical
    params_eod = {"api_token": EOD_HISTORICAL_API_KEY}
    try:
        all_market_data["eod_historical"] = await get_cached_response("https://eodhistoricaldata.com/api", params_eod)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from EOD Historical: {str(e)}")
        all_market_data["eod_historical"] = None

    return all_market_data
