from aiopg.sa import create_engine
import asyncio
import os
import aiohttp
import cachetools.func
import asyncio
import requests

async def create_db_pool():
    return await create_engine(
        user='your_username',
        database='your_database',
        host='your_host',
        password='your_password',
        minsize=1,
        maxsize=10  # Adjust pool size according to your application's requirements
    )

# Global variable to hold the connection pool
db_pool = asyncio.run(create_db_pool())




# Market API keys
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
# IEX_CLOUD_API_KEY = os.getenv('IEX_CLOUD_API_KEY')
# QUANDL_API_KEY = os.getenv('QUANDL_API_KEY')
# POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
# EOD_HISTORICAL_API_KEY = os.getenv('EOD_HISTORICAL_API_KEY')

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
        all_market_data["alpha_vantage"] = await get_cached_response("https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey=demo", params_alpha)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Alpha Vantage: {str(e)}")
        all_market_data["alpha_vantage"] = None

    # # IEX Cloud
    # headers_iex = {"Authorization": f"Bearer {IEX_CLOUD_API_KEY}"}
    # try:
    #     all_market_data["iex_cloud"] = await get_cached_response("https://cloud.iexapis.com/stable", headers_iex)
    # except requests.exceptions.RequestException as e:
    #     print(f"Error fetching data from IEX Cloud: {str(e)}")
    #     all_market_data["iex_cloud"] = None

    # # Quandl
    # params_quandl = {"api_key": QUANDL_API_KEY}
    # try:
    #     all_market_data["quandl"] = await get_cached_response("https://www.quandl.com/api/v3/datasets", params_quandl)
    # except requests.exceptions.RequestException as e:
    #     print(f"Error fetching data from Quandl: {str(e)}")
    #     all_market_data["quandl"] = None

    # # Polygon
    # headers_polygon = {"Authorization": f"Bearer {POLYGON_API_KEY}"}
    # try:
    #     all_market_data["polygon"] = await get_cached_response("https://api.polygon.io/v2", headers_polygon)
    # except requests.exceptions.RequestException as e:
    #     print(f"Error fetching data from Polygon: {str(e)}")
    #     all_market_data["polygon"] = None

    # # EOD Historical
    # params_eod = {"api_token": EOD_HISTORICAL_API_KEY}
    # try:
    #     all_market_data["eod_historical"] = await get_cached_response("https://eodhistoricaldata.com/api", params_eod)
    # except requests.exceptions.RequestException as e:
    #     print(f"Error fetching data from EOD Historical: {str(e)}")
    #     all_market_data["eod_historical"] = None

    return all_market_data

# async def fetch_news_data():
#     # Assuming API details and endpoints for news data
#     news_api_key = os.getenv('NEWS_API_KEY')
#     try:
#         news_data = await get_cached_response("https://newsapi.org/v2/everything", {"apikey": news_api_key})
#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching news data: {str(e)}")
#         news_data = None
#     return news_data

# async def fetch_social_media_data():
#     # Assuming API details and endpoints for social media sentiment data
#     social_media_api_key = os.getenv('SOCIAL_MEDIA_API_KEY')
#     try:
#         social_media_data = await get_cached_response("https://socialmediaapi.com/data", {"apikey": social_media_api_key})
#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching social media data: {str(e)}")
#         social_media_data = None
#     return social_media_data

# async def fetch_top_investors_data():
#     pass

# Updating user funds
async def update_user_funds(amount):
    # Add funds to the user's account
    # Placeholder for updating funds logic
    print(f"Updated user funds by ${amount}")

# Withdrawing user funds
async def withdraw_user_funds(amount, portfolio):
    # Withdraw funds from the user's account
    # Placeholder for withdrawing funds logic
    print(f"Withdrew ${amount} from user funds based on portfolio")

# Fetching portfolio data
async def fetch_portfolio_data():
    # Fetch current portfolio data
    # Placeholder for fetching portfolio data logic
    return {"stocks": {"AAPL": 100, "GOOGL": 150}, "total_value": 25000}

