import aiohttp
import asyncio
import backoff

async def robust_fetch(session, url, params=None):
    """ Asynchronously fetch data from a URL with error handling and exponential backoff. """
    @backoff.on_exception(backoff.expo, aiohttp.ClientError, max_tries=5)
    async def get_request():
        async with session.get(url, params=params) as response:
            response.raise_for_status()  # Will raise HTTPError for bad responses
            return await response.json()

    try:
        return await get_request()
    except aiohttp.ClientError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")

async def fetch_market_data(api_key, real_time=True):
    """ Fetch market data asynchronously using API key, with a choice between real-time or historical data. """
    async with aiohttp.ClientSession() as session:
        base_url = "https://api.financialmodelingprep.com/api/v3"
        market_url = f"{base_url}/quotes/nyse" if real_time else f"{base_url}/historical/nyse"
        return await robust_fetch(session, market_url, params={"apikey": api_key})

async def fetch_news_data(api_key, real_time=True):
    """ Fetch news data asynchronously using API key, with a choice between real-time or historical data. """
    async with aiohttp.ClientSession() as session:
        base_url = "https://api.newsprovider.com"
        news_url = f"{base_url}/news" if real_time else f"{base_url}/historical/news"
        return await robust_fetch(session, news_url, params={"apikey": api_key})

async def fetch_top_investors_data(api_key):
    """ Fetch top investors data asynchronously using API key. """
    async with aiohttp.ClientSession() as session:
        investors_url = "https://api.example.com/top_investors"
        return await robust_fetch(session, investors_url, params={"apikey": api_key})

async def fetch_social_media_data(api_key):
    """ Fetch social media data asynchronously using API key. """
    async with aiohttp.ClientSession() as session:
        social_media_url = "https://api.example.com/social_media"
        return await robust_fetch(session, social_media_url, params={"apikey": api_key})

async def main():
    """ Main function to execute asynchronous API calls. """
    market_data = await fetch_market_data("your_api_key", real_time=False)
    news_data = await fetch_news_data("your_api_key", real_time=False)
    investors_data = await fetch_top_investors_data("your_api_key")
    social_media_data = await fetch_social_media_data("your_api_key")
    print(market_data, news_data, investors_data, social_media_data)

if __name__ == "__main__":
    asyncio.run(main())
