# import os
# import requests
# import time
# from dotenv import load_dotenv

# # Load API keys from environment variables
# load_dotenv()
# ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
# IEX_CLOUD_API_KEY = os.getenv('IEX_CLOUD_API_KEY')
# YAHOO_FINANCE_API_KEY = os.getenv('YAHOO_FINANCE_API_KEY')
# QUANDL_API_KEY = os.getenv('QUANDL_API_KEY')
# POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
# EOD_HISTORICAL_API_KEY = os.getenv('EOD_HISTORICAL_API_KEY')
# # BLOOMBERG_API_KEY = os.getenv('BLOOMBERG_API_KEY')

# MAX_RETRIES = 5
# BACKOFF_FACTOR = 2

# def retry_request(func):
#     """Decorator to retry API requests with exponential back-off."""
#     def wrapper(*args, **kwargs):
#         retries = 0
#         while retries < MAX_RETRIES:
#             try:
#                 return func(*args, **kwargs)
#             except requests.exceptions.RequestException:
#                 retries += 1
#                 wait_time = BACKOFF_FACTOR ** retries
#                 print(f"Request failed ({retries}/{MAX_RETRIES}). Retrying in {wait_time}s...")
#                 time.sleep(wait_time)
#         raise Exception(f"Request failed after {MAX_RETRIES} retries.")
#     return wrapper


# class AlphaVantageClient:
#     """Client to interact with the Alpha Vantage API."""
#     BASE_URL = "https://www.alphavantage.co/query"

#     def __init__(self, api_key: str):
#         self.api_key = api_key

#     @retry_request
#     def get_stock_data(self, symbol: str):
#         """Fetches stock data."""
#         params = {
#             "function": "TIME_SERIES_DAILY",
#             "symbol": symbol,
#             "apikey": self.api_key
#         }
#         response = requests.get(self.BASE_URL, params=params)
#         response.raise_for_status()
#         return response.json()

#     @retry_request
#     def get_crypto_data(self, symbol: str, market: str = "USD"):
#         """Fetches crypto data."""
#         params = {
#             "function": "DIGITAL_CURRENCY_DAILY",
#             "symbol": symbol,
#             "market": market,
#             "apikey": self.api_key
#         }
#         response = requests.get(self.BASE_URL, params=params)
#         response.raise_for_status()
#         return response.json()


# class IexCloudClient:
#     """Client to interact with the IEX Cloud API."""
#     BASE_URL = "https://cloud.iexapis.com/stable"

#     def __init__(self, api_key: str):
#         self.api_key = api_key

#     @retry_request
#     def get_stock_data(self, symbol: str):
#         """Fetches stock data."""
#         url = f"{self.BASE_URL}/stock/{symbol}/quote"
#         params = {"token": self.api_key}
#         response = requests.get(url, params=params)
#         response.raise_for_status()
#         return response.json()

#     @retry_request
#     def get_options_data(self, symbol: str):
#         """Fetches options data."""
#         url = f"{self.BASE_URL}/stock/{symbol}/options"
#         params = {"token": self.api_key}
#         response = requests.get(url, params=params)
#         response.raise_for_status()
#         return response.json()


# class YahooFinanceClient:
#     """Client to interact with Yahoo Finance API."""
#     BASE_URL = "https://yfapi.net/v8/finance/chart"

#     def __init__(self, api_key: str):
#         self.api_key = api_key

#     @retry_request
#     def get_stock_data(self, symbol: str):
#         """Fetches stock data."""
#         headers = {"x-api-key": self.api_key}
#         params = {
#             "symbol": symbol,
#             "interval": "1d"
#         }
#         response = requests.get(self.BASE_URL, headers=headers, params=params)
#         response.raise_for_status()
#         return response.json()

#     # @retry_request
#     # def get_index_data(self, symbol: str):
#     #     """Fetches index data."""
#     #     headers = {"x-api-key": self.api_key}
#     #     params = {
#     #         "symbol": symbol,
#     #         "interval": "1d"
#     #     }
#     #     response = requests.get(self.BASE_URL, headers=headers, params=params)
#     #     response.raise_for_status()
#     #     return response.json()


# class QuandlClient:
#     """Client to interact with the Quandl API."""
#     BASE_URL = "https://www.quandl.com/api/v3/datasets"

#     def __init__(self, api_key: str):
#         self.api_key = api_key

#     @retry_request
#     def get_stock_data(self, dataset_code: str, symbol: str):
#         """Fetches stock data."""
#         url = f"{self.BASE_URL}/{dataset_code}/{symbol}.json"
#         params = {"api_key": self.api_key}
#         response = requests.get(url, params=params)
#         response.raise_for_status()
#         return response.json()


# class PolygonClient:
#     """Client to interact with the Polygon API."""
#     BASE_URL = "https://api.polygon.io/v2"

#     def __init__(self, api_key: str):
#         self.api_key = api_key

#     @retry_request
#     def get_stock_data(self, symbol: str):
#         """Fetches stock data."""
#         url = f"{self.BASE_URL}/aggs/ticker/{symbol}/range/1/day/2021-01-01/2021-12-31"
#         params = {"apiKey": self.api_key}
#         response = requests.get(url, params=params)
#         response.raise_for_status()
#         return response.json()

#     @retry_request
#     def get_crypto_data(self, symbol: str):
#         """Fetches cryptocurrency data."""
#         url = f"{self.BASE_URL}/aggs/ticker/X:{symbol}USD/range/1/day/2021-01-01/2021-12-31"
#         params = {"apiKey": self.api_key}
#         response = requests.get(url, params=params)
#         response.raise_for_status()
#         return response.json()

#     # @retry_request
#     # def get_forex_data(self, from_currency: str, to_currency: str):
#     #     """Fetches foreign exchange data."""
#     #     url = f"{self.BASE_URL}/aggs/ticker/C:{from_currency}{to_currency}/range/1/day/2021-01-01/2021-12-31"
#     #     params = {"apiKey": self.api_key}
#     #     response = requests.get(url, params=params)
#     #     response.raise_for_status()
#     #     return response.json()


# class EodHistoricalDataClient:
#     """Client to interact with the EOD Historical Data API."""
#     BASE_URL = "https://eodhistoricaldata.com/api/eod"

#     def __init__(self, api_key: str):
#         self.api_key = api_key

#     @retry_request
#     def get_stock_data(self, symbol: str):
#         """Fetches stock data."""
#         params = {
#             "api_token": self.api_key,
#             "fmt": "json"
#         }
#         response = requests.get(f"{self.BASE_URL}/{symbol}.US", params=params)
#         response.raise_for_status()
#         return response.json()


# class BloombergClient:
#     """Client to interact with Bloomberg Market and Financial News API."""
#     BASE_URL = "https://www.bloomberg.com/markets/api"

#     def __init__(self, api_key: str):
#         self.api_key = api_key

#     @retry_request
#     def get_financial_news(self):
#         """Fetches financial news."""
#         headers = {"x-api-key": self.api_key}
#         response = requests.get(f"{self.BASE_URL}/news", headers=headers)
#         response.raise_for_status()
#         return response.json()


# def aggregate_data(asset: str) -> dict:
#     """Fetches data from multiple APIs.

#     Args:
#         asset (str): The name or symbol of the asset.

#     Returns:
#         dict: Aggregated data.
#     """
#     clients = [
#         AlphaVantageClient(ALPHA_VANTAGE_API_KEY),
#         IexCloudClient(IEX_CLOUD_API_KEY),
#         YahooFinanceClient(YAHOO_FINANCE_API_KEY),
#         QuandlClient(QUANDL_API_KEY),
#         PolygonClient(POLYGON_API_KEY),
#         EodHistoricalDataClient(EOD_HISTORICAL_API_KEY),
#         # BloombergClient(BLOOMBERG_API_KEY)
#     ]

#     aggregated_data = {"asset": asset}

#     for client in clients:
#         client_name = client.__class__.__name__
#         aggregated_data[client_name] = {
#             "stock": client.get_stock_data(asset),
#             "crypto": client.get_crypto_data(asset),
#             "options": client.get_options_data(asset)
#         }

#     return aggregated_data
# /mnt/data/api_clients.py
# /mnt/data/api_clients.py

import os
import requests

# Market API keys
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
IEX_CLOUD_API_KEY = os.getenv('IEX_CLOUD_API_KEY')
YAHOO_FINANCE_API_KEY = os.getenv('YAHOO_FINANCE_API_KEY')
QUANDL_API_KEY = os.getenv('QUANDL_API_KEY')
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
EOD_HISTORICAL_API_KEY = os.getenv('EOD_HISTORICAL_API_KEY')

# Market API URLs
MARKET_API_ENDPOINTS = {
    "alpha_vantage": "https://www.alphavantage.co/query",
    "iex_cloud": "https://cloud.iexapis.com/stable",
    "yahoo_finance": "https://yahoo-finance-api.com/",
    "quandl": "https://www.quandl.com/api/v3/datasets",
    "polygon": "https://api.polygon.io/v2",
    "eod_historical": "https://eodhistoricaldata.com/api"
}

# Investor API URLs
INVESTOR_API_ENDPOINTS = {
    "investor_1": "https://example.com/api/investor-1",
    "investor_2": "https://example.com/api/investor-2"
}

def fetch_market_data():
    """Fetch market data from multiple APIs using their respective API keys."""
    all_market_data = {}

    # Alpha Vantage
    params_alpha = {"apikey": ALPHA_VANTAGE_API_KEY}
    try:
        response = requests.get(MARKET_API_ENDPOINTS["alpha_vantage"], params=params_alpha)
        response.raise_for_status()
        all_market_data["alpha_vantage"] = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Alpha Vantage: {str(e)}")
        all_market_data["alpha_vantage"] = None

    # IEX Cloud
    headers_iex = {"Authorization": f"Bearer {IEX_CLOUD_API_KEY}"}
    try:
        response = requests.get(MARKET_API_ENDPOINTS["iex_cloud"], headers=headers_iex)
        response.raise_for_status()
        all_market_data["iex_cloud"] = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from IEX Cloud: {str(e)}")
        all_market_data["iex_cloud"] = None

    # Yahoo Finance
    headers_yahoo = {"Authorization": f"Bearer {YAHOO_FINANCE_API_KEY}"}
    try:
        response = requests.get(MARKET_API_ENDPOINTS["yahoo_finance"], headers=headers_yahoo)
        response.raise_for_status()
        all_market_data["yahoo_finance"] = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Yahoo Finance: {str(e)}")
        all_market_data["yahoo_finance"] = None

    # Quandl
    params_quandl = {"api_key": QUANDL_API_KEY}
    try:
        response = requests.get(MARKET_API_ENDPOINTS["quandl"], params=params_quandl)
        response.raise_for_status()
        all_market_data["quandl"] = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Quandl: {str(e)}")
        all_market_data["quandl"] = None

    # Polygon
    headers_polygon = {"Authorization": f"Bearer {POLYGON_API_KEY}"}
    try:
        response = requests.get(MARKET_API_ENDPOINTS["polygon"], headers=headers_polygon)
        response.raise_for_status()
        all_market_data["polygon"] = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Polygon: {str(e)}")
        all_market_data["polygon"] = None

    # EOD Historical
    params_eod = {"api_token": EOD_HISTORICAL_API_KEY}
    try:
        response = requests.get(MARKET_API_ENDPOINTS["eod_historical"], params=params_eod)
        response.raise_for_status()
        all_market_data["eod_historical"] = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from EOD Historical: {str(e)}")
        all_market_data["eod_historical"] = None

    return all_market_data


def fetch_top_investors_data():
    """Fetch data about top investors from multiple APIs using their respective API keys."""
    all_investors_data = {}

    # Example: Replace this with actual APIs for investor data
    headers_investor_1 = {"Authorization": f"Bearer {os.getenv('INVESTOR_API_KEY_1')}"}
    headers_investor_2 = {"Authorization": f"Bearer {os.getenv('INVESTOR_API_KEY_2')}"}

    # Investor API 1
    try:
        response = requests.get(INVESTOR_API_ENDPOINTS["investor_1"], headers=headers_investor_1)
        response.raise_for_status()
        all_investors_data["investor_1"] = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Investor 1: {str(e)}")
        all_investors_data["investor_1"] = None

    # Investor API 2
    try:
        response = requests.get(INVESTOR_API_ENDPOINTS["investor_2"], headers=headers_investor_2)
        response.raise_for_status()
        all_investors_data["investor_2"] = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Investor 2: {str(e)}")
        all_investors_data["investor_2"] = None

    return all_investors_data

