import os
import requests
import time
from dotenv import load_dotenv

# Load API keys from environment variables
load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
IEX_CLOUD_API_KEY = os.getenv('IEX_CLOUD_API_KEY')
YAHOO_FINANCE_API_KEY = os.getenv('YAHOO_FINANCE_API_KEY')
QUANDL_API_KEY = os.getenv('QUANDL_API_KEY')
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
EOD_HISTORICAL_API_KEY = os.getenv('EOD_HISTORICAL_API_KEY')
# BLOOMBERG_API_KEY = os.getenv('BLOOMBERG_API_KEY')

MAX_RETRIES = 5
BACKOFF_FACTOR = 2

def retry_request(func):
    """Decorator to retry API requests with exponential back-off."""
    def wrapper(*args, **kwargs):
        retries = 0
        while retries < MAX_RETRIES:
            try:
                return func(*args, **kwargs)
            except requests.exceptions.RequestException:
                retries += 1
                wait_time = BACKOFF_FACTOR ** retries
                print(f"Request failed ({retries}/{MAX_RETRIES}). Retrying in {wait_time}s...")
                time.sleep(wait_time)
        raise Exception(f"Request failed after {MAX_RETRIES} retries.")
    return wrapper


class AlphaVantageClient:
    """Client to interact with the Alpha Vantage API."""
    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str):
        self.api_key = api_key

    @retry_request
    def get_stock_data(self, symbol: str):
        """Fetches stock data."""
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "apikey": self.api_key
        }
        response = requests.get(self.BASE_URL, params=params)
        response.raise_for_status()
        return response.json()

    @retry_request
    def get_crypto_data(self, symbol: str, market: str = "USD"):
        """Fetches crypto data."""
        params = {
            "function": "DIGITAL_CURRENCY_DAILY",
            "symbol": symbol,
            "market": market,
            "apikey": self.api_key
        }
        response = requests.get(self.BASE_URL, params=params)
        response.raise_for_status()
        return response.json()


class IexCloudClient:
    """Client to interact with the IEX Cloud API."""
    BASE_URL = "https://cloud.iexapis.com/stable"

    def __init__(self, api_key: str):
        self.api_key = api_key

    @retry_request
    def get_stock_data(self, symbol: str):
        """Fetches stock data."""
        url = f"{self.BASE_URL}/stock/{symbol}/quote"
        params = {"token": self.api_key}
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()

    @retry_request
    def get_options_data(self, symbol: str):
        """Fetches options data."""
        url = f"{self.BASE_URL}/stock/{symbol}/options"
        params = {"token": self.api_key}
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()


class YahooFinanceClient:
    """Client to interact with Yahoo Finance API."""
    BASE_URL = "https://yfapi.net/v8/finance/chart"

    def __init__(self, api_key: str):
        self.api_key = api_key

    @retry_request
    def get_stock_data(self, symbol: str):
        """Fetches stock data."""
        headers = {"x-api-key": self.api_key}
        params = {
            "symbol": symbol,
            "interval": "1d"
        }
        response = requests.get(self.BASE_URL, headers=headers, params=params)
        response.raise_for_status()
        return response.json()

    # @retry_request
    # def get_index_data(self, symbol: str):
    #     """Fetches index data."""
    #     headers = {"x-api-key": self.api_key}
    #     params = {
    #         "symbol": symbol,
    #         "interval": "1d"
    #     }
    #     response = requests.get(self.BASE_URL, headers=headers, params=params)
    #     response.raise_for_status()
    #     return response.json()


class QuandlClient:
    """Client to interact with the Quandl API."""
    BASE_URL = "https://www.quandl.com/api/v3/datasets"

    def __init__(self, api_key: str):
        self.api_key = api_key

    @retry_request
    def get_stock_data(self, dataset_code: str, symbol: str):
        """Fetches stock data."""
        url = f"{self.BASE_URL}/{dataset_code}/{symbol}.json"
        params = {"api_key": self.api_key}
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()


class PolygonClient:
    """Client to interact with the Polygon API."""
    BASE_URL = "https://api.polygon.io/v2"

    def __init__(self, api_key: str):
        self.api_key = api_key

    @retry_request
    def get_stock_data(self, symbol: str):
        """Fetches stock data."""
        url = f"{self.BASE_URL}/aggs/ticker/{symbol}/range/1/day/2021-01-01/2021-12-31"
        params = {"apiKey": self.api_key}
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()

    @retry_request
    def get_crypto_data(self, symbol: str):
        """Fetches cryptocurrency data."""
        url = f"{self.BASE_URL}/aggs/ticker/X:{symbol}USD/range/1/day/2021-01-01/2021-12-31"
        params = {"apiKey": self.api_key}
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()

    # @retry_request
    # def get_forex_data(self, from_currency: str, to_currency: str):
    #     """Fetches foreign exchange data."""
    #     url = f"{self.BASE_URL}/aggs/ticker/C:{from_currency}{to_currency}/range/1/day/2021-01-01/2021-12-31"
    #     params = {"apiKey": self.api_key}
    #     response = requests.get(url, params=params)
    #     response.raise_for_status()
    #     return response.json()


class EodHistoricalDataClient:
    """Client to interact with the EOD Historical Data API."""
    BASE_URL = "https://eodhistoricaldata.com/api/eod"

    def __init__(self, api_key: str):
        self.api_key = api_key

    @retry_request
    def get_stock_data(self, symbol: str):
        """Fetches stock data."""
        params = {
            "api_token": self.api_key,
            "fmt": "json"
        }
        response = requests.get(f"{self.BASE_URL}/{symbol}.US", params=params)
        response.raise_for_status()
        return response.json()


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


def aggregate_data(asset: str) -> dict:
    """Fetches data from multiple APIs.

    Args:
        asset (str): The name or symbol of the asset.

    Returns:
        dict: Aggregated data.
    """
    clients = [
        AlphaVantageClient(ALPHA_VANTAGE_API_KEY),
        IexCloudClient(IEX_CLOUD_API_KEY),
        YahooFinanceClient(YAHOO_FINANCE_API_KEY),
        QuandlClient(QUANDL_API_KEY),
        PolygonClient(POLYGON_API_KEY),
        EodHistoricalDataClient(EOD_HISTORICAL_API_KEY),
        # BloombergClient(BLOOMBERG_API_KEY)
    ]

    aggregated_data = {"asset": asset}

    for client in clients:
        client_name = client.__class__.__name__
        aggregated_data[client_name] = {
            "stock": client.get_stock_data(asset),
            "crypto": client.get_crypto_data(asset),
            "options": client.get_options_data(asset)
        }

    return aggregated_data