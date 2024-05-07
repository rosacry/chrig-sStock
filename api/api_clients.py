import os
import requests
import cachetools.func  

# Market API keys
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
IEX_CLOUD_API_KEY = os.getenv('IEX_CLOUD_API_KEY')
QUANDL_API_KEY = os.getenv('QUANDL_API_KEY')
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
EOD_HISTORICAL_API_KEY = os.getenv('EOD_HISTORICAL_API_KEY')

# Cache for 10 minutes
@cachetools.func.ttl_cache(maxsize=100, ttl=600)
def get_cached_response(url, params):
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def fetch_market_data():
    """Fetch market data from multiple APIs using their respective API keys."""
    all_market_data = {}

    # Alpha Vantage
    params_alpha = {"apikey": ALPHA_VANTAGE_API_KEY}
    try:
        all_market_data["alpha_vantage"] = get_cached_response("https://www.alphavantage.co/query", params_alpha)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Alpha Vantage: {str(e)}")
        all_market_data["alpha_vantage"] = None

    # IEX Cloud
    headers_iex = {"Authorization": f"Bearer {IEX_CLOUD_API_KEY}"}
    try:
        all_market_data["iex_cloud"] = get_cached_response("https://cloud.iexapis.com/stable", headers_iex)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from IEX Cloud: {str(e)}")
        all_market_data["iex_cloud"] = None

    # Quandl
    params_quandl = {"api_key": QUANDL_API_KEY}
    try:
        all_market_data["quandl"] = get_cached_response("https://www.quandl.com/api/v3/datasets", params_quandl)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Quandl: {str(e)}")
        all_market_data["quandl"] = None

    # Polygon
    headers_polygon = {"Authorization": f"Bearer {POLYGON_API_KEY}"}
    try:
        all_market_data["polygon"] = get_cached_response("https://api.polygon.io/v2", headers_polygon)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Polygon: {str(e)}")
        all_market_data["polygon"] = None

    # EOD Historical
    params_eod = {"api_token": EOD_HISTORICAL_API_KEY}
    try:
        all_market_data["eod_historical"] = get_cached_response("https://eodhistoricaldata.com/api", params_eod)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from EOD Historical: {str(e)}")
        all_market_data["eod_historical"] = None

    return all_market_data
