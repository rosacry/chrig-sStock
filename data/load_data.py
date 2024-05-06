from api_clients import fetch_market_data, fetch_top_investors_data

def load_raw_data():
    """Load raw data from various market and investment sources."""
    # Fetch market data (adjust this function as per your API client methods)
    market_data = fetch_market_data()
    
    # Fetch data about top investors (if needed)
    top_investors_data = fetch_top_investors_data()

    # Merge or concatenate data sources as required
    combined_data = merge_data_sources(market_data, top_investors_data)

    return combined_data

def merge_data_sources(market_data, investors_data):
    """Merge different data sources together for a unified dataset."""
    # Placeholder merging logic - adjust based on your specific data structure
    combined_data = market_data.merge(investors_data, on='common_key', how='inner')

    return combined_data
