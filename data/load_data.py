from api.api_clients import fetch_market_data, fetch_top_investors_data, fetch_news_data, fetch_social_media_data
from data.data_processing import clean_and_normalize_data
from features.feature_engineering import FeatureEngineeringPipeline 

def load_raw_data():
    """Load raw data from various market and investment sources."""
    # Fetch market data
    market_data = fetch_market_data()
    
    # Fetch data about top investors
    top_investors_data = fetch_top_investors_data()

    # NEW: Fetch news and social media sentiment data
    news_data = fetch_news_data()
    social_media_data = fetch_social_media_data()

    # Merge or concatenate data sources as required
    combined_data = merge_data_sources(market_data, top_investors_data, news_data, social_media_data)

    return combined_data

def merge_data_sources(market_data, investors_data):
    """Merge different data sources together for a unified dataset."""
    # Placeholder merging logic - adjust based on your specific data structure
    combined_data = market_data.merge(investors_data, on='common_key', how='inner')

    return combined_data

def get_processed_data():
    """Load, clean, process, and return features and targets."""
    # Step 1: Load raw data
    raw_data = load_raw_data()  # Adjust this to match your actual data loading method

    # Step 2: Clean and process raw data
    processed_data = clean_and_normalize_data(raw_data)

    # Step 3: Apply feature engineering
    pipeline = FeatureEngineeringPipeline()
    features, targets = pipeline.fit_transform(processed_data)

    return features, targets
