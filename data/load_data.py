from api.api_clients import fetch_market_data, fetch_top_investors_data, fetch_news_data, fetch_social_media_data
from features.feature_engineering import FeatureEngineeringPipeline 
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

def process_individual_data(data, key):
    try:
        df = pd.DataFrame(data)
        df['source'] = key
        return df
    except ValueError:
        print(f"Error processing data from source {key}. Skipping...")
        return pd.DataFrame()

def clean_and_normalize_data(aggregated_data: dict):
    """Clean and normalize aggregated stock data.

    Args:
        aggregated_data (dict): Raw data from various APIs and scraped data.
    
    Returns:
        pd.DataFrame: Cleaned and normalized data.
    """
    # Parallel processing of data from different sources
    data_frames = Parallel(n_jobs=-1)(delayed(process_individual_data)(data, key) 
                                      for key, data in aggregated_data.items() if key != "symbol")
    
    combined_df = pd.concat(data_frames, ignore_index=True, sort=False)

    # Handle missing values
    combined_df.dropna(axis=0, thresh=int(0.5 * combined_df.shape[1]), inplace=True)
    combined_df.fillna(method='ffill', inplace=True)

    # Normalize numerical columns
    numerical_cols = combined_df.select_dtypes(include=np.number).columns
    combined_df[numerical_cols] = (combined_df[numerical_cols] - combined_df[numerical_cols].mean()) / combined_df[numerical_cols].std()

    return combined_df

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
