import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from .web_scraping import scrape_financial_news

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

