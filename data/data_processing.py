import numpy as np
import pandas as pd

class DataProcessor:
    @staticmethod
    def clean_and_normalize_data(aggregated_data: dict):
        """Clean and normalize aggregated stock data.

        Args:
            aggregated_data (dict): Raw data from various APIs.
        
        Returns:
            pd.DataFrame: Cleaned and normalized data.
        """
        # Extract relevant sub-dataframes from aggregated data
        data_frames = []
        for key, data in aggregated_data.items():
            if key == "symbol":
                continue

            try:
                # Convert JSON data to DataFrame
                df = pd.DataFrame(data)
                df["source"] = key  # Track data source for reference
                data_frames.append(df)
            except ValueError:
                print(f"Error processing data from source {key}. Skipping...")

        # Concatenate all frames into a single DataFrame
        combined_df = pd.concat(data_frames, ignore_index=True, sort=False)

        # Drop rows with significant missing values
        combined_df.dropna(axis=0, thresh=int(0.5 * combined_df.shape[1]), inplace=True)

        # Fill remaining missing values with the median
        for col in combined_df.select_dtypes(include=np.number).columns:
            combined_df[col].fillna(combined_df[col].median(), inplace=True)

        # Normalize numerical columns
        numerical_cols = combined_df.select_dtypes(include=np.number).columns
        combined_df[numerical_cols] = (combined_df[numerical_cols] - combined_df[numerical_cols].mean()) / combined_df[numerical_cols].std()

        return combined_df
