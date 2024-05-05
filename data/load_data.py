# Import necessary clients and utilities
from api.api_clients import aggregate_data  # Assuming aggregate_data retrieves raw asset data
from features.feature_engineering import FeatureEngineer
from data_processing import DataProcessor
from sklearn.model_selection import train_test_split

# Function to load and preprocess data based on asset and type
def load_data(asset: str, asset_type: str):
    # Retrieve the raw data for the given asset
    raw_data = aggregate_data(asset, asset_type)

    # Feature engineering: Apply technical indicators, etc.
    feature_engineer = FeatureEngineer()
    processed_data = feature_engineer.create_features(raw_data)

    # Data processing: Handle missing values and normalize features
    data_processor = DataProcessor()
    processed_data = data_processor.clean_and_normalize(processed_data)

    # Split into training and test sets (80-20 split, customize if needed)
    X = processed_data.drop(columns=["target"])  # Replace "target" with the actual target column name
    y = processed_data["target"]  # Replace "target" with the actual target column name
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
