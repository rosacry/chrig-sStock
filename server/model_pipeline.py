from google.cloud import storage
from data_processing import clean_and_process_raw_data
from feature_engineering import FeatureEngineeringPipeline
from load_data import load_raw_data


def upload_model(bucket_name, source_file_name, destination_blob_name):
    """Uploads a model file to the GCS bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"Model {source_file_name} uploaded to {bucket_name}/{destination_blob_name}.")

def download_model(bucket_name, source_blob_name, destination_file_name):
    """Downloads a model file from the GCS bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Model {bucket_name}/{source_blob_name} downloaded to {destination_file_name}.")

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

