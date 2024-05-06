from google.cloud import storage

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
