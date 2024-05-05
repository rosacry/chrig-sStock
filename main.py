# Import necessary modules
from api.api_clients import PolygonClient, AlphaVantageClient
from data.load_data import DataLoader
from data.data_processing import DataProcessor
from features.feature_engineering import FeatureEngineer
from models.model_training import ModelTrainer
from models.model_tuning import ModelTuner
from optimization.unified_tuning import UnifiedTuner
from distributed.distributed_training import DistributedTraining
from ui.cli import UserInterface
from ui.streamlit_app import main as streamlit_main
from utils.logging import setup_logging

# Set up logging
logger = setup_logging()

# Initialize API clients
polygon_client = PolygonClient()
alpha_vantage_client = AlphaVantageClient()

# Initialize data loading and processing modules
data_loader = DataLoader()
data_processor = DataProcessor()

# Initialize feature engineering
feature_engineer = FeatureEngineer()

# Initialize training, tuning, and optimization components
model_trainer = ModelTrainer()
model_tuner = ModelTuner()
unified_tuner = UnifiedTuner()
distributed_training = DistributedTraining()

# Initialize CLI user interface
ui = UserInterface()

# Start the AI bot
def start_bot():
    try:
        # Prompt user to choose between CLI or Streamlit
        interface_choice = input("Choose your interface (1: CLI, 2: Streamlit): ").strip()

        if interface_choice == "1":
            # CLI Interface
            logger.info("Using CLI interface.")
            # Fetch data using clients and data loader
            stock_raw_data = polygon_client.get_stock_data()
            crypto_raw_data = alpha_vantage_client.get_crypto_data()

            # Load additional data if required
            extra_data = data_loader.load_additional_data()
            logger.info("Loaded additional data for analysis.")

            # Process data
            stock_data = data_processor.process(stock_raw_data)
            crypto_data = data_processor.process(crypto_raw_data)

            # Generate features
            stock_features = feature_engineer.generate(stock_data)
            crypto_features = feature_engineer.generate(crypto_data)

            # Train models
            stock_model = model_trainer.train(stock_features)
            crypto_model = model_trainer.train(crypto_features)

            # Tune models
            tuned_stock_model = model_tuner.tune(stock_model)
            tuned_crypto_model = model_tuner.tune(crypto_model)

            # Optimize models
            optimized_stock_model = unified_tuner.optimize(tuned_stock_model)
            optimized_crypto_model = unified_tuner.optimize(tuned_crypto_model)

            # Enable distributed training if needed
            distributed_training.train_with_contributors(optimized_stock_model, optimized_crypto_model)

            # Interact with user via CLI
            ui.interact(optimized_stock_model, optimized_crypto_model)

        elif interface_choice == "2":
            # Streamlit Interface
            logger.info("Using Streamlit interface.")
            streamlit_main()

        else:
            logger.error("Invalid choice. Please choose either 1 (CLI) or 2 (Streamlit).")

    except Exception as e:
        logger.error(f"Error encountered: {e}")
        raise

# Run the bot
if __name__ == "__main__":
    start_bot()
