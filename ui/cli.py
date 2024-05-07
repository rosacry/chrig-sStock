import argparse
from rich.console import Console
from tqdm import tqdm
from features.feature_engineering import FeatureEngineer
from data.data_processing import DataProcessor
from models.model_training import ModelTrainer
from models.model_tuning import ModelTuner
from optimization.unified_tuning import UnifiedTuner
from distributed.distributed_training import DistributedTraining
import pandas as pd
import inquirer  # Added for interactive features

# Initialize components
fe = FeatureEngineer()
mt = ModelTrainer()
mtu = ModelTuner()
ut = UnifiedTuner()
dt = DistributedTraining()
console = Console()

class UserInterface:
    def __init__(self):
        self.console = console

    def display_welcome_message(self):
        self.console.print("[bold cyan]Welcome to Stock AI Bot CLI[/bold cyan]")

    def configure_arguments(self):
        questions = [
            inquirer.Text('symbol', message="Enter the stock symbol to analyze (e.g., AAPL)"),
            inquirer.List('risk', message="Select your investment risk tolerance", choices=['low', 'medium', 'high']),
            inquirer.Text('capital', message="Enter total capital available for investment"),
            inquirer.List('asset', message="Select the asset type to invest in", choices=['stocks', 'crypto', 'options', 'auto'])
        ]
        return inquirer.prompt(questions)

    def main(self):
        args = this.configure_arguments()
        this.display_welcome_message()

        # Fetch the stock symbol and aggregate its data
        symbol = args['symbol']
        this.console.print(f"\n[bold]Analyzing {symbol} with {args['capital']} capital and {args['risk']} risk tolerance...[/bold]")
        with this.console.status("[bold green]Fetching and processing data...[/bold]") as status:
            from api.api_clients import aggregate_stock_data
            aggregated_data = aggregate_stock_data(symbol)
            cleaned_data = DataProcessor.clean_and_normalize_data(aggregated_data)
            enhanced_data = fe.add_technical_indicators(cleaned_data)

        with tqdm(total=100, desc="Training Progress", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as progress_bar:
            progress_bar.set_description("Training the Model")
            trained_model = mt.train(enhanced_data)
            progress_bar.update(40)

            progress_bar.set_description("Tuning the Model")
            tuned_model = mtu.tune(trained_model)
            progress_bar.update(30)

            progress_bar.set_description("Optimizing the Model")
            optimized_model = ut.unified_tuning(tuned_model)
            progress_bar.update(30)

            this.console.print("[bold]Model Training, Tuning, and Optimization completed successfully![/bold]")

        # Execute distributed training
        try:
            dt.run_distributed_training(pd.concat([enhanced_data], axis=1), optimized_model)
            this.console.print("[bold]Successfully completed the distributed training session.[/bold]")
        except Exception as e:
            this.console.print(f"[bold red]Error during distributed training: {e}[/bold red]")

if __name__ == "__main__":
    ui = UserInterface()
    ui.main()

