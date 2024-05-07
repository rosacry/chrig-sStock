import argparse
from rich.console import Console
from tqdm import tqdm
from features.feature_engineering import FeatureEngineer
from data.data_processing import DataProcessor
from models.model_training import ModelTrainer
from models.model_tuning import ModelTuner
from distributed.distributed_training import DistributedTraining
import pandas as pd
import inquirer
import json

# Initialize components
fe = FeatureEngineer()
mt = ModelTrainer()
mtu = ModelTuner()
dt = DistributedTraining()
console = Console()

class UserInterface:
    def __init__(self):
        self.console = console

    def display_welcome_message(self):
        self.console.print("[bold cyan]Welcome to the Stock AI Bot CLI[/bold cyan]")

    def configure_arguments(self):
        questions = [
            inquirer.Text('symbol', message="Enter the stock symbol to analyze (e.g., AAPL)"),
            inquirer.List('risk', message="Select your investment risk tolerance", choices=['low', 'medium', 'high', 'speculative']),
            inquirer.Text('capital', message="Enter total capital available for investment"),
            inquirer.List('asset', message="Select the asset type to invest in", choices=['stocks', 'crypto', 'options', 'auto']),
            inquirer.Confirm('contribute_compute', message="Would you like to contribute compute power for distributed training?")
        ]
        return inquirer.prompt(questions)

    def main(self):
        args = self.configure_arguments()
        self.display_welcome_message()
        self.run_full_stock_analysis(args)

    def run_full_stock_analysis(self, args):
        symbol = args['symbol']
        self.console.print(f"\n[bold]Starting analysis for {symbol}...[/bold]")

        # Fetch and process data
        with self.console.status("[bold green]Fetching and processing data...[/bold]") as status:
            # Simulate data aggregation
            data = {'data': 'Simulated stock data'}
            cleaned_data = DataProcessor.clean_and_normalize_data(json.dumps(data))
            enhanced_data = fe.add_technical_indicators(pd.DataFrame(cleaned_data))

        # Machine learning operations
        with tqdm(total=100, desc="Model Operations", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as progress_bar:
            trained_model = mt.train(enhanced_data)
            progress_bar.update(40)
            tuned_model = mtu.tune(trained_model)
            progress_bar.update(30)
            optimized_model = mtu.optimize(tuned_model)
            progress_bar.update(30)

        # Distributed training
        if args['contribute_compute']:
            try:
                dt.run_distributed_training(enhanced_data, optimized_model)
                self.console.print("[bold]Distributed training completed successfully![/bold]")
            except Exception as e:
                self.console.print(f"[bold red]Error during distributed training: {e}[/bold red]")

        # Autonomous trading decision
        self.make_trading_decision(optimized_model, args)

    def make_trading_decision(self, model, args):
        # Placeholder for trading logic
        self.console.print("[bold]Making trading decisions based on the model...[/bold]")
        # Simulated decision-making process
        self.console.print(f"[bold green]Buy 10 shares of {args['symbol']} based on AI recommendation.[/bold]")

if __name__ == "__main__":
    ui = UserInterface()
    ui.main()

