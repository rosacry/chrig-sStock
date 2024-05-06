import os
import subprocess
import argparse

def run_cli():
    # Execute the CLI interface via Python subprocess
    subprocess.run(["python", "cli.py"])

def run_streamlit():
    # Execute the Streamlit app via subprocess
    subprocess.run(["streamlit", "run", "streamlit_app.py"])

def main():
    # Setup argument parser to select the UI type
    parser = argparse.ArgumentParser(description="Stock AI Bot Main Application")
    parser.add_argument("--interface", type=str, choices=["cli", "gui"], required=True,
                        help="Choose the interface to start the application ('cli' or 'gui').")
    args = parser.parse_args()

    # Launch the desired interface
    if args.interface == "cli":
        run_cli()
    elif args.interface == "gui":
        run_streamlit()

if __name__ == "__main__":
    main()