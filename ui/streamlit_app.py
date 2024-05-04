# file: stock_ai_bot/ui/streamlit_app.py

import streamlit as st
import pandas as pd
from features.feature_engineering import add_technical_indicators
from data.data_processing import clean_and_normalize_data
from models.model_training import advanced_grid_search_tune_model
from models.optuna_optimization import run_optuna_optimization
from api.api_clients import aggregate_stock_data

def main():
    # Streamlit page configuration
    st.set_page_config(page_title="Stock AI Bot", page_icon="📈", layout="wide")
    st.title("📈 Stock AI Bot Dashboard")

    # User inputs
    stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL)", "AAPL")
    capital = st.number_input("Enter Total Capital Available ($)", min_value=0.0, value=1000.0)
    risk_tolerance = st.selectbox("Investment Risk Tolerance", ["Low", "Medium", "High"])
    training_mode = st.radio("Choose Training Mode", ["Advanced Grid Search", "Optuna Optimization"])

    # Execute when the "Analyze" button is pressed
    if st.button("Analyze and Train"):
        st.write(f"Analyzing {stock_symbol} with ${capital} capital and {risk_tolerance} risk tolerance.")
        with st.spinner("Fetching and processing data..."):
            # Aggregate, clean, and enhance stock data
            aggregated_data = aggregate_stock_data(stock_symbol)
            cleaned_data = clean_and_normalize_data(aggregated_data)
            enhanced_data = add_technical_indicators(cleaned_data)

        if training_mode == "Advanced Grid Search":
            st.write("Performing Advanced Grid Search Tuning...")
            results = advanced_grid_search_tune_model(enhanced_data)
            evaluation_metrics = results["evaluation_metrics"]
            st.write("Model Evaluation Metrics (Grid Search):")
            st.json(evaluation_metrics)
        elif training_mode == "Optuna Optimization":
            st.write("Running Optuna Optimization...")
            run_optuna_optimization(enhanced_data)

if __name__ == "__main__":
    main()
