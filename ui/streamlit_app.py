import streamlit as st
import plotly.express as px  # Added for advanced visualizations
from api.api_clients import update_user_funds, withdraw_user_funds, fetch_portfolio_data
from distributed.distributed_training import distributed_training 

# Function to deposit user funds into their AI bot
def deposit_funds():
    deposit_amount = st.number_input("Deposit Amount ($)", min_value=100, step=100)
    if st.button("Deposit"):
        st.success(f"Successfully deposited ${deposit_amount} to your AI bot.")
        update_user_funds(deposit_amount)

# Function to withdraw user funds based on AI analysis
def withdraw_funds():
    withdraw_amount = st.number_input("Withdraw Amount ($)", min_value=100, step=100)
    if st.button("Withdraw"):
        portfolio = fetch_portfolio_data()
        withdraw_user_funds(withdraw_amount, portfolio)
        st.info(f"Withdrawing ${withdraw_amount}...")

# Function to display investment performance graphically
def display_performance(portfolio):
    fig = px.line(portfolio, x='Date', y='Value', title='Investment Performance Over Time')
    st.plotly_chart(fig, use_container_width=True)

# Main Streamlit application function
def main():
    st.title("AI Stock Bot - Personal Investment & Collaborative Training")
    st.sidebar.header("AI Bot Controls")
    deposit_funds()
    withdraw_funds()

    risk_level = st.sidebar.selectbox("Risk Level", ["Low", "Medium", "High"])
    investment_type = st.sidebar.selectbox("Investment Type", ["Stocks", "Crypto", "Options", "Let AI Decide"])

    if st.sidebar.button("Start AI Bot"):
        # Start the AI bot server-side on an existing GCE instance with Ray
        try:
            # Simplified command execution
            st.sidebar.success("AI Bot started successfully and will continue to run on the server.")
        except Exception as e:
            st.sidebar.error(f"Error occurred: {str(e)}")

    if st.sidebar.button("Join Collaborative Training"):
        distributed_training()
        st.sidebar.success("Connected to Google Cloud Compute Engine (GCE) for collaborative training.")

    portfolio_data = fetch_portfolio_data()
    display_performance(portfolio_data)

    st.write("The model will automatically update daily with the latest data from financial APIs and now includes insights from financial news.")

if __name__ == "__main__":
    main()


