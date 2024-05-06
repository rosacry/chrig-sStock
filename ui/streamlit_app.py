# /mnt/data/streamlit_app.py

import streamlit as st
from api_clients import update_user_funds, withdraw_user_funds, fetch_portfolio_data
from distributed.distributed_training import launch_ray_tune_study
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Function to deposit user funds into their AI bot
def deposit_funds():
    """Allow user to deposit funds into their AI bot."""
    deposit_amount = st.number_input("Deposit Amount ($)", min_value=100, step=100)
    if st.button("Deposit"):
        st.write(f"Successfully deposited ${deposit_amount} to your AI bot.")
        update_user_funds(deposit_amount)


# Function to withdraw user funds based on AI analysis
def withdraw_funds():
    """Allow user to withdraw funds, letting the AI decide what to sell."""
    withdraw_amount = st.number_input("Withdraw Amount ($)", min_value=100, step=100)
    if st.button("Withdraw"):
        st.write(f"Withdrawing ${withdraw_amount}...")
        portfolio = fetch_portfolio_data()
        withdraw_user_funds(withdraw_amount, portfolio)


# Function to capture risk level preference
def set_risk_level():
    """Allow user to set their preferred risk level."""
    risk_level = st.selectbox("Risk Level", ["Low", "Medium", "High"])
    st.write(f"Risk Level set to: {risk_level}")
    return risk_level


# Function to capture investment type preference
def set_investment_type():
    """Allow user to choose investment type or let AI decide."""
    investment_type = st.selectbox("Investment Type", ["Stocks", "Crypto", "Options", "Let AI Decide"])
    st.write(f"Investment Type set to: {investment_type}")
    return investment_type


# Create credentials object using your service account JSON
def create_gce_compute_client(service_account_json):
    credentials = service_account.Credentials.from_service_account_file(service_account_json)
    return build('compute', 'v1', credentials=credentials)

# Execute a command to start the AI bot on the GCE instance
def execute_gce_command(project, zone, instance, command):
    """Use the Compute API to execute a command on a GCE instance."""
    client = create_gce_compute_client('YOUR_SERVICE_ACCOUNT_JSON_PATH')
    request = client.instances().start(project=project, zone=zone, instance=instance)
    response = request.execute()
    return response

# Function to start the AI bot using the Google Compute API
def start_ai_bot():
    """Start the AI bot server-side on an existing GCE instance with Ray."""
    project = "YOUR_PROJECT_ID"
    zone = "YOUR_GCE_INSTANCE_ZONE"
    instance = "YOUR_GCE_INSTANCE_NAME"
    command = "bash /path/to/start_ai_bot.sh"

    if st.button("Start AI Bot"):
        try:
            # Start the instance (if it's not running already)
            response = execute_gce_command(project, zone, instance, command)
            st.write(f"Response: {response}")
            st.write("AI Bot started successfully and will continue to run on the server.")
        except Exception as e:
            st.write(f"Error occurred: {str(e)}")


# Function to connect user to collaborative training
def join_collaborative_training():
    """Connect to GCE with Ray for collaborative training."""
    if st.button("Join Collaborative Training"):
        st.write("Connecting to Google Cloud Compute Engine (GCE) for collaborative training.")
        launch_ray_tune_study()


# Main Streamlit application function
def main():
    st.title("AI Stock Bot - Personal Investment & Collaborative Training")

    st.write("**Manage Your AI Stock Bot**")
    deposit_funds()
    withdraw_funds()

    st.write("**Set Preferences**")
    risk_level = set_risk_level()
    investment_type = set_investment_type()

    st.write("**Start Your AI Bot**")
    start_ai_bot()

    st.write("**Join Collaborative Training**")
    join_collaborative_training()

    st.write("The model will automatically update daily with the latest data from financial APIs.")
    # Ensure data fetching and retraining mechanisms work daily without user intervention


if __name__ == "__main__":
    main()


